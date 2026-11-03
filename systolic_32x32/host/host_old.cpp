/**
* Host for 32x32 float matrix-multiply kernel "mmult"
* Adapts your vector-add host to 2D matrices (flattened row-major).
*/

#include "cmdlineparser.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// Match kernel compile-time sizes
static constexpr int M = 32;
static constexpr int N = 32;
static constexpr int P = 32;

int main(int argc, char** argv) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id",   "-d", "device index", "1");
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = std::stoi(parser.value("device_id"));

    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);

    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    // Kernel handle: symbol name must match the HLS kernel top function
    // extern "C" void mmult_base(const float A[M][N], const float B[N][P], float C[M][P]);
    auto krnl = xrt::kernel(device, uuid, "mmult_base");

    // Buffer sizes (in bytes) for row-major flattened arrays
    const size_t size_a_bytes = sizeof(float) * M * N;
    const size_t size_b_bytes = sizeof(float) * N * P;
    const size_t size_c_bytes = sizeof(float) * M * P;

    std::cout << "Allocate buffers in device global memory\n";
    auto boA = xrt::bo(device, size_a_bytes, krnl.group_id(0)); // A
    auto boB = xrt::bo(device, size_b_bytes, krnl.group_id(1)); // B
    auto boC = xrt::bo(device, size_c_bytes, krnl.group_id(2)); // C (output)

    // Map the contents of the buffer objects into host memory
    auto A = boA.map<float*>();
    auto B = boB.map<float*>();
    auto C = boC.map<float*>();

    // Initialize inputs; zero the output buffer
    std::fill(C, C + (M * P), 0.0f);

    // Example deterministic test pattern (easy to verify and stable across runs)
    // A[i,k] = i + 0.1*k, B[k,j] = k + 0.2*j
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < N; ++k) {
            A[i * N + k] = static_cast<float>(i) + 0.1f * static_cast<float>(k);
        }
    }
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < P; ++j) {
            B[k * P + j] = static_cast<float>(k) + 0.2f * static_cast<float>(j);
        }
    }

    // Build a CPU reference result for validation
    std::vector<float> C_ref(M * P, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * P + j]; // row-major indexing
            }
            C_ref[i * P + j] = sum;
        }
    }

    // Synchronize input buffers to the device
    std::cout << "Synchronize input buffers (A,B) to device global memory\n";
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Launch the kernel. For the fixed-size HLS signature there is no size argument.
    std::cout << "Execution of the kernel\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    auto run = krnl(boA, boB, boC);
    run.wait();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed_us =
    std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Kernel execution time (host measured): "
          << elapsed_us << " us" << std::endl;

    // Get the output back from device
    std::cout << "Get the output data from the device\n";
    boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Validate with a small epsilon (float math)
    int errors = 0;
    const float eps = 1e-3f;
    for (int idx = 0; idx < M * P; ++idx) {
        float diff = std::fabs(C[idx] - C_ref[idx]);
        if (diff > eps) {
            if (errors < 10) { // print only a few mismatches
                int i = idx / P;
                int j = idx % P;
                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << "FPGA=" << C[idx] << " REF=" << C_ref[idx]
                          << " |diff|=" << diff << "\n";
            }
            ++errors;
        }
    }

    if (errors) {
        std::cerr << "TEST FAILED with " << errors << " mismatches\n";
        return EXIT_FAILURE;
    }

    std::cout << "TEST PASSED\n";
    return 0;
}

