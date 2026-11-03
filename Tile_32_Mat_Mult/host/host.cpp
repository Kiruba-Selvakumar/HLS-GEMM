/**
 * Host code for int16 matrix-multiply kernel "mmult"
 * Compatible with runtime-selectable matrix sizes (up to 256x256)
 */

#include "cmdlineparser.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdint>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// Match kernel compile-time sizes
static constexpr int M_default = 256;
static constexpr int N_default = 256;
static constexpr int P_default = 256;

int main(int argc, char** argv) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "1");
    parser.addSwitch("--M", "-m", "rows of A and C", std::to_string(M_default));
    parser.addSwitch("--N", "-n", "columns of A / rows of B", std::to_string(N_default));
    parser.addSwitch("--P", "-p", "columns of B and C", std::to_string(P_default));
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = std::stoi(parser.value("device_id"));
    int M = std::stoi(parser.value("M"));
    int N = std::stoi(parser.value("N"));
    int P = std::stoi(parser.value("P"));

    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Device index: " << device_index << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << " * " << N << "x" << P << std::endl;

    // Open device and load xclbin
    std::cout << "Opening device..." << std::endl;
    auto device = xrt::device(device_index);

    std::cout << "Loading xclbin: " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    // Kernel handle
    auto krnl = xrt::kernel(device, uuid, "mmult");

    // Buffer sizes in bytes
    const size_t size_a_bytes = sizeof(int16_t) * M * N;
    const size_t size_b_bytes = sizeof(int16_t) * N * P;
    const size_t size_c_bytes = sizeof(int32_t) * M * P;

    // Allocate buffers
    std::cout << "Allocating buffers..." << std::endl;
    auto boA = xrt::bo(device, size_a_bytes, krnl.group_id(0));
    auto boB = xrt::bo(device, size_b_bytes, krnl.group_id(1));
    auto boC = xrt::bo(device, size_c_bytes, krnl.group_id(2));

    // Map buffers to host memory
    auto A = boA.map<int16_t*>();
    auto B = boB.map<int16_t*>();
    auto C = boC.map<int32_t*>();

    // Initialize input data
    std::cout << "Initializing matrices..." << std::endl;

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < N; ++k) {
            A[i * N + k] = static_cast<int16_t>((i + k) % 8); // pattern 0..7
        }
    }
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < P; ++j) {
            B[k * P + j] = static_cast<int16_t>((k + j) % 8); // pattern 0..7
        }
    }
    std::fill(C, C + (M * P), 0);

    // Build CPU reference result
    std::vector<int32_t> C_ref(M * P, 0);
    std::cout << "Computing reference on CPU..." << std::endl;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += (int32_t)A[i * N + k] * (int32_t)B[k * P + j];
            }
            C_ref[i * P + j] = sum;
        }
    }

    // Synchronize input buffers to device
    std::cout << "Copying inputs to device global memory..." << std::endl;
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Launch kernel
    std::cout << "Running kernel..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto run = krnl(boA, boB, boC, M, N, P);
    run.wait();
    auto t2 = std::chrono::high_resolution_clock::now();

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Kernel execution time: " << elapsed_us << " us" << std::endl;

    // Get output back from device
    std::cout << "Reading results back..." << std::endl;
    boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Validate results
    std::cout << "Validating results..." << std::endl;
    int errors = 0;
    for (int idx = 0; idx < M * P; ++idx) {
        if (C[idx] != C_ref[idx]) {
            if (errors < 10) { // show first few mismatches
                int i = idx / P;
                int j = idx % P;
                std::cerr << "Mismatch at (" << i << "," << j << "): FPGA=" << C[idx]
                          << ", REF=" << C_ref[idx] << std::endl;
            }
            ++errors;
        }
    }

    if (errors)
        std::cerr << "TEST FAILED with " << errors << " mismatches\n";
    else
        std::cout << "TEST PASSED\n";

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

