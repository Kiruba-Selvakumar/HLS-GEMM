/**
 * Host code for Systolic Matrix Multiplication
 * Target Kernel: mmult_systolic
 * API: XRT Native C++ (experimental/xrt_*.h)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <string>
#include <algorithm>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// Default Parameters
static const int M_default = 256;
static const int N_default = 256;
static const int P_default = 256;
static const int Device_default = 1;

// Simple helper to parse command line args since we don't have the external header
struct CmdArgs {
    std::string xclbin_file;
    int device_index = Device_default;
    int M = M_default;
    int N = N_default;
    int P = P_default;
    bool valid = false;
};

CmdArgs parse_args(int argc, char** argv) {
    CmdArgs args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-x" || arg == "--xclbin_file") && i + 1 < argc) {
            args.xclbin_file = argv[++i];
        } else if ((arg == "-d" || arg == "--device_id") && i + 1 < argc) {
            args.device_index = std::stoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--M") && i + 1 < argc) {
            args.M = std::stoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--N") && i + 1 < argc) {
            args.N = std::stoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--P") && i + 1 < argc) {
            args.P = std::stoi(argv[++i]);
        }
    }
    if (!args.xclbin_file.empty()) args.valid = true;
    return args;
}

int main(int argc, char** argv) {
    // 1. Parse Arguments
    CmdArgs args = parse_args(argc, argv);
    if (!args.valid) {
        std::cout << "Usage: " << argv[0] << " -x <xclbin_file> [-d <device_id>] [-m <rows_A>] [-n <cols_A>] [-p <cols_B>]" << std::endl;
        return EXIT_FAILURE;
    }

    int M = args.M;
    int N = args.N;
    int P = args.P;

    std::cout << "------------------------------------------------\n";
    std::cout << "Systolic Matrix Mult Host (XRT Native API)\n";
    std::cout << "------------------------------------------------\n";
    std::cout << "Device index : " << args.device_index << std::endl;
    std::cout << "Xclbin       : " << args.xclbin_file << std::endl;
    std::cout << "Matrix Dims  : " << M << "x" << N << " * " << N << "x" << P << std::endl;

    // 2. Open Device and Load Binary
    std::cout << "Opening device..." << std::endl;
    auto device = xrt::device(args.device_index);

    std::cout << "Loading xclbin..." << std::endl;
    auto uuid = device.load_xclbin(args.xclbin_file);

    // 3. Create Kernel Handle
    // Note: The kernel name must match the function name in the HLS code
    std::cout << "Creating kernel handle 'mmult_systolic'..." << std::endl;
    auto krnl = xrt::kernel(device, uuid, "mmult_systolic");

    // 4. Allocate Buffer Objects (BOs)
    // Size in bytes
    const size_t size_a_bytes = sizeof(float) * M * N;
    const size_t size_b_bytes = sizeof(float) * N * P;
    const size_t size_c_bytes = sizeof(float) * M * P;

    // Allocate BOs in the memory bank corresponding to the kernel argument
    // arg 0 = A, arg 1 = B, arg 2 = C
    std::cout << "Allocating buffers..." << std::endl;
    auto boA = xrt::bo(device, size_a_bytes, krnl.group_id(0));
    auto boB = xrt::bo(device, size_b_bytes, krnl.group_id(1));
    auto boC = xrt::bo(device, size_c_bytes, krnl.group_id(2));

    // 5. Map Buffers to Host Pointers
    auto A = boA.map<float*>();
    auto B = boB.map<float*>();
    auto C = boC.map<float*>();

    // 6. Initialize Data
    // We use a deterministic pattern to make verification easy and consistent
    std::cout << "Initializing matrices..." << std::endl;
    
    // Matrix A (Row Major)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < N; ++k) {
            A[i * N + k] = static_cast<float>((i + k) % 8); 
        }
    }
    
    // Matrix B (Row Major)
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < P; ++j) {
            B[k * P + j] = static_cast<float>((k + j) % 8) * 0.5f; 
        }
    }
    
    // Matrix C (Output - Clear it)
    std::fill(C, C + (M * P), 0.0f);

    // 7. Compute Golden Reference (CPU)
    std::cout << "Computing CPU reference..." << std::endl;
    std::vector<float> C_ref(M * P, 0.0f);
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * P + j];
            }
            C_ref[i * P + j] = sum;
        }
    }
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(t_cpu_end - t_cpu_start).count();
    std::cout << "CPU time: " << cpu_time << " us" << std::endl;

    // 8. Transfer Data to Device
    std::cout << "Syncing buffers to device..." << std::endl;
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 9. Execute Kernel
    std::cout << "Running FPGA kernel..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // XRT Run object: arguments must match the kernel signature
    // void mmult_systolic(const float *A, const float *B, float *C, int M, int N, int P)
    auto run = krnl(boA, boB, boC, M, N, P);
    
    // Wait for kernel to finish
    run.wait();
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "FPGA execution time: " << elapsed_us << " us" << std::endl;

    // 10. Transfer Results back to Host
    std::cout << "Reading results..." << std::endl;
    boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // 11. Validate Results
    std::cout << "Validating..." << std::endl;
    int errors = 0;
    float epsilon = 0.01f;

    for (int idx = 0; idx < M * P; ++idx) {
        float diff = std::abs(C[idx] - C_ref[idx]);
        if (diff > epsilon) {
            if (errors < 10) { // Limit error output
                int i = idx / P;
                int j = idx % P;
                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << "FPGA=" << std::fixed << std::setprecision(4) << C[idx]
                          << ", REF=" << C_ref[idx] 
                          << ", Diff=" << diff << std::endl;
            }
            ++errors;
        }
    }

    if (errors) {
        std::cerr << "TEST FAILED with " << errors << " mismatches\n";
    } else {
        std::cout << "TEST PASSED!\n";
    }

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}