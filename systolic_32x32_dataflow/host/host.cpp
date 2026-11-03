/**
 * Host code for Systolic Matrix Multiplication
 * Target Kernel: mmult_systolic
 * API: XRT Native C++ (experimental/xrt_*.h)
 * * MERGED VERSION:
 * 1. Includes Memory Tiling (Critical for correctness)
 * 2. Includes CPU Timing & Verification details
 * 3. Skips CPU verification for large matrices (>1024)
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

// Constants
static const int SA_DIM = 32; // Systolic Array Dimension

// Default Parameters
static const int M_default = 256;
static const int N_default = 256;
static const int P_default = 256;
static const int Device_default = 1;

// -------------------------------------------------------------------------
// Helper: Convert Linear Row-Major -> Block Tiled
// -------------------------------------------------------------------------
void convert_to_tiled(const std::vector<float>& linear_data, std::vector<float>& tiled_data, int Rows, int Cols) {
    int num_tiles_R = Rows / SA_DIM;
    int num_tiles_C = Cols / SA_DIM;
    
    int tiled_idx = 0;
    
    // Iterate through tiles
    for (int tr = 0; tr < num_tiles_R; ++tr) {
        for (int tc = 0; tc < num_tiles_C; ++tc) {
            
            // Iterate inside the 32x32 tile
            for (int r = 0; r < SA_DIM; ++r) {
                for (int c = 0; c < SA_DIM; ++c) {
                    // Global coordinates
                    int global_r = tr * SA_DIM + r;
                    int global_c = tc * SA_DIM + c;
                    
                    // Linear Index = Row * Width + Col
                    int linear_idx = global_r * Cols + global_c;
                    
                    tiled_data[tiled_idx++] = linear_data[linear_idx];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Helper: Convert Block Tiled -> Linear Row-Major
// -------------------------------------------------------------------------
void convert_from_tiled(const std::vector<float>& tiled_data, std::vector<float>& linear_data, int Rows, int Cols) {
    int num_tiles_R = Rows / SA_DIM;
    int num_tiles_C = Cols / SA_DIM;
    
    int tiled_idx = 0;
    
    for (int tr = 0; tr < num_tiles_R; ++tr) {
        for (int tc = 0; tc < num_tiles_C; ++tc) {
            for (int r = 0; r < SA_DIM; ++r) {
                for (int c = 0; c < SA_DIM; ++c) {
                    int global_r = tr * SA_DIM + r;
                    int global_c = tc * SA_DIM + c;
                    
                    int linear_idx = global_r * Cols + global_c;
                    linear_data[linear_idx] = tiled_data[tiled_idx++];
                }
            }
        }
    }
}

// Command Line Argument Parser
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

    // Validate Dimensions
    if (M % SA_DIM != 0 || N % SA_DIM != 0 || P % SA_DIM != 0) {
        std::cerr << "Error: M, N, P must be multiples of " << SA_DIM << " for this architecture." << std::endl;
        return EXIT_FAILURE;
    }

    // DECISION: Skip CPU verification for large matrices
    bool check_cpu_ref = (M <= 1024 && N <= 1024 && P <= 1024);

    std::cout << "------------------------------------------------\n";
    std::cout << "Systolic Matrix Mult Host (XRT Native API)\n";
    std::cout << "------------------------------------------------\n";
    std::cout << "Device index : " << args.device_index << std::endl;
    std::cout << "Xclbin       : " << args.xclbin_file << std::endl;
    std::cout << "Matrix Dims  : " << M << "x" << N << " * " << N << "x" << P << std::endl;
    std::cout << "CPU Verify   : " << (check_cpu_ref ? "ENABLED" : "DISABLED (>1024)") << std::endl;

    // 2. Open Device and Load Binary
    std::cout << "Opening device..." << std::endl;
    auto device = xrt::device(args.device_index);

    std::cout << "Loading xclbin..." << std::endl;
    auto uuid = device.load_xclbin(args.xclbin_file);

    // 3. Create Kernel Handle
    std::cout << "Creating kernel handle 'mmult_systolic'..." << std::endl;
    auto krnl = xrt::kernel(device, uuid, "mmult_systolic");

    // 4. Allocate Buffer Objects (BOs)
    const size_t size_a_bytes = sizeof(float) * M * N;
    const size_t size_b_bytes = sizeof(float) * N * P;
    const size_t size_c_bytes = sizeof(float) * M * P;

    std::cout << "Allocating buffers..." << std::endl;
    auto boA = xrt::bo(device, size_a_bytes, krnl.group_id(0));
    auto boB = xrt::bo(device, size_b_bytes, krnl.group_id(1));
    auto boC = xrt::bo(device, size_c_bytes, krnl.group_id(2));

    // 5. Map Buffers to Host Pointers
    auto A_mapped = boA.map<float*>();
    auto B_mapped = boB.map<float*>();
    auto C_mapped = boC.map<float*>();

    // 6. Initialize Data
    // NOTE: We initialize LINEAR vectors first, then convert to Tiled for HW
    std::cout << "Initializing matrices..." << std::endl;
    
    std::vector<float> A_linear(M * N);
    std::vector<float> B_linear(N * P);

    // Matrix A Initialization (Your pattern)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < N; ++k) {
            A_linear[i * N + k] = static_cast<float>((i + k) % 8); 
        }
    }
    
    // Matrix B Initialization (Your pattern)
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < P; ++j) {
            B_linear[k * P + j] = static_cast<float>((k + j) % 8) * 0.5f; 
        }
    }

    // Tiling Process: Convert Linear -> Tiled -> Memcpy to Mapped Ptr
    std::vector<float> A_tiled(M * N);
    std::vector<float> B_tiled(N * P);
    
    convert_to_tiled(A_linear, A_tiled, M, N);
    convert_to_tiled(B_linear, B_tiled, N, P);

    std::memcpy(A_mapped, A_tiled.data(), size_a_bytes);
    std::memcpy(B_mapped, B_tiled.data(), size_b_bytes);
    std::memset(C_mapped, 0, size_c_bytes);

    // 7. Compute Golden Reference (CPU) - CONDITIONALLY
    std::vector<float> C_ref;
    if (check_cpu_ref) {
        std::cout << "Computing CPU reference..." << std::endl;
        C_ref.resize(M * P, 0.0f);
        auto t_cpu_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < P; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += A_linear[i * N + k] * B_linear[k * P + j];
                }
                C_ref[i * P + j] = sum;
            }
        }
        auto t_cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::micro>(t_cpu_end - t_cpu_start).count();
        std::cout << "CPU time: " << cpu_time << " us" << std::endl;
    } else {
        std::cout << "Skipping CPU reference (Dims > 1024)..." << std::endl;
    }

    // 8. Transfer Data to Device
    std::cout << "Syncing buffers to device..." << std::endl;
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 9. Execute Kernel
    std::cout << "Running FPGA kernel..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    
    auto run = krnl(boA, boB, boC, M, N, P);
    
    // Wait for kernel to finish
    run.wait();
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "FPGA execution time: " << elapsed_us << " us" << std::endl;

    // 10. Transfer Results back to Host
    std::cout << "Reading results..." << std::endl;
    boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // 11. Validate Results (Untile First) - CONDITIONALLY
    if (check_cpu_ref) {
        std::cout << "Validating..." << std::endl;
        
        std::vector<float> C_tiled_out(M * P);
        std::vector<float> C_final_linear(M * P);

        // Copy from Device Ptr to Vector
        std::memcpy(C_tiled_out.data(), C_mapped, size_c_bytes);
        
        // Convert Tiled -> Linear
        convert_from_tiled(C_tiled_out, C_final_linear, M, P);

        int errors = 0;
        float epsilon = 0.01f;

        for (int idx = 0; idx < M * P; ++idx) {
            float diff = std::abs(C_final_linear[idx] - C_ref[idx]);
            if (diff > epsilon) {
                if (errors < 10) { // Limit error output
                    int i = idx / P;
                    int j = idx % P;
                    std::cerr << "Mismatch at (" << i << "," << j << "): "
                              << "FPGA=" << std::fixed << std::setprecision(4) << C_final_linear[idx]
                              << ", REF=" << C_ref[idx] 
                              << ", Diff=" << diff << std::endl;
                }
                ++errors;
            }
        }

        if (errors) {
            std::cerr << "TEST FAILED with " << errors << " mismatches\n";
            return EXIT_FAILURE;
        } else {
            std::cout << "TEST PASSED!\n";
        }
    } else {
        std::cout << "Validation skipped. FPGA run complete.\n";
    }

    return EXIT_SUCCESS;
}