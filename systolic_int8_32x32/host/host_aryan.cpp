/**
 * Host code for Int8 Systolic Matrix Multiplication
 * Target Kernel: mmult_systolic
 * API: XRT Native C++
 * * fixes: Uses int8/int32 types and performs Software Tiling
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

// --- Architecture Configuration ---
#define SA_DIM 32 

// Default Parameters
static const int M_default = 256; // Must be multiple of 32
static const int N_default = 256;
static const int P_default = 256;
static const int Device_default = 0;

// Command Line Parser
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

// -------------------------------------------------------------------------
// CPU Golden Reference
// -------------------------------------------------------------------------
void mmult_cpu(const std::vector<int8_t>& A, const std::vector<int8_t>& B, std::vector<int32_t>& C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += (int32_t)A[i * N + k] * (int32_t)B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// -------------------------------------------------------------------------
// DATA PACKING FUNCTIONS (Linear -> Tiled)
// -------------------------------------------------------------------------

// Pack A: Standard Block Tiling
void pack_A(int8_t* bo_ptr, const std::vector<int8_t>& A_linear, int M, int N) {
    int num_tiles_R = M / SA_DIM;
    int num_tiles_C = N / SA_DIM;
    int bo_idx = 0;

    for (int tr = 0; tr < num_tiles_R; tr++) {
        for (int tc = 0; tc < num_tiles_C; tc++) {
            // Inside Tile: Pack 2 rows per 64-byte chunk (512-bit)
            for (int i = 0; i < SA_DIM; i += 2) {
                // Copy Row i (32 bytes)
                int global_r1 = tr * SA_DIM + i;
                int global_c_start = tc * SA_DIM;
                std::memcpy(&bo_ptr[bo_idx], &A_linear[global_r1 * N + global_c_start], 32);
                bo_idx += 32;

                // Copy Row i+1 (32 bytes)
                int global_r2 = tr * SA_DIM + (i + 1);
                std::memcpy(&bo_ptr[bo_idx], &A_linear[global_r2 * N + global_c_start], 32);
                bo_idx += 32;
            }
        }
    }
}

// Pack B: Column-Major Tiles (Transposed Logic) but Row-Major Data inside
void pack_B(int8_t* bo_ptr, const std::vector<int8_t>& B_linear, int N, int P) {
    int num_tiles_N = N / SA_DIM; // Rows of B (Depth)
    int num_tiles_P = P / SA_DIM; // Cols of B

    int bo_idx = 0;

    // Outer Loops: Iterate P (Cols) then N (Depth) to match Kernel Read Order
    for (int tc = 0; tc < num_tiles_P; tc++) {
        for (int tk = 0; tk < num_tiles_N; tk++) {
            
            // Inside Tile: We write Rows of B
            for (int k = 0; k < SA_DIM; k += 2) {
                // Row k
                int global_k1 = tk * SA_DIM + k;
                int global_p_start = tc * SA_DIM;
                std::memcpy(&bo_ptr[bo_idx], &B_linear[global_k1 * P + global_p_start], 32);
                bo_idx += 32;

                // Row k+1
                int global_k2 = tk * SA_DIM + (k + 1);
                std::memcpy(&bo_ptr[bo_idx], &B_linear[global_k2 * P + global_p_start], 32);
                bo_idx += 32;
            }
        }
    }
}

// Unpack C: Tiled -> Linear
void unpack_C(const int32_t* bo_ptr, std::vector<int32_t>& C_linear, int M, int P) {
    int num_tiles_R = M / SA_DIM;
    int num_tiles_C = P / SA_DIM;
    int bo_idx = 0;

    for (int tr = 0; tr < num_tiles_R; tr++) {
        for (int tc = 0; tc < num_tiles_C; tc++) {
            
            // Inside Tile: Read Rows
            for (int i = 0; i < SA_DIM; i++) {
                int global_r = tr * SA_DIM + i;
                int global_c_start = tc * SA_DIM;
                
                // Copy one row (32 ints) from BO to Linear Vector
                // In BO, this is stored contiguously
                std::memcpy(&C_linear[global_r * P + global_c_start], &bo_ptr[bo_idx], 32 * sizeof(int32_t));
                bo_idx += 32; // Increment by 32 ints
            }
        }
    }
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    // 1. Parse Arguments
    CmdArgs args = parse_args(argc, argv);
    if (!args.valid) {
        std::cout << "Usage: " << argv[0] << " -x <xclbin_file> [-d <device_id>] [-m <rows>] [-n <common>] [-p <cols>]" << std::endl;
        return EXIT_FAILURE;
    }

    // Alignment Check
    if (args.M % SA_DIM != 0 || args.N % SA_DIM != 0 || args.P % SA_DIM != 0) {
        std::cerr << "Error: M, N, P must be multiples of " << SA_DIM << std::endl;
        return EXIT_FAILURE;
    }

    int M = args.M;
    int N = args.N;
    int P = args.P;

    std::cout << "------------------------------------------------\n";
    std::cout << "Systolic Matrix Mult Host (Int8 Tiled)\n";
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
    std::cout << "Creating kernel handle 'mmult_systolic'..." << std::endl;
    auto krnl = xrt::kernel(device, uuid, "mmult_systolic");

    // 4. Allocate Buffer Objects (BOs)
    // Size in bytes
    // A and B are Int8 (1 byte per element)
    // C is Int32 (4 bytes per element)
    const size_t size_a_bytes = M * N * sizeof(int8_t);
    const size_t size_b_bytes = N * P * sizeof(int8_t);
    const size_t size_c_bytes = M * P * sizeof(int32_t);

    std::cout << "Allocating buffers..." << std::endl;
    auto boA = xrt::bo(device, size_a_bytes, krnl.group_id(0));
    auto boB = xrt::bo(device, size_b_bytes, krnl.group_id(1));
    auto boC = xrt::bo(device, size_c_bytes, krnl.group_id(2));

    // 5. Initialize Host Data (Linear)
    std::vector<int8_t> A_linear(M * N);
    std::vector<int8_t> B_linear(N * P);
    std::vector<int32_t> C_linear_ref(M * P);
    std::vector<int32_t> C_linear_out(M * P);

    std::cout << "Initializing matrices..." << std::endl;
    srand(42);
    for (size_t i = 0; i < A_linear.size(); i++) A_linear[i] = rand() % 10;
    for (size_t i = 0; i < B_linear.size(); i++) B_linear[i] = rand() % 10;

    // 6. Map and Pack Data (Linear Host -> Tiled Device BO)
    std::cout << "Packing data into tiles..." << std::endl;
    
    // Map BOs to pointers
    // Note: We map to int8_t* for byte-level access during packing
    auto A_dev_ptr = boA.map<int8_t*>();
    auto B_dev_ptr = boB.map<int8_t*>();
    auto C_dev_ptr = boC.map<int32_t*>(); // C is accessed as int32

    // Perform Software Tiling
    pack_A(A_dev_ptr, A_linear, M, N);
    pack_B(B_dev_ptr, B_linear, N, P);
    
    // Clear Output Buffer
    std::memset(C_dev_ptr, 0, size_c_bytes);

    // 7. Sync to Device
    std::cout << "Syncing buffers to device..." << std::endl;
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 8. Execute Kernel
    std::cout << "Running FPGA kernel..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Launch Kernel
    auto run = krnl(boA, boB, boC, M, N, P);
    run.wait();
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    
    // Calculate GOPS (Integer Operations)
    // 2 Ops per MAC (Multiply + Add)
    double gops = (2.0 * M * N * P) / (elapsed_us * 1000.0); 
    std::cout << "FPGA execution time: " << elapsed_us << " us" << std::endl;
    std::cout << "Performance: " << gops << " GOPS" << std::endl;

    // 9. Sync Results Back
    std::cout << "Reading results..." << std::endl;
    boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // 10. Unpack and Verify
    std::cout << "Unpacking and Validating..." << std::endl;
    
    // Convert Tiled Device Output -> Linear Host Vector
    unpack_C(C_dev_ptr, C_linear_out, M, P);

    // Compute Golden Reference
    std::cout << "Computing CPU reference..." << std::endl;
    mmult_cpu(A_linear, B_linear, C_linear_ref, M, N, P);

    // Compare
    int errors = 0;
    for (int i = 0; i < M * P; ++i) {
        if (C_linear_out[i] != C_linear_ref[i]) {
            if (errors < 10) {
                std::cerr << "Mismatch at index " << i << ": HW=" << C_linear_out[i] 
                          << " SW=" << C_linear_ref[i] << std::endl;
            }
            errors++;
        }
    }

    if (errors) {
        std::cerr << "TEST FAILED with aryan" << errors << " mismatches\n";
    } else {
        std::cout << "TEST PASSED!\n";
    }

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}