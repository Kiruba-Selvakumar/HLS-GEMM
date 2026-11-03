/**
 * Host code for Double-Buffered Systolic GEMM
 * Target Kernel: mmult_systolic
 * API: XRT Native C++
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

// --- Configuration ---
static const int SA_DIM = 32; // Must match HLS design

// Default to 1024 to match your Latency Simulation
static const int M_default = 1024;
static const int N_default = 1024;
static const int P_default = 1024;
static const int Device_default = 0;

// -------------------------------------------------------------------------
// Helper: Software Tiling (Swizzling)
// Input:  Standard Row-Major Matrix (Linear)
// Output: Block-Major Matrix (Tiles of 32x32 stored sequentially)
// -------------------------------------------------------------------------
void convert_to_tiled(const std::vector<float>& linear_data, std::vector<float>& tiled_data, int Rows, int Cols) {
    int num_tiles_R = Rows / SA_DIM;
    int num_tiles_C = Cols / SA_DIM;
    
    int tiled_idx = 0;
    
    // 1. Iterate over Tile Coordinates (tr, tc)
    for (int tr = 0; tr < num_tiles_R; ++tr) {
        for (int tc = 0; tc < num_tiles_C; ++tc) {
            
            // 2. Iterate inside the specific 32x32 tile
            for (int r = 0; r < SA_DIM; ++r) {
                for (int c = 0; c < SA_DIM; ++c) {
                    // Calculate global row/col in the linear matrix
                    int global_r = tr * SA_DIM + r;
                    int global_c = tc * SA_DIM + c;
                    
                    int linear_idx = global_r * Cols + global_c;
                    tiled_data[tiled_idx++] = linear_data[linear_idx];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Helper: Reverse Tiling (Un-Swizzling)
// Input:  Block-Major Matrix (Result from FPGA)
// Output: Standard Row-Major Matrix (For Verification)
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

// Arg Parser
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
        if ((arg == "-x" || arg == "--xclbin") && i + 1 < argc) {
            args.xclbin_file = argv[++i];
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            args.device_index = std::stoi(argv[++i]);
        } else if ((arg == "-m") && i + 1 < argc) args.M = std::stoi(argv[++i]);
        else if ((arg == "-n") && i + 1 < argc) args.N = std::stoi(argv[++i]);
        else if ((arg == "-p") && i + 1 < argc) args.P = std::stoi(argv[++i]);
    }
    if (!args.xclbin_file.empty()) args.valid = true;
    return args;
}

int main(int argc, char** argv) {
    CmdArgs args = parse_args(argc, argv);
    if (!args.valid) {
        std::cout << "Usage: " << argv[0] << " -x <xclbin> [-m 1024] [-n 1024] [-p 1024]" << std::endl;
        return EXIT_FAILURE;
    }

    // Constraint Check
    if (args.M % SA_DIM != 0 || args.N % SA_DIM != 0 || args.P % SA_DIM != 0) {
        std::cerr << "Error: Dimensions must be multiples of " << SA_DIM << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "------------------------------------------------\n";
    std::cout << "  Double-Buffered Systolic GEMM Host\n";
    std::cout << "------------------------------------------------\n";
    std::cout << "  Dims: " << args.M << "x" << args.N << " * " << args.N << "x" << args.P << "\n";

    // 1. XRT Setup
    auto device = xrt::device(args.device_index);
    auto uuid = device.load_xclbin(args.xclbin_file);
    auto krnl = xrt::kernel(device, uuid, "mmult_systolic");

    // 2. Data Preparation (Host Side)
    size_t bytes_A = sizeof(float) * args.M * args.N;
    size_t bytes_B = sizeof(float) * args.N * args.P;
    size_t bytes_C = sizeof(float) * args.M * args.P;

    std::vector<float> A_linear(args.M * args.N);
    std::vector<float> B_linear(args.N * args.P);
    
    // Init with random data
    std::generate(A_linear.begin(), A_linear.end(), [](){ return (std::rand()%10) / 10.0f; });
    std::generate(B_linear.begin(), B_linear.end(), [](){ return (std::rand()%10) / 10.0f; });

    // *** CRITICAL: Perform Software Tiling ***
    // The FPGA reads 32x32 blocks linearly. We must rearrange memory to match.
    std::cout << "[Host] Swizzling data to Block-Major format..." << std::endl;
    std::vector<float> A_tiled(args.M * args.N);
    std::vector<float> B_tiled(args.N * args.P);
    
    convert_to_tiled(A_linear, A_tiled, args.M, args.N);
    convert_to_tiled(B_linear, B_tiled, args.N, args.P);

    // 3. Allocate & Transfer
    auto boA = xrt::bo(device, bytes_A, krnl.group_id(0));
    auto boB = xrt::bo(device, bytes_B, krnl.group_id(1));
    auto boC = xrt::bo(device, bytes_C, krnl.group_id(2));

    boA.write(A_tiled.data());
    boB.write(B_tiled.data());

    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 4. Execution
    std::cout << "[Host] Running FPGA Kernel..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto run = krnl(boA, boB, boC, args.M, args.N, args.P);
    run.wait();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // GFLOPS Calculation
    double gops = (2.0 * args.M * args.N * args.P) * 1e-9;
    double gflops = gops / (duration * 1e-6);

    std::cout << "[Host] Done. Time: " << duration << " us. Perf: " << gflops << " GFLOPS" << std::endl;

    // 5. Readback & Validation
    // Only validate smaller matrices to save time
    if (args.M <= 1024) {
        std::cout << "[Host] Validating results..." << std::endl;
        boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        
        std::vector<float> C_tiled_out(args.M * args.P);
        boC.read(C_tiled_out.data());

        // Un-Swizzle back to Linear for comparison
        std::vector<float> C_linear_out(args.M * args.P);
        convert_from_tiled(C_tiled_out, C_linear_out, args.M, args.P);

        // CPU Ref
        int errors = 0;
        for(int i=0; i<args.M; i++) {
            for(int j=0; j<args.P; j++) {
                float sum = 0.0f;
                for(int k=0; k<args.N; k++) {
                    sum += A_linear[i*args.N + k] * B_linear[k*args.P + j];
                }
                float diff = std::abs(C_linear_out[i*args.P + j] - sum);
                if(diff > 0.1f) {
                    if (errors < 5) std::cout << "Mismatch at " << i << "," << j << " CPU=" << sum << " FPGA=" << C_linear_out[i*args.P + j] << "\n";
                    errors++;
                }
            }
        }
        if(errors == 0) std::cout << "PASSED!" << std::endl;
        else std::cout << "FAILED with " << errors << " errors." << std::endl;
    }

    return 0;
}