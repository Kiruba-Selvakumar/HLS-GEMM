/**
 * Host code for Optimized 512-bit Systolic Kernel
 * - Handles Float -> ap_uint<512> packing
 * - Handles Matrix B Transposition
 * - Verifies results
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

// We need a struct to mimic ap_uint<512> on the host side for data packing
// 512 bits = 64 bytes = 16 floats
struct alignas(64) PackedFloat512 {
    uint32_t data[16]; 
};

// Union for bit-safe casting
union fp_bit_cast {
    uint32_t i;
    float f;
};

// --- Helper: Pack Floats to 512-bit Structure ---
void pack_matrix(const std::vector<float>& input, std::vector<PackedFloat512>& output, int rows, int cols) {
    int vec_width = 16;
    int num_vectors = (rows * cols + vec_width - 1) / vec_width;
    output.resize(num_vectors);
    
    // Clear memory to avoid garbage in padding
    std::memset(output.data(), 0, num_vectors * sizeof(PackedFloat512));

    for (int i = 0; i < num_vectors; i++) {
        for (int v = 0; v < vec_width; v++) {
            int flat_idx = i * vec_width + v;
            if (flat_idx < input.size()) {
                fp_bit_cast conv;
                conv.f = input[flat_idx];
                output[i].data[v] = conv.i;
            } else {
                output[i].data[v] = 0;
            }
        }
    }
}

// --- Helper: Unpack 512-bit Structure to Floats ---
void unpack_matrix(const std::vector<PackedFloat512>& input, std::vector<float>& output, int rows, int cols) {
    output.resize(rows * cols);
    int vec_width = 16;
    
    for (int i = 0; i < input.size(); i++) {
        for (int v = 0; v < vec_width; v++) {
            int flat_idx = i * vec_width + v;
            if (flat_idx < output.size()) {
                fp_bit_cast conv;
                conv.i = input[i].data[v];
                output[flat_idx] = conv.f;
            }
        }
    }
}

// --- Argument Parsing ---
struct CmdArgs {
    std::string xclbin_file;
    int device_index = 0;
    int M = 1024;
    int N = 1024;
    int P = 1024;
    bool valid = false;
};

CmdArgs parse_args(int argc, char** argv) {
    CmdArgs args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-x" || arg == "--xclbin_file") && i + 1 < argc) args.xclbin_file = argv[++i];
        else if ((arg == "-d" || arg == "--device_id") && i + 1 < argc) args.device_index = std::stoi(argv[++i]);
        else if ((arg == "-m" || arg == "--M") && i + 1 < argc) args.M = std::stoi(argv[++i]);
        else if ((arg == "-n" || arg == "--N") && i + 1 < argc) args.N = std::stoi(argv[++i]);
        else if ((arg == "-p" || arg == "--P") && i + 1 < argc) args.P = std::stoi(argv[++i]);
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

    int M = args.M;
    int N = args.N;
    int P = args.P;

    std::cout << "Starting Host for Optimized Systolic MMULT..." << std::endl;
    std::cout << "Matrix: " << M << "x" << N << " * " << N << "x" << P << std::endl;

    // 1. Initialize Float Data
    std::vector<float> A_float(M * N);
    std::vector<float> B_float(N * P);
    std::vector<float> B_transposed_float(P * N); // Transpose B for kernel

    for (int i = 0; i < M * N; i++) A_float[i] = (float)(i % 10);
    for (int i = 0; i < N * P; i++) B_float[i] = (float)(i % 7);

    // 2. Perform Host-Side Transpose of B
    // Kernel reads B rows to simulate columns.
    std::cout << "Transposing Matrix B..." << std::endl;
    for(int k = 0; k < N; k++) {
        for(int j = 0; j < P; j++) {
            B_transposed_float[j * N + k] = B_float[k * P + j];
        }
    }

    // 3. Pack Data for FPGA (Float -> 512-bit)
    std::cout << "Packing data to 512-bit vectors..." << std::endl;
    std::vector<PackedFloat512> A_hw;
    std::vector<PackedFloat512> B_hw;
    std::vector<PackedFloat512> C_hw_vec; // Output buffer

    pack_matrix(A_float, A_hw, M, N);
    pack_matrix(B_transposed_float, B_hw, P, N);
    
    // Resize C buffer
    int c_vecs = (M * P + 15) / 16;
    C_hw_vec.resize(c_vecs);

    // 4. Open Device & Load Binary
    auto device = xrt::device(args.device_index);
    auto uuid = device.load_xclbin(args.xclbin_file);
    auto krnl = xrt::kernel(device, uuid, "mmult_systolic");

    // 5. Allocate Buffers
    size_t vec_size_bytes = sizeof(PackedFloat512);
    auto boA = xrt::bo(device, A_hw.size() * vec_size_bytes, krnl.group_id(0));
    auto boB = xrt::bo(device, B_hw.size() * vec_size_bytes, krnl.group_id(1));
    auto boC = xrt::bo(device, C_hw_vec.size() * vec_size_bytes, krnl.group_id(2));

    // 6. Write Data
    boA.write(A_hw.data());
    boB.write(B_hw.data());
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 7. Execute
    std::cout << "Running Kernel..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto run = krnl(boA, boB, boC, M, N, P);
    run.wait();
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Kernel Execution Time: " << duration << " us" << std::endl;
    double gflops = (2.0 * M * N * P) / (duration * 1000.0); // 10^9 = 1000us * 10^6
    std::cout << "Approx Performance: " << gflops << " GFLOPS" << std::endl;

    // 8. Read Result
    boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boC.read(C_hw_vec.data());

    // 9. Unpack & Verify (Optional: only verify small subset for large matrices)
    std::cout << "Unpacking results..." << std::endl;
    std::vector<float> C_final;
    unpack_matrix(C_hw_vec, C_final, M, P);

    // Basic check on corner elements
    std::cout << "Checking a few elements (No Full CPU Verification for speed)..." << std::endl;
    
    // Check Top-Left (0,0)
    float cpu_00 = 0.0f;
    for(int k=0; k<N; k++) cpu_00 += A_float[k] * B_float[k*P];
    
    std::cout << "(0,0) HW: " << C_final[0] << " | Expected: " << cpu_00 << std::endl;

    // Check Bottom-Right
    float cpu_end = 0.0f;
    for(int k=0; k<N; k++) cpu_end += A_float[(M-1)*N + k] * B_float[k*P + (P-1)];
    int last_idx = (M-1)*P + (P-1);
    std::cout << "(MAX,MAX) HW: " << C_final[last_idx] << " | Expected: " << cpu_end << std::endl;

    if (std::abs(C_final[0] - cpu_00) < 0.1 && std::abs(C_final[last_idx] - cpu_end) < 0.1) {
        std::cout << "Sanity Check PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Sanity Check FAILED!" << std::endl;
        return 1;
    }
}