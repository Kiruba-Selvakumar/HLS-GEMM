#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ap_int.h>
#include <hls_stream.h>
#include "ap_axi_sdata.h"

// Define Tile Size (Must match Kernel)
#define SA_DIM 32

// Declare the Top-Level Kernel Function
extern "C" {
void mmult_systolic(
    const ap_uint<512> *A,      
    const ap_uint<512> *B,      
    ap_uint<512> *C,            
    int M, int N, int P
);
}

// ------------------------------------------------------------
// Helper: Software Reference Matrix Multiplication (Standard)
// ------------------------------------------------------------
void mmult_sw(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// ------------------------------------------------------------
// Helper: Convert Standard Linear Layout to Tiled (Block) Layout
// ------------------------------------------------------------
// This rearranges a standard row-major matrix into 32x32 blocks
// so the hardware can burst-read full tiles.
// ------------------------------------------------------------
void convert_to_tiled(const std::vector<float>& input, std::vector<float>& output, int Rows, int Cols) {
    int num_tiles_R = Rows / SA_DIM;
    int num_tiles_C = Cols / SA_DIM;

    int idx = 0;
    // Iterate over Tiles
    for (int tr = 0; tr < num_tiles_R; tr++) {
        for (int tc = 0; tc < num_tiles_C; tc++) {
            // Iterate inside the Tile (32x32)
            for (int r = 0; r < SA_DIM; r++) {
                for (int c = 0; c < SA_DIM; c++) {
                    int global_row = tr * SA_DIM + r;
                    int global_col = tc * SA_DIM + c;
                    output[idx++] = input[global_row * Cols + global_col];
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Helper: Convert Tiled Layout back to Standard Linear Layout
// ------------------------------------------------------------
void convert_from_tiled(const std::vector<float>& input, std::vector<float>& output, int Rows, int Cols) {
    int num_tiles_R = Rows / SA_DIM;
    int num_tiles_C = Cols / SA_DIM;

    int idx = 0;
    for (int tr = 0; tr < num_tiles_R; tr++) {
        for (int tc = 0; tc < num_tiles_C; tc++) {
            for (int r = 0; r < SA_DIM; r++) {
                for (int c = 0; c < SA_DIM; c++) {
                    int global_row = tr * SA_DIM + r;
                    int global_col = tc * SA_DIM + c;
                    output[global_row * Cols + global_col] = input[idx++];
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Helper: Pack Float Vector into ap_uint<512> Vector
// ------------------------------------------------------------
void pack_data(const std::vector<float>& input, std::vector<ap_uint<512>>& output) {
    // 16 floats per 512-bit packet
    int num_packets = input.size() / 16;
    output.resize(num_packets);

    for (int i = 0; i < num_packets; i++) {
        ap_uint<512> packet;
        for (int j = 0; j < 16; j++) {
            float val = input[i * 16 + j];
            // Bit-cast float to uint32 to preserve bits
            uint32_t raw = *reinterpret_cast<uint32_t*>(&val);
            packet.range(32 * (j + 1) - 1, 32 * j) = raw;
        }
        output[i] = packet;
    }
}

// ------------------------------------------------------------
// Helper: Unpack ap_uint<512> Vector to Float Vector
// ------------------------------------------------------------
void unpack_data(const std::vector<ap_uint<512>>& input, std::vector<float>& output) {
    int num_elements = input.size() * 16;
    output.resize(num_elements);

    for (int i = 0; i < input.size(); i++) {
        ap_uint<512> packet = input[i];
        for (int j = 0; j < 16; j++) {
            uint32_t raw = packet.range(32 * (j + 1) - 1, 32 * j);
            float val = *reinterpret_cast<float*>(&raw);
            output[i * 16 + j] = val;
        }
    }
}

// ------------------------------------------------------------
// Main Testbench
// ------------------------------------------------------------
int main() {
    // 1. Setup Dimensions (Must be multiples of 32)
    const int M = 64;
    const int N = 64;
    const int P = 64;

    if (M % 32 != 0 || N % 32 != 0 || P % 32 != 0) {
        std::cerr << "Error: Dimensions must be multiples of 32 for this kernel." << std::endl;
        return 1;
    }

    std::cout << "Starting C-Simulation for Systolic MMULT..." << std::endl;
    std::cout << "Matrix Size: " << M << "x" << N << " * " << N << "x" << P << std::endl;

    // 2. Allocate Standard Memory
    std::vector<float> A_std(M * N);
    std::vector<float> B_std(N * P);
    std::vector<float> C_sw(M * P);
    std::vector<float> C_final(M * P); // Unpacked HW result

    // 3. Initialize with Random Floats
    for (int i = 0; i < M * N; i++) A_std[i] = (float)(rand() % 10) / 2.0f;
    for (int i = 0; i < N * P; i++) B_std[i] = (float)(rand() % 10) / 2.0f;

    // 4. Run Software Reference
    std::cout << "Running SW Reference..." << std::endl;
    mmult_sw(A_std, B_std, C_sw, M, N, P);

    // 5. Transform to Tiled Layout
    std::vector<float> A_tiled(M * N);
    std::vector<float> B_tiled(N * P);
    convert_to_tiled(A_std, A_tiled, M, N);
    convert_to_tiled(B_std, B_tiled, N, P);

    // 6. Pack into ap_uint<512>
    std::vector<ap_uint<512>> A_hw, B_hw, C_hw;
    pack_data(A_tiled, A_hw);
    pack_data(B_tiled, B_hw);
    // Allocate space for Output
    C_hw.resize((M * P) / 16); 

    // 7. Run Hardware Kernel
    std::cout << "Running HW Kernel..." << std::endl;
    mmult_systolic(A_hw.data(), B_hw.data(), C_hw.data(), M, N, P);

    // 8. Unpack and Untile Output
    std::vector<float> C_tiled_out;
    unpack_data(C_hw, C_tiled_out);
    convert_from_tiled(C_tiled_out, C_final, M, P);

    // 9. Verify Results
    std::cout << "Verifying Results..." << std::endl;
    int errors = 0;
    float epsilon = 0.01f; // Tolerance for float comparison

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float hw_val = C_final[i * P + j];
            float sw_val = C_sw[i * P + j];
            float diff = std::abs(hw_val - sw_val);

            if (diff > epsilon) {
                errors++;
                if (errors < 10) { // Print first 10 errors
                    std::cout << "Mismatch at (" << i << "," << j << "): "
                              << "HW=" << hw_val << " SW=" << sw_val 
                              << " Diff=" << diff << std::endl;
                }
            }
        }
    }

    if (errors == 0) {
        std::cout << "TEST PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED with " << errors << " errors." << std::endl;
        return 1;
    }
}