#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include "ap_axi_sdata.h"

// --- Configuration ---
#define SA_DIM 32
#define MAX_DEPTH 4096 // Matches the Kernel Pragma

// --- Kernel Declaration ---
// Must match the Kernel definition exactly (ap_uint<512>*, not float*)
extern "C" {
void mmult_systolic(
    const ap_uint<512> *A,
    const ap_uint<512> *B,
    ap_uint<512> *C,
    int M, int N, int P
);
}

// ------------------------------------------------------------
// Helper: Software Reference (Standard Row-Major)
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
// Helper: Convert Linear -> Block Tiled (Swizzling)
// ------------------------------------------------------------
void convert_to_tiled(const std::vector<float>& input, std::vector<float>& output, int Rows, int Cols) {
    int num_tiles_R = Rows / SA_DIM;
    int num_tiles_C = Cols / SA_DIM;
    int idx = 0;
    
    for (int tr = 0; tr < num_tiles_R; tr++) {
        for (int tc = 0; tc < num_tiles_C; tc++) {
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
// Helper: Convert Block Tiled -> Linear (Un-Swizzling)
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
// Helper: Pack Floats into ap_uint<512>
// ------------------------------------------------------------
void pack_data(const std::vector<float>& input, std::vector<ap_uint<512>>& output) {
    int num_packets = input.size() / 16;
    output.resize(num_packets);

    for (int i = 0; i < num_packets; i++) {
        ap_uint<512> packet;
        for (int j = 0; j < 16; j++) {
            float val = input[i * 16 + j];
            uint32_t raw = *reinterpret_cast<const uint32_t*>(&val);
            packet.range(32 * (j + 1) - 1, 32 * j) = raw;
        }
        output[i] = packet;
    }
}

// ------------------------------------------------------------
// Helper: Unpack ap_uint<512> to Floats
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
    const int M = 64;
    const int N = 64;
    const int P = 64;

    std::cout << "Starting C-Simulation..." << std::endl;

    // 1. Standard Float Initialization
    std::vector<float> A_std(M * N);
    std::vector<float> B_std(N * P);
    std::vector<float> C_sw(M * P);
    std::vector<float> C_final(M * P); 

    for (int i = 0; i < M * N; i++) A_std[i] = (float)(rand() % 10);
    for (int i = 0; i < N * P; i++) B_std[i] = (float)(rand() % 10);

    // 2. Run CPU Reference
    mmult_sw(A_std, B_std, C_sw, M, N, P);

    // 3. Prepare HW Data: Swizzle (Tile) -> Pack
    std::vector<float> A_tiled(M * N);
    std::vector<float> B_tiled(N * P);
    convert_to_tiled(A_std, A_tiled, M, N);
    convert_to_tiled(B_std, B_tiled, N, P);

    std::vector<ap_uint<512>> A_hw, B_hw, C_hw;
    pack_data(A_tiled, A_hw);
    pack_data(B_tiled, B_hw);
    
    // 4. *** FIX FOR SEGFAULT ***: Pad vectors to MAX_DEPTH (4096)
    // The kernel interface pragma depth=4096 forces the simulator to check this size.
    if (A_hw.size() < MAX_DEPTH) A_hw.resize(MAX_DEPTH);
    if (B_hw.size() < MAX_DEPTH) B_hw.resize(MAX_DEPTH);
    C_hw.resize(MAX_DEPTH); 

    // 5. Run HW Kernel
    // Note: We use .data() on the RESIZED vectors, so they are safe to read up to 4096
    mmult_systolic(A_hw.data(), B_hw.data(), C_hw.data(), M, N, P);

    // 6. Unpack & Verify
    // We only care about the valid data (M*P/16 packets), not the padding
    int valid_packets = (M * P) / 16;
    std::vector<ap_uint<512>> C_hw_valid(C_hw.begin(), C_hw.begin() + valid_packets);
    
    std::vector<float> C_tiled_out;
    unpack_data(C_hw_valid, C_tiled_out);
    convert_from_tiled(C_tiled_out, C_final, M, P);

    int errors = 0;
    for (int i = 0; i < M * P; i++) {
        if (std::abs(C_final[i] - C_sw[i]) > 0.01f) errors++;
    }

    if (errors == 0) std::cout << "CSIM PASSED!" << std::endl;
    else std::cout << "TEST FAILED with " << errors << " errors." << std::endl;
    
    return (errors == 0) ? 0 : 1;
}