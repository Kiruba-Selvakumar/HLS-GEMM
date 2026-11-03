#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ap_int.h>
#include <hls_stream.h>
#include "ap_axi_sdata.h"

// --- Architecture Configuration ---
#define SA_DIM 32

// *** CRITICAL: Must match the 'depth' specified in the Kernel Pragma ***
// The HLS C-Simulation wrapper expects pointers to valid memory of this size.
#define MAX_DEPTH 4096 

// ------------------------------------------------------------
// Declare the Top-Level Kernel Function
// ------------------------------------------------------------
extern "C" {
void mmult_systolic(
    const ap_uint<512> *A,      
    const ap_uint<512> *B,      
    ap_uint<512> *C,            
    int M, int N, int P
);
}

// ------------------------------------------------------------
// Helper 1: Software Reference (Golden Model)
// ------------------------------------------------------------
void mmult_sw(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int P) {
    std::cout << "  [SW] Calculating Golden Reference..." << std::endl;
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
// Helper 2: Convert Linear -> Tiled
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
// Helper 3: Convert Tiled -> Linear
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
// Helper 4: Pack Floats into ap_uint<512>
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
// Helper 5: Unpack ap_uint<512> to Floats
// ------------------------------------------------------------
void unpack_data(const std::vector<ap_uint<512>>& input, std::vector<float>& output) {
    // Note: We only unpack the valid data, not the padding
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

    if (M % SA_DIM != 0 || N % SA_DIM != 0 || P % SA_DIM != 0) {
        std::cerr << "Error: Dimensions must be multiples of " << SA_DIM << std::endl;
        return 1;
    }

    std::cout << "================================================================" << std::endl;
    std::cout << "  Starting C-Simulation: Systolic Matrix Mult (" << M << "x" << N << " * " << N << "x" << P << ")" << std::endl;
    std::cout << "================================================================" << std::endl;

    // 2. Allocate Standard Memory
    std::vector<float> A_std(M * N);
    std::vector<float> B_std(N * P);
    std::vector<float> C_sw(M * P);
    std::vector<float> C_final_result(M * P); 

    // 3. Initialize with Random Floats
    std::cout << "  [TB] Initializing Data..." << std::endl;
    for (int i = 0; i < M * N; i++) A_std[i] = (float)(rand() % 10);
    for (int i = 0; i < N * P; i++) B_std[i] = (float)(rand() % 10);

    // 4. Run Software Reference
    mmult_sw(A_std, B_std, C_sw, M, N, P);

    // 5. Data Swizzling (Linear -> Tiled)
    std::cout << "  [TB] Swizzling Data to Block-Major Format..." << std::endl;
    std::vector<float> A_tiled(M * N);
    std::vector<float> B_tiled(N * P);
    convert_to_tiled(A_std, A_tiled, M, N);
    convert_to_tiled(B_std, B_tiled, N, P);

    // 6. Pack into AXI Width
    std::cout << "  [TB] Packing data into 512-bit AXI packets..." << std::endl;
    std::vector<ap_uint<512>> A_hw, B_hw, C_hw;
    pack_data(A_tiled, A_hw);
    pack_data(B_tiled, B_hw);
    
    // =========================================================================
    // FIX FOR SIGSEGV: Padding vectors to match Kernel 'depth=4096'
    // =========================================================================
    if (A_hw.size() < MAX_DEPTH) {
        // std::cout << "  [TB] Padding A_hw from " << A_hw.size() << " to " << MAX_DEPTH << std::endl;
        A_hw.resize(MAX_DEPTH);
    }
    if (B_hw.size() < MAX_DEPTH) {
        // std::cout << "  [TB] Padding B_hw from " << B_hw.size() << " to " << MAX_DEPTH << std::endl;
        B_hw.resize(MAX_DEPTH);
    }
    // Allocate Output Buffer to Max Depth
    C_hw.resize(MAX_DEPTH); 

    // 7. Run Hardware Kernel
    std::cout << "  [TB] invoking mmult_systolic..." << std::endl;
    // We pass the .data() pointers which now point to valid memory of size MAX_DEPTH
    mmult_systolic(A_hw.data(), B_hw.data(), C_hw.data(), M, N, P);
    std::cout << "  [TB] Kernel Return." << std::endl;

    // 8. Post-Process HW Results
    std::cout << "  [TB] Unpacking and Un-Swizzling Results..." << std::endl;
    
    // Create a temporary vector for the *valid* output data (excluding padding)
    // The actual valid data size is (M*P)/16 packets.
    std::vector<ap_uint<512>> C_hw_valid;
    int valid_packets = (M * P) / 16;
    
    // Copy only the valid part from the padded C_hw buffer
    C_hw_valid.assign(C_hw.begin(), C_hw.begin() + valid_packets);

    std::vector<float> C_tiled_out;
    unpack_data(C_hw_valid, C_tiled_out);
    convert_from_tiled(C_tiled_out, C_final_result, M, P);

    // 9. Verify Results
    std::cout << "  [TB] Verifying..." << std::endl;
    int errors = 0;
    float epsilon = 0.01f; 

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float hw_val = C_final_result[i * P + j];
            float sw_val = C_sw[i * P + j];
            float diff = std::abs(hw_val - sw_val);

            if (diff > epsilon) {
                errors++;
                if (errors < 10) { 
                    std::cout << "Mismatch at (" << i << "," << j << "): "
                              << "HW=" << hw_val << " SW=" << sw_val 
                              << " Diff=" << diff << std::endl;
                }
            }
        }
    }

    std::cout << "================================================================" << std::endl;
    if (errors == 0) {
        std::cout << "  TEST PASSED! (0 errors)" << std::endl;
        std::cout << "================================================================" << std::endl;
        return 0;
    } else {
        std::cout << "  TEST FAILED with " << errors << " errors." << std::endl;
        std::cout << "================================================================" << std::endl;
        return 1;
    }
}