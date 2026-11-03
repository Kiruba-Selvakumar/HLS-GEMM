#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <ap_int.h> // Required for ap_uint

// --- Configuration ---
// Must match the kernel's tile concept (SA_DIM = 48)
// Using 96x96 to ensure we test multiple tiles (48 * 2 = 96)
#define M_TEST 96
#define N_TEST 96
#define P_TEST 96
#define VEC_WIDTH 16

// Union for bit-safe casting
union fp_bit_cast {
    uint32_t i;
    float f;
};

// --- Kernel Declaration ---
// Matches the updated signature in mmult_systolic.cpp
extern "C" {
void mmult_systolic(
    const ap_uint<512> *A,
    const ap_uint<512> *B,
    ap_uint<512> *C,
    int M,
    int N,
    int P
);
}

// --- Helper: Pack Floats to 512-bit Vector ---
void pack_matrix(const std::vector<float>& input, std::vector<ap_uint<512>>& output, int rows, int cols) {
    // We assume rows*cols is a multiple of 16 for simplicity in this test
    int num_vectors = (rows * cols) / VEC_WIDTH;
    output.resize(num_vectors);

    for (int i = 0; i < num_vectors; i++) {
        ap_uint<512> vec;
        for (int v = 0; v < VEC_WIDTH; v++) {
            fp_bit_cast conv;
            // Handle edge case if size isn't perfect multiple
            if (i * VEC_WIDTH + v < input.size()) {
                conv.f = input[i * VEC_WIDTH + v];
                vec.range(32 * (v + 1) - 1, 32 * v) = conv.i;
            } else {
                vec.range(32 * (v + 1) - 1, 32 * v) = 0;
            }
        }
        output[i] = vec;
    }
}

// --- Helper: Unpack 512-bit Vector to Floats ---
void unpack_matrix(const std::vector<ap_uint<512>>& input, std::vector<float>& output, int rows, int cols) {
    output.resize(rows * cols);
    int num_vectors = input.size();

    for (int i = 0; i < num_vectors; i++) {
        ap_uint<512> vec = input[i];
        for (int v = 0; v < VEC_WIDTH; v++) {
            int flat_idx = i * VEC_WIDTH + v;
            if (flat_idx < output.size()) {
                fp_bit_cast conv;
                conv.i = vec.range(32 * (v + 1) - 1, 32 * v);
                output[flat_idx] = conv.f;
            }
        }
    }
}

// --- Golden Reference (CPU) ---
void mmult_cpu(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int M, int N, int P) {
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

int main() {
    std::cout << "Starting C-Simulation for Systolic MMULT (512-bit Optimized)..." << std::endl;
    std::cout << "Matrix Size: " << M_TEST << "x" << N_TEST << " * " << N_TEST << "x" << P_TEST << std::endl;

    // 1. Initialize Float Data
    std::vector<float> A_float(M_TEST * N_TEST);
    std::vector<float> B_float(N_TEST * P_TEST);
    
    // B Transposed is required for the HW kernel input packing
    std::vector<float> B_transposed_float(P_TEST * N_TEST);
    
    srand(42); 
    for (int i = 0; i < M_TEST * N_TEST; i++) A_float[i] = (float)(rand() % 10);
    for (int i = 0; i < N_TEST * P_TEST; i++) B_float[i] = (float)(rand() % 10);

    // 2. Transpose B (Required for Optimized Kernel)
    for(int k = 0; k < N_TEST; k++) {
        for(int j = 0; j < P_TEST; j++) {
            B_transposed_float[j * N_TEST + k] = B_float[k * P_TEST + j];
        }
    }

    // 3. Pack to ap_uint<512>
    std::vector<ap_uint<512>> A_hw;
    std::vector<ap_uint<512>> B_hw;
    std::vector<ap_uint<512>> C_hw_vec;
    
    // Allocate space for C output (Vectors)
    int c_vectors = (M_TEST * P_TEST + VEC_WIDTH - 1) / VEC_WIDTH;
    C_hw_vec.resize(c_vectors, 0);

    pack_matrix(A_float, A_hw, M_TEST, N_TEST);
    pack_matrix(B_transposed_float, B_hw, P_TEST, N_TEST); // Packing Transposed B

    // 4. Run Hardware Kernel
    std::cout << "Running HW Kernel..." << std::endl;
    mmult_systolic(A_hw.data(), B_hw.data(), C_hw_vec.data(), M_TEST, N_TEST, P_TEST);

    // 5. Unpack HW Result
    std::vector<float> C_hw_float;
    unpack_matrix(C_hw_vec, C_hw_float, M_TEST, P_TEST);

    // 6. Run Software Reference
    std::cout << "Running SW Reference..." << std::endl;
    std::vector<float> C_sw(M_TEST * P_TEST, 0.0f);
    mmult_cpu(A_float, B_float, C_sw, M_TEST, N_TEST, P_TEST);

    // 7. Verify Results
    std::cout << "Verifying Results..." << std::endl;
    int errors = 0;
    float epsilon = 0.001f; 

    for (int i = 0; i < M_TEST * P_TEST; i++) {
        float hw_val = C_hw_float[i];
        float sw_val = C_sw[i];
        float diff = std::abs(hw_val - sw_val);

        if (diff > epsilon) {
            errors++;
            if (errors < 10) { 
                int r = i / P_TEST;
                int c = i % P_TEST;
                std::cout << "Mismatch at (" << r << "," << c << "): "
                          << "HW=" << hw_val << " SW=" << sw_val << " Diff=" << diff << std::endl;
            }
        }
    }

    if (errors == 0) {
        std::cout << "CSIM PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED with " << errors << " errors." << std::endl;
        return 1;
    }
}