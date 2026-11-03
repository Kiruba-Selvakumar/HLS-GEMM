#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>

// Define the maximum dimensions based on your kernel header
#define MAX_M 256
#define MAX_N 256
#define MAX_P 256

// Error tolerance for float comparison
#define EPSILON 1e-3

// Declaration of the Top Level Kernel Function
extern "C" {
    void mmult(const float *A, const float *B, float *C, int M, int N, int P);
}

// Software Golden Reference Implementation
// Used to verify the correctness of the hardware kernel
void mmult_sw(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int M, int N, int P) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                // A is MxN, B is NxP
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

int main() {
    // 1. Define Matrix Dimensions
    // We pick sizes that are multiples of TILE_SIZE (16) and some that aren't 
    // to ensure the boundary checks in your kernel work.
    // Let's use 64x64x64 for a standard test. 
    // (Must be <= MAX constants defined in kernel)
    int M = 64; 
    int N = 64; 
    int P = 64;
    const int MAX_DEPTH = 65536;
    std::cout << "================================================================" << std::endl;
    std::cout << "Starting Matrix Multiplication Testbench" << std::endl;
    std::cout << "Dimensions: M=" << M << ", N=" << N << ", P=" << P << std::endl;
    std::cout << "================================================================" << std::endl;

    // 2. Allocate Memory
    // Using std::vector for automatic memory management
    std::vector<float> A(MAX_DEPTH);
    std::vector<float> B(MAX_DEPTH);
    std::vector<float> C_hw(MAX_DEPTH); // Hardware Result
    std::vector<float> C_sw(MAX_DEPTH); // Software Reference Result

    // 3. Initialize Input Data
    // Fill with random floats between 0.0 and 1.0
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < M * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < N * P; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }

    // 4. Run Software Golden Reference
    std::cout << "Running Software Reference..." << std::endl;
    mmult_sw(A, B, C_sw, M, N, P);

    // 5. Run Hardware Kernel (C-Simulation)
    std::cout << "Running Hardware Kernel..." << std::endl;
    // .data() returns the pointer to the underlying array
    mmult(A.data(), B.data(), C_hw.data(), M, N, P);

    // 6. Verification
    std::cout << "Verifying Results..." << std::endl;
    int err_count = 0;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float hw_val = C_hw[i * P + j];
            float sw_val = C_sw[i * P + j];
            
            float diff = std::abs(hw_val - sw_val);

            // Check against threshold
            if (diff > EPSILON) {
                err_count++;
                if (err_count < 10) { // Only print first few errors
                    std::cout << "ERROR at [" << i << "][" << j << "]: "
                              << "HW=" << hw_val << ", SW=" << sw_val 
                              << ", Diff=" << diff << std::endl;
                }
            }
        }
    }

    // 7. Result Reporting
    std::cout << "================================================================" << std::endl;
    if (err_count == 0) {
        std::cout << "TEST PASSED!" << std::endl;
        std::cout << "================================================================" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED with " << err_count << " errors." << std::endl;
        std::cout << "================================================================" << std::endl;
        return 1;
    }
}
