#include <iostream>
#include <vector>
#include <cstdlib>
#include <stdint.h>
#include "ap_int.h"

// --- Configuration ---
// These dimensions (64x64) fit easily into the 4096 depth buffer
#define M_TEST 64
#define N_TEST 64
#define P_TEST 64
#define SA_DIM 32

// Defined in Kernel Pragma (Must match to avoid SIGSEGV in CoSim)
#define COSIM_BUFFER_DEPTH 4096 

extern "C" {
void mmult_systolic(
    const ap_uint<512> *A, 
    const ap_uint<512> *B, 
    ap_uint<512> *C, 
    int M, int N, int P
);
}

// --- Golden Reference (CPU) ---
void mmult_cpu(const std::vector<int8_t> &A, const std::vector<int8_t> &B, std::vector<int32_t> &C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < N; ++k) {
                // Standard Row-Major Multiply
                sum += (int32_t)A[i * N + k] * (int32_t)B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// --- PACKING A (Standard Block Tiling) ---
// Rearranges A into 32x32 blocks to match FPGA reading order
void pack_A_tiled(const std::vector<int8_t>& input, std::vector<ap_uint<512>>& packed_out, int Rows, int Cols) {
    int num_tiles_R = Rows / SA_DIM;
    int num_tiles_C = Cols / SA_DIM;

    for(int tr = 0; tr < num_tiles_R; tr++) {
        for(int tc = 0; tc < num_tiles_C; tc++) {
            // Pack 2 rows per 512-bit packet
            for(int i = 0; i < SA_DIM; i += 2) {
                ap_uint<512> packet = 0;
                // Row i
                for(int j = 0; j < 32; j++) {
                    int global_r = tr * SA_DIM + i;
                    int global_c = tc * SA_DIM + j;
                    uint8_t val = (uint8_t)input[global_r * Cols + global_c];
                    packet.range(8*(j+1)-1, 8*j) = val;
                }
                // Row i+1
                for(int j = 0; j < 32; j++) {
                    int global_r = tr * SA_DIM + (i+1);
                    int global_c = tc * SA_DIM + j;
                    uint8_t val = (uint8_t)input[global_r * Cols + global_c];
                    packet.range(8*(j+33)-1, 8*(j+32)) = val;
                }
                packed_out.push_back(packet);
            }
        }
    }
}

// --- PACKING B (Special Hybrid Packing) ---
// 1. Order of Tiles: Column-Major (to match Transposed read order: P then N)
// 2. Data inside Packet: Row-Major (to match Compute need: B[k][j])
void pack_B_special(const std::vector<int8_t>& B_standard, std::vector<ap_uint<512>>& packed_out, int N, int P) {
    int num_tiles_N = N / SA_DIM; // Depth (K)
    int num_tiles_P = P / SA_DIM; // Cols (J)

    // Outer Loops match Kernel's Read Order: Tiles of P, then Tiles of N
    for(int tc = 0; tc < num_tiles_P; tc++) {      
        for(int tk = 0; tk < num_tiles_N; tk++) {  
            
            // Inside the tile, we pack ROWS of B (varying J, fixed K)
            for(int k = 0; k < SA_DIM; k += 2) {
                ap_uint<512> packet = 0;
                
                // Row k
                for(int j = 0; j < 32; j++) {
                    int global_k = tk * SA_DIM + k;
                    int global_p = tc * SA_DIM + j;
                    uint8_t val = (uint8_t)B_standard[global_k * P + global_p];
                    packet.range(8*(j+1)-1, 8*j) = val;
                }
                
                // Row k+1
                for(int j = 0; j < 32; j++) {
                    int global_k = tk * SA_DIM + (k+1);
                    int global_p = tc * SA_DIM + j;
                    uint8_t val = (uint8_t)B_standard[global_k * P + global_p];
                    packet.range(8*(j+33)-1, 8*(j+32)) = val;
                }
                packed_out.push_back(packet);
            }
        }
    }
}

// --- UNPACKING C (Tiled Output -> Linear) ---
void unpack_tiled(const std::vector<ap_uint<512>>& packed_in, std::vector<int32_t>& output, int Rows, int Cols) {
    int num_tiles_R = Rows / SA_DIM;
    int num_tiles_C = Cols / SA_DIM;
    
    // Safety check: ensure we don't read past the actual data even if buffer is huge
    int expected_packets = (Rows * Cols) / 16; 
    
    int packet_idx = 0;
    for(int tr = 0; tr < num_tiles_R; tr++) {
        for(int tc = 0; tc < num_tiles_C; tc++) {
            // Each SA_DIM row of output takes packets
            for(int i = 0; i < SA_DIM; i++) {
                // One row (32 ints) takes 2 packets (16 ints each)
                for(int p = 0; p < 2; p++) {
                   ap_uint<512> packet = packed_in[packet_idx++];
                   for(int v=0; v<16; v++) {
                       int global_r = tr * SA_DIM + i;
                       int global_c = tc * SA_DIM + (p*16 + v);
                       output[global_r * Cols + global_c] = (int32_t)packet.range(32*(v+1)-1, 32*v);
                   }
                }
            }
        }
    }
}

int main() {
    std::cout << "Starting Int8 MMULT Test (Padding Enabled)..." << std::endl;

    // 1. Initialize Host Data
    std::vector<int8_t> A(M_TEST * N_TEST);
    std::vector<int8_t> B(N_TEST * P_TEST);
    std::vector<int32_t> C_sw(M_TEST * P_TEST, 0);
    std::vector<int32_t> C_hw(M_TEST * P_TEST, 0);

    // Random Init
    srand(42);
    for (int i = 0; i < A.size(); i++) A[i] = (rand() % 10);
    for (int i = 0; i < B.size(); i++) B[i] = (rand() % 10);

    // 2. Pack Data for FPGA
    std::vector<ap_uint<512>> A_hw_packed;
    std::vector<ap_uint<512>> B_hw_packed;
    std::vector<ap_uint<512>> C_hw_packed;

    pack_A_tiled(A, A_hw_packed, M_TEST, N_TEST);
    pack_B_special(B, B_hw_packed, N_TEST, P_TEST);

    // ---------------------------------------------------------
    // CRITICAL FIX FOR CO-SIM SIGSEGV
    // The Kernel Interface has `depth=4096`.
    // The C-Sim wrapper expects vectors of size 4096.
    // We must resize (pad with zeros) to match the interface depth.
    // ---------------------------------------------------------
    if (A_hw_packed.size() < COSIM_BUFFER_DEPTH) A_hw_packed.resize(COSIM_BUFFER_DEPTH, 0);
    if (B_hw_packed.size() < COSIM_BUFFER_DEPTH) B_hw_packed.resize(COSIM_BUFFER_DEPTH, 0);
    
    // Output buffer must also be sized to the interface depth
    C_hw_packed.resize(COSIM_BUFFER_DEPTH, 0);
    // ---------------------------------------------------------

    // 3. Run Hardware Kernel
    std::cout << "Running HW Kernel..." << std::endl;
    mmult_systolic(A_hw_packed.data(), B_hw_packed.data(), C_hw_packed.data(), M_TEST, N_TEST, P_TEST);

    // 4. Verification
    // Note: We unpack only the relevant data, ignoring the padded zeros
    unpack_tiled(C_hw_packed, C_hw, M_TEST, P_TEST);
    mmult_cpu(A, B, C_sw, M_TEST, N_TEST, P_TEST);

    int errors = 0;
    for (int i = 0; i < M_TEST * P_TEST; i++) {
        if (C_hw[i] != C_sw[i]) {
            errors++;
            if (errors < 5) {
                std::cout << "Mismatch at " << i << ": HW=" << C_hw[i] << " SW=" << C_sw[i] << std::endl;
            }
        }
    }

    if (errors == 0) {
        std::cout << "PASSED! Matrix Multiplication Successful." << std::endl;
        return 0;
    } else {
        std::cout << "FAILED: " << errors << " errors found." << std::endl;
        return 1;
    }
}