/**
* Corrected version of the tiled matrix multiplication kernel.
* This version applies BIND_OP to the entire function
* to force the tool to use DSP slices for multiplication.
*/

#include <hls_stream.h>
#include <string.h> // For memcpy

// --- Constants defined from the paper ---
#define TILE_SIZE 32 
#define BLOCK_M 256 
#define MAX_N 64
#define MAX_K 768

typedef char T; // int8 data type 

// --- Persistent on-chip buffer for Matrix A ---
static T bram_a[MAX_N][MAX_K];

/**
 * @brief The core compute engine (32x32 unrolled array).
 * *** DSP USAGE FORCED WITH BIND_OP ***
 */
static void compute_tile(T local_a[TILE_SIZE][TILE_SIZE],
                         T local_b[TILE_SIZE][TILE_SIZE],
                         int local_c[TILE_SIZE][TILE_SIZE]) {
    
    // --- START: MODIFIED CODE ---
    // Force HLS to use a DSP48 block for ALL 'mul' operations
    // within this function's scope.
    #pragma HLS BIND_OP op=mul impl=dsp
    // --- END: MODIFIED CODE ---

K_LOOP:
    for (int k = 0; k < TILE_SIZE; k++) {
    #pragma HLS PIPELINE II=1 
    I_LOOP:
        for (int i = 0; i < TILE_SIZE; i++) {
        #pragma HLS UNROLL 
        J_LOOP:
            for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
                // The pragma above will now apply to this multiplication
                local_c[i][j] += local_a[i][k] * local_b[k][j];
            }
        }
    }
}

/**
 * @brief The main dataflow pipeline for processing.
 * This function contains the processing loops and will be
 * synthesized as a dataflow region.
 */
static void mmult_dataflow(const T* b, int* c, int N, int K, int M) {
    // This pragma now applies to this function's body
#pragma HLS DATAFLOW

    // On-chip buffers for B and C
    T bram_b[MAX_K][BLOCK_M];
    int bram_c[TILE_SIZE][BLOCK_M];

    // Partition the A buffer for high-bandwidth reads
#pragma HLS ARRAY_PARTITION variable=bram_a complete dim=1

// Outer Tiling Loop (over columns of B)
J_BLOCK_LOOP:
    for (int j_block = 0; j_block < M; j_block += BLOCK_M) {
        #pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
        int current_block_m = (j_block + BLOCK_M > M) ? (M - j_block) : BLOCK_M;

    // Load a block of B from DDR into on-chip BRAM
    LOAD_B_BLOCK:
        for (int k = 0; k < K; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=768 max=768 avg=768
            for (int j = 0; j < current_block_m; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
            #pragma HLS PIPELINE II=1
                bram_b[k][j] = b[k * M + j_block + j];
            }
        }

    // Inner Tiling Loops 
    I_TILE_LOOP:
        for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
            #pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2
        J_TILE_LOOP:
            for (int j0 = 0; j0 < current_block_m; j0 += TILE_SIZE) {
                #pragma HLS LOOP_TRIPCOUNT min=8 max=8 avg=8
                // Local tile buffers (will become registers)
                T local_a[TILE_SIZE][TILE_SIZE];
                T local_b[TILE_SIZE][TILE_SIZE];
                int local_c[TILE_SIZE][TILE_SIZE];
            #pragma HLS ARRAY_PARTITION variable=local_a complete dim=0
            #pragma HLS ARRAY_PARTITION variable=local_b complete dim=0

                // Initialize local C buffer 
                for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
                    #pragma HLS PIPELINE
                        local_c[i][j] = 0;
                    }
                }

            // K-dimension tile loop 
            K_TILE_LOOP:
                for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                    #pragma HLS LOOP_TRIPCOUNT min=24 max=24 avg=24
                    // Load tiles from BRAMs into local registers
                    for (int i = 0; i < TILE_SIZE; i++) {
                        for (int k = 0; k < TILE_SIZE; k++) {
                        #pragma HLS PIPELINE
                            local_a[i][k] = bram_a[i0 + i][k0 + k];
                        }
                    }
                    for (int k = 0; k < TILE_SIZE; k++) {
                        for (int j = 0; j < TILE_SIZE; j++) {
                        #pragma HLS PIPELINE
                            local_b[k][j] = bram_b[k0 + k][j0 + j];
                        }
                    }
                    // Call the unrolled compute engine
                    compute_tile(local_a, local_b, local_c);
                }

                // Write the computed tile back to DDR 
                for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
                    #pragma HLS PIPELINE
                        if ((i0 + i < N) && (j0 + j < current_block_m)) {
                            c[(i0 + i) * M + j_block + j0 + j] = local_c[i][j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Top-level kernel function called by the host.
 * Handles AXI interfaces and the conditional load of Matrix A.
 */
extern "C" {
void mmult(const T* a, const T* b, int* c, int N, int K, int M, bool update_A) {
    // AXI Interfaces
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem2

#pragma HLS INTERFACE s_axilite port = a
#pragma HLS INTERFACE s_axilite port = b
#pragma HLS INTERFACE s_axilite port = c
#pragma HLS INTERFACE s_axilite port = N
#pragma HLS INTERFACE s_axilite port = K
#pragma HLS INTERFACE s_axilite port = M
#pragma HLS INTERFACE s_axilite port = update_A 
#pragma HLS INTERFACE s_axilite port = return

    // 1. Conditional load of Matrix A (Setup step) 
    if (update_A) {
    LOAD_A_OUTER:
        for (int i = 0; i < N; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
        LOAD_A_INNER:
            for (int j = 0; j < K; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=768 max=768 avg=768
            #pragma HLS PIPELINE II=1
                bram_a[i][j] = a[i * K + j];
            }
        }
    }

    // 2. Call the dataflow function to do the processing
    mmult_dataflow(b, c, N, K, M);
}
}

