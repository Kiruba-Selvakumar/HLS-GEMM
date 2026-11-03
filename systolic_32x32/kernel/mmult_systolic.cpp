#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>

// --- Architecture Configuration ---
// PERFORMANCE BOOST 1: Increase SA_DIM to 32.
// This utilizes ~1024 DSPs (32x32) instead of ~256 (16x16).
#define SA_DIM 32 

// Block sizes for memory tiling
#define TILE_M SA_DIM
#define TILE_N SA_DIM
#define TILE_P SA_DIM

// Maximum Matrix Size (for simulation/tripcounts)
#define MAX_SIZE 1024

// Vector width for 512-bit memory access (16 floats x 32 bit)
#define VEC_WIDTH 16

// Helper for bit-cast (float <-> uint32) without type violation
union fp_bit_cast {
    uint32_t i;
    float f;
};

extern "C" {

/**
 * mmult_systolic_optimized
 * * PERFORMANCE NOTES:
 * 1. Wide Interface (ap_uint<512>): Loads 16 floats per cycle.
 * 2. Transposed B: Assumes Matrix B is Transposed on the Host (B^T) to allow fast row-reads.
 * 3. 32x32 Array: Higher DSP utilization.
 */
void mmult_systolic(
    const ap_uint<512> *A,    // Read-only A (Wide Port)
    const ap_uint<512> *B,    // Read-only B (Wide Port) - MUST BE TRANSPOSED!
    ap_uint<512> *C,          // Output C (Wide Port)
    int M,                    // Rows of A
    int N,                    // Cols of A / Rows of B
    int P                     // Cols of B
    )
{
    // Port configurations
    // PERFORMANCE BOOST 2: 512-bit width (loading 16 floats at once)
    // Depth calculation: 4096 floats / 16 floats-per-vec = 256 vectors
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=256 latency=64
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=256 latency=64
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=256 latency=64
    
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=M bundle=control
#pragma HLS INTERFACE s_axilite port=N bundle=control
#pragma HLS INTERFACE s_axilite port=P bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // -----------------------------------------------------------------------
    // Local BRAM Buffers
    // -----------------------------------------------------------------------
    float A_local[SA_DIM][MAX_SIZE]; 
    #pragma HLS ARRAY_PARTITION variable=A_local complete dim=1

    float B_local[SA_DIM][MAX_SIZE];
    #pragma HLS ARRAY_PARTITION variable=B_local complete dim=1

    float C_local[SA_DIM][SA_DIM];
    #pragma HLS ARRAY_PARTITION variable=C_local complete dim=0

    // -----------------------------------------------------------------------
    // Processing Loops
    // -----------------------------------------------------------------------
    for (int r = 0; r < M; r += SA_DIM) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=64
        for (int c = 0; c < P; c += SA_DIM) {
            #pragma HLS LOOP_TRIPCOUNT min=64 max=64

            // 1. Initialize Accumulator Registers (The PE Array)
            Init_PEs:
            for(int i=0; i<SA_DIM; i++) {
                #pragma HLS UNROLL
                for(int j=0; j<SA_DIM; j++) {
                    #pragma HLS UNROLL
                    C_local[i][j] = 0.0f;
                }
            }

            // 2. Iterate across the depth (N dimension) in chunks
            for (int k_base = 0; k_base < N; k_base += MAX_SIZE) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1
                
                int chunk_size = (N - k_base > MAX_SIZE) ? MAX_SIZE : (N - k_base);

                // --- DATA LOAD PHASE (VECTORIZED) ---
                
                // Load A: Reads Rows. A_local[i][k]
                // We read 'chunk_size' elements (N dimension).
                // We process VEC_WIDTH (16) elements per clock cycle.
                Load_A:
                for (int i = 0; i < SA_DIM; i++) {
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    int global_row = r + i;
                    
                    // Iterate through columns in chunks of 16 (512 bits)
                    for (int k = 0; k < chunk_size; k += VEC_WIDTH) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                        
                        if (global_row < M && (k_base + k) < N) {
                            // Calculate vector index
                            // Address = (Row * TotalCols + Col) / 16
                            int vec_idx = (global_row * N + (k_base + k)) / VEC_WIDTH;
                            ap_uint<512> raw_vec = A[vec_idx];

                            // Unpack 512 bits into 16 floats
                            for (int v = 0; v < VEC_WIDTH; v++) {
                                #pragma HLS UNROLL
                                if (k + v < chunk_size) {
                                    uint32_t raw_bits = raw_vec.range(32*(v+1)-1, 32*v);
                                    fp_bit_cast converter;
                                    converter.i = raw_bits;
                                    A_local[i][k+v] = converter.f;
                                }
                            }
                        } else {
                            // Zero padding
                            for (int v = 0; v < VEC_WIDTH; v++) { 
                                #pragma HLS UNROLL
                                if (k + v < chunk_size) A_local[i][k+v] = 0.0f;
                            }
                        }
                    }
                }

                // Load B: Assumes Host Transposed B!
                // We read Rows of B^T (which are Cols of B).
                Load_B:
                for (int j = 0; j < SA_DIM; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    int global_col = c + j; // This corresponds to a Row in B^T
                    
                    for (int k = 0; k < chunk_size; k += VEC_WIDTH) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                        
                        // NOTE: Index calculation assumes B is PxN (Transposed layout)
                        // Address = (GlobalCol * N + K) / 16
                        if (global_col < P && (k_base + k) < N) {
                            int vec_idx = (global_col * N + (k_base + k)) / VEC_WIDTH;
                            ap_uint<512> raw_vec = B[vec_idx];

                            for (int v = 0; v < VEC_WIDTH; v++) {
                                #pragma HLS UNROLL
                                if (k + v < chunk_size) {
                                    uint32_t raw_bits = raw_vec.range(32*(v+1)-1, 32*v);
                                    fp_bit_cast converter;
                                    converter.i = raw_bits;
                                    B_local[j][k+v] = converter.f;
                                }
                            }
                        } else {
                            for (int v = 0; v < VEC_WIDTH; v++) {
                                #pragma HLS UNROLL
                                if (k + v < chunk_size) B_local[j][k+v] = 0.0f;
                            }
                        }
                    }
                }

                // --- SYSTOLIC COMPUTE PHASE ---
                float val_A[SA_DIM][SA_DIM];
                #pragma HLS ARRAY_PARTITION variable=val_A complete dim=0
                float val_B[SA_DIM][SA_DIM];
                #pragma HLS ARRAY_PARTITION variable=val_B complete dim=0

                Init_Shift_Regs:
                for(int i=0; i<SA_DIM; i++) {
                    #pragma HLS UNROLL
                    for(int j=0; j<SA_DIM; j++) {
                        #pragma HLS UNROLL
                        val_A[i][j] = 0.0f;
                        val_B[i][j] = 0.0f;
                    }
                }

                Systolic_Run:
                for (int t = 0; t < chunk_size + (2 * SA_DIM); t++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1088 max=1088

                    // Step 1: Shift Data
                    for (int i = 0; i < SA_DIM; i++) {
                        #pragma HLS UNROLL
                        for (int j = SA_DIM - 1; j >= 1; j--) {
                            #pragma HLS UNROLL
                            val_A[i][j] = val_A[i][j-1]; 
                        }
                    }
                    for (int j = 0; j < SA_DIM; j++) {
                        #pragma HLS UNROLL
                        for (int i = SA_DIM - 1; i >= 1; i--) {
                            #pragma HLS UNROLL
                            val_B[i][j] = val_B[i-1][j]; 
                        }
                    }

                    // Step 2: Feed New Data
                    for (int i = 0; i < SA_DIM; i++) {
                        #pragma HLS UNROLL
                        int k_idx = t - i; 
                        if (k_idx >= 0 && k_idx < chunk_size) {
                            val_A[i][0] = A_local[i][k_idx];
                        } else {
                            val_A[i][0] = 0.0f;
                        }
                    }
                    for (int j = 0; j < SA_DIM; j++) {
                        #pragma HLS UNROLL
                        int k_idx = t - j; 
                        if (k_idx >= 0 && k_idx < chunk_size) {
                            val_B[0][j] = B_local[j][k_idx];
                        } else {
                            val_B[0][j] = 0.0f;
                        }
                    }

                    // Step 3: Compute
                    for (int i = 0; i < SA_DIM; i++) {
                        #pragma HLS UNROLL
                        for (int j = 0; j < SA_DIM; j++) {
                            #pragma HLS UNROLL
                            // Force DSP usage
                            #pragma HLS BIND_OP variable=C_local op=fmul impl=dsp
                            #pragma HLS BIND_OP variable=C_local op=fadd impl=dsp
                            C_local[i][j] += val_A[i][j] * val_B[i][j];
                        }
                    }
                } 
            } // End K-loop

            // 3. Write Back Phase (VECTORIZED)
            Write_C:
            for (int i = 0; i < SA_DIM; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                int global_row = r + i;
                
                // Write in chunks of 16
                for (int j = 0; j < SA_DIM; j += VEC_WIDTH) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                    
                    if (global_row < M) { // Simple check, handling partial C writes is complex in vector mode
                        int vec_idx = (global_row * P + (c + j)) / VEC_WIDTH;
                        ap_uint<512> out_vec;
                        
                        for (int v = 0; v < VEC_WIDTH; v++) {
                            #pragma HLS UNROLL
                            // Pack float to uint32
                            if (j + v < SA_DIM && (c + j + v) < P) {
                                fp_bit_cast converter;
                                converter.f = C_local[i][j+v];
                                out_vec.range(32*(v+1)-1, 32*v) = converter.i;
                            } else {
                                out_vec.range(32*(v+1)-1, 32*v) = 0;
                            }
                        }
                        C[vec_idx] = out_vec;
                    }
                }
            }

        } // End Col_Tile
    } // End Row_Tile
}
}