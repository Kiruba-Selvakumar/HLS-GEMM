#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>

#define MAX_M 256
#define MAX_N 256
#define MAX_P 256
#define TILE_SIZE 16 // Keeping 16 to manage resource usage with FP32

extern "C" {
void mmult(
    const float *A,  // Changed to float (FP32)
    const float *B,  // Changed to float (FP32)
    float *C,        // Changed to float (FP32)
    int M,
    int N,
    int P)
{
    // Interface Pragmas
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=65536
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=65536
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=65536
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=M bundle=control
#pragma HLS INTERFACE s_axilite port=N bundle=control
#pragma HLS INTERFACE s_axilite port=P bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local Buffers (All changed to float)
    float A_local[TILE_SIZE][TILE_SIZE];
    float B_local[TILE_SIZE][TILE_SIZE];
    float C_local[TILE_SIZE][TILE_SIZE];

    // Partitioning remains the same for parallel access
#pragma HLS ARRAY_PARTITION variable=A_local complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=4 dim=2


    // Main Processing Loops
Row_Tile:
    for (int i_tile = 0; i_tile < M; i_tile += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_M / TILE_SIZE)
    Col_Tile:
        for (int j_tile = 0; j_tile < P; j_tile += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_P / TILE_SIZE)

            // 1. Initialize C_local
        Init_C_Tile:
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    C_local[i][j] = 0.0f; // Initialize with float zero
                }
            }

            // 2. Loop over K dimension (Accumulation)
        K_Tile:
            for (int k_tile = 0; k_tile < N; k_tile += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_N / TILE_SIZE)

                // 3. Load A Tile
            Load_A_Tile:
                for (int i = 0; i < TILE_SIZE; i++) {
                    for (int k = 0; k < TILE_SIZE; k++) {
#pragma HLS PIPELINE II=1
                        int a_row = i_tile + i;
                        int a_col = k_tile + k;
                        if (a_row < M && a_col < N)
                            A_local[i][k] = A[a_row * N + a_col];
                        else
                            A_local[i][k] = 0.0f;
                    }
                }

                // 4. Load B Tile
            Load_B_Tile:
                for (int k = 0; k < TILE_SIZE; k++) {
                    for (int j = 0; j < TILE_SIZE; j++) { 
#pragma HLS PIPELINE II=1
                        int b_row = k_tile + k;
                        int b_col = j_tile + j;
                        if (b_row < N && b_col < P)
                            B_local[k][j] = B[b_row * P + b_col];
                        else
                            B_local[k][j] = 0.0f;
                    }
                }

                // 5. Compute
            Compute_Tile:
                for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1 
                        float sum = C_local[i][j];
                        
                        // Unroll the reduction loop to do 16 FP32 MACs in parallel
                        for (int k = 0; k < TILE_SIZE; k++) {
#pragma HLS UNROLL
                            // FP32 Multiplication and Addition
                            // HLS will map this to DSPs automatically
                            sum += A_local[i][k] * B_local[k][j];
                        }
                        C_local[i][j] = sum;
                    }
                }
            } // End K_Tile

            // 6. Write Result
        Write_C_Tile:
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
                    int c_row = i_tile + i;
                    int c_col = j_tile + j;
                    if (c_row < M && c_col < P)
                        C[c_row * P + c_col] = C_local[i][j];
                }
            }
        } // End Col_Tile
    } // End Row_Tile
}
}
