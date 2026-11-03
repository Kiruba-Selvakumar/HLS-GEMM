#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>

#define MAX_M 256
#define MAX_N 256
#define MAX_P 256
#define TILE_SIZE 32 // 4x4 tiling

extern "C" {
void mmult(
    const int16_t *A,   // Flattened arrays for proper host-device mapping
    const int16_t *B,
    int32_t *C,
    int M,
    int N,
    int P)
{
    // =========================================================
    // Interface Pragmas
    // =========================================================
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

    // =========================================================
    // Local Buffers for Tiles
    // =========================================================
    int16_t A_local[TILE_SIZE][TILE_SIZE];
    int16_t B_local[TILE_SIZE][TILE_SIZE];
    int32_t C_local[TILE_SIZE][TILE_SIZE]; // 32-bit accumulation to avoid overflow

#pragma HLS ARRAY_PARTITION variable=A_local complete dim=0
#pragma HLS ARRAY_PARTITION variable=B_local complete dim=0
#pragma HLS ARRAY_PARTITION variable=C_local complete dim=0

    // =========================================================
    // Tiled Matrix Multiplication
    // =========================================================
Row_Tile:
    for (int i_tile = 0; i_tile < M; i_tile += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_M / TILE_SIZE)
    Col_Tile:
        for (int j_tile = 0; j_tile < P; j_tile += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_P / TILE_SIZE)

            // Initialize C_local tile
        Init_C_Tile:
            for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS PIPELINE II=1
                    C_local[i][j] = 0;
                }
            }

            // Loop over K dimension tiles
        K_Tile:
            for (int k_tile = 0; k_tile < N; k_tile += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_N / TILE_SIZE)

                // Load A tile
            Load_A_Tile:
                for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                    for (int k = 0; k < TILE_SIZE; k++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS PIPELINE II=1
                        int a_row = i_tile + i;
                        int a_col = k_tile + k;
                        if (a_row < M && a_col < N)
                            A_local[i][k] = A[a_row * N + a_col];
                        else
                            A_local[i][k] = 0;
                    }
                }

                // Load B tile
            Load_B_Tile:
                for (int k = 0; k < TILE_SIZE; k++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS PIPELINE II=1
                        int b_row = k_tile + k;
                        int b_col = j_tile + j;
                        if (b_row < N && b_col < P)
                            B_local[k][j] = B[b_row * P + b_col];
                        else
                            B_local[k][j] = 0;
                    }
                }

                // Compute local tile (DSP-optimized)
            Compute_Tile:
                for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                        int32_t sum = C_local[i][j];
#pragma HLS UNROLL
                        for (int k = 0; k < TILE_SIZE; k++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
#pragma HLS UNROLL
                            int32_t prod = (int32_t)A_local[i][k] * (int32_t)B_local[k][j];
#pragma HLS RESOURCE variable=prod core=Mul_DSP
                            sum += prod;
                        }
                        C_local[i][j] = sum;
                    }
                }
            } // End K_Tile

            // Write C_local back to global memory
        Write_C_Tile:
            for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
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

