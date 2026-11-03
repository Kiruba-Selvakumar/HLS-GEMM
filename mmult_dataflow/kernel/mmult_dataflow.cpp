#include <hls_stream.h>
#include <stdint.h>

#define MAX_M 256
#define MAX_N 256
#define MAX_P 256
#define TILE_SIZE 32

extern "C" {
void mmult(
    const float *A,    // input matrix A (MxN)
    const float *B,    // input matrix B (NxP)
    float *C,          // output matrix C (MxP)
    int M,
    int N,
    int P)
{
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

    // Local accumulator for one TILE_SIZE x TILE_SIZE block of C
    float C_local[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=C_local complete dim=0

Row_Tile: for (int i_tile = 0; i_tile < M; i_tile += TILE_SIZE) {
    #pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_M/TILE_SIZE)
Col_Tile: for (int j_tile = 0; j_tile < P; j_tile += TILE_SIZE) {
    #pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_P/TILE_SIZE)

    // --- Initialize C_local tile to zero ---
Init_C: 
    for (int i = 0; i < TILE_SIZE; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
            #pragma HLS PIPELINE II=1
            // Zero-out only valid entries (others also set to zero harmlessly)
            C_local[i][j] = 0.0f;
        }
    }

    // Create streams for A-tiles and B-tiles.
    // These act as FIFOs between load and compute tasks.
    hls::stream<float> A_str;
    hls::stream<float> B_str;
    // Small FIFO depths to allow buffering between tasks
    #pragma HLS STREAM variable=A_str depth=16
    #pragma HLS STREAM variable=B_str depth=16

    // --- Dataflow region: Load tiles and Compute overlap ---
    {
        #pragma HLS DATAFLOW

        // (1) Load all A-tiles for this i_tile into A_str stream
LoadA_stream:
        for (int k_tile = 0; k_tile < N; k_tile += TILE_SIZE) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_N/TILE_SIZE)
            for (int i = 0; i < TILE_SIZE; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                for (int k = 0; k < TILE_SIZE; k++) {
                    #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                    #pragma HLS PIPELINE II=1
                    int a_row = i_tile + i;
                    int a_col = k_tile + k;
                    float a_val = 0.0f;
                    if (a_row < M && a_col < N) {
                        a_val = A[a_row * N + a_col];
                    }
                    A_str.write(a_val);
                }
            }
        }

        // (2) Load all B-tiles for this j_tile into B_str stream
LoadB_stream:
        for (int k_tile = 0; k_tile < N; k_tile += TILE_SIZE) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_N/TILE_SIZE)
            for (int k = 0; k < TILE_SIZE; k++) {
                #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                for (int j = 0; j < TILE_SIZE; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                    #pragma HLS PIPELINE II=1
                    int b_row = k_tile + k;
                    int b_col = j_tile + j;
                    float b_val = 0.0f;
                    if (b_row < N && b_col < P) {
                        b_val = B[b_row * P + b_col];
                    }
                    B_str.write(b_val);
                }
            }
        }

        // (3) Compute: read tiles from streams and accumulate into C_local
Compute_tile:
        for (int k_tile = 0; k_tile < N; k_tile += TILE_SIZE) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_N/TILE_SIZE)
            // Local buffers for one TILE from streams
            float A_buf[TILE_SIZE][TILE_SIZE];
            float B_buf[TILE_SIZE][TILE_SIZE];
            #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=0
            #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=0

            // Read one TILE from A_str
        ReadA:
            for (int i = 0; i < TILE_SIZE; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                for (int k = 0; k < TILE_SIZE; k++) {
                    #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                    #pragma HLS PIPELINE II=1
                    A_buf[i][k] = A_str.read();
                }
            }
            // Read one TILE from B_str
        ReadB:
            for (int k = 0; k < TILE_SIZE; k++) {
                #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                for (int j = 0; j < TILE_SIZE; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                    #pragma HLS PIPELINE II=1
                    B_buf[k][j] = B_str.read();
                }
            }
            // Perform tile multiply-accumulate: C_local[i][j] += sum_k A_buf[i][k]*B_buf[k][j]
        Compute_inner:
            for (int i = 0; i < TILE_SIZE; i++) {
                #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                for (int j = 0; j < TILE_SIZE; j++) {
                    #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
                    #pragma HLS PIPELINE II=1
                    float sum = C_local[i][j];
                    for (int k = 0; k < TILE_SIZE; k++) {
                        #pragma HLS UNROLL
                        sum += A_buf[i][k] * B_buf[k][j];
                    }
                    C_local[i][j] = sum;
                }
            }
        } // end Compute_tile loop
    } // end DATAFLOW region

    // --- Write the accumulated tile back to global memory ---
Write_C:
    for (int i = 0; i < TILE_SIZE; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=TILE_SIZE max=TILE_SIZE
            #pragma HLS PIPELINE II=1
            int c_row = i_tile + i;
            int c_col = j_tile + j;
            if (c_row < M && c_col < P) {
                C[c_row * P + c_col] = C_local[i][j];
            }
        }
    }

} // end Col_Tile
} // end Row_Tile

} // end mmult()
} // end extern "C"

