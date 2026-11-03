#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>

#define SA_DIM 32 
#define VEC_WIDTH_IN 64 
#define VEC_WIDTH_OUT 16

extern "C" {

// --------------------------------------------------------
// 1. Read Engine A (Updated for Tiled Layout)
// --------------------------------------------------------
void read_A(const ap_uint<512>* A, hls::stream<ap_uint<512>>& stream_A, int M, int N, int P) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_N = N / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    
    // One tile (32x32 int8) = 1024 bytes.
    // 512-bit interface = 64 bytes per packet.
    // Packets per tile = 1024 / 64 = 16 packets.
    const int PACKETS_PER_TILE = 16;

    for (int r = 0; r < num_tiles_M; r++) {
        for (int c = 0; c < num_tiles_P; c++) {
            for (int k = 0; k < num_tiles_N; k++) {
                
                // Calculate which Block to read from Tiled Memory
                // A is tiled by [Tile_M][Tile_N]
                int block_idx = r * num_tiles_N + k;
                int offset = block_idx * PACKETS_PER_TILE;

                // Burst read the tile
                for (int i = 0; i < PACKETS_PER_TILE; i++) {
                    #pragma HLS PIPELINE II=1
                    stream_A.write(A[offset + i]);
                }
            }
        }
    }
}

// --------------------------------------------------------
// 2. Read Engine B (Updated for Tiled Layout)
// --------------------------------------------------------
void read_B(const ap_uint<512>* B, hls::stream<ap_uint<512>>& stream_B, int M, int N, int P) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_N = N / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    const int PACKETS_PER_TILE = 16;

    for (int r = 0; r < num_tiles_M; r++) {
        for (int c = 0; c < num_tiles_P; c++) {
            for (int k = 0; k < num_tiles_N; k++) {
                
                // B is Transposed PxN. Tiled by [Tile_P][Tile_N]
                // We are at loop indices: r (M), c (P), k (N).
                // We need Tile B[c][k]
                int block_idx = c * num_tiles_N + k;
                int offset = block_idx * PACKETS_PER_TILE;

                for (int i = 0; i < PACKETS_PER_TILE; i++) {
                    #pragma HLS PIPELINE II=1
                    stream_B.write(B[offset + i]);
                }
            }
        }
    }
}

// --------------------------------------------------------
// 3. Compute Engine (Unchanged - Logic is correct)
// --------------------------------------------------------
void compute_gemm(
    hls::stream<ap_uint<512>>& stream_A, 
    hls::stream<ap_uint<512>>& stream_B, 
    hls::stream<ap_uint<512>>& stream_C,
    int M, int N, int P
) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    int num_tiles_N = N / SA_DIM;

    ap_int<8> A_local[SA_DIM][SA_DIM];
    ap_int<8> B_local[SA_DIM][SA_DIM]; 
    ap_int<32> C_local[SA_DIM][SA_DIM];

    #pragma HLS ARRAY_PARTITION variable=A_local dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=B_local dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=C_local dim=0 complete

    for (int r = 0; r < num_tiles_M; r++) {
        for (int c = 0; c < num_tiles_P; c++) {
            
            Init_C:
            for(int i=0; i<SA_DIM; i++)
                for(int j=0; j<SA_DIM; j++)
                     C_local[i][j] = 0;

            for (int k = 0; k < num_tiles_N; k++) {
                
                // Load A: Unpacking 2 rows per packet
                Load_A_Local:
                for (int i = 0; i < SA_DIM; i += 2) { 
                    #pragma HLS PIPELINE II=1
                    ap_uint<512> packet = stream_A.read();
                    for (int v = 0; v < 64; ++v) {
                        int row_offset = v / 32; 
                        int col_idx = v % 32;
                        A_local[i + row_offset][col_idx] = packet.range(8*(v+1)-1, 8*v);
                    }
                }

                // Load B: Unpacking 2 rows per packet
                Load_B_Local:
                for (int j = 0; j < SA_DIM; j += 2) {
                    #pragma HLS PIPELINE II=1
                    ap_uint<512> packet = stream_B.read();
                    for (int v = 0; v < 64; ++v) {
                        int row_offset = v / 32; 
                        int col_idx = v % 32;
                        B_local[j + row_offset][col_idx] = packet.range(8*(v+1)-1, 8*v);
                    }
                }

                Compute_Core:
                for (int kk = 0; kk < SA_DIM; ++kk) {
                    #pragma HLS PIPELINE II=1
                    for (int i = 0; i < SA_DIM; ++i) {
                        for (int j = 0; j < SA_DIM; ++j) {
                            C_local[i][j] += A_local[i][kk] * B_local[kk][j];
                        }
                    }
                }
            } 

            // Write C (Linear Packing for Output is okay, or we can tile it too. 
            // For simplicity, we keep linear stream write, but Host needs to handle it.
            // Actually, let's keep simpler Block output to match input style)
            Store_C_Local:
            for (int i = 0; i < SA_DIM; i++) {
                for (int j = 0; j < SA_DIM; j += VEC_WIDTH_OUT) {
                    #pragma HLS PIPELINE II=1
                    ap_uint<512> out_packet;
                    for (int v = 0; v < VEC_WIDTH_OUT; v++) {
                        out_packet.range(32*(v+1)-1, 32*v) = C_local[i][j+v];
                    }
                    stream_C.write(out_packet);
                }
            }
        }
    }
}

// --------------------------------------------------------
// 4. Write Engine C (Tiled Output)
// --------------------------------------------------------
void write_C(ap_uint<512>* C, hls::stream<ap_uint<512>>& stream_C, int M, int N, int P) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    // Output C is Int32. One tile is 32x32 ints.
    // 512 bit packet = 16 ints.
    // Packets per tile = (32*32)/16 = 64 packets.
    const int PACKETS_PER_TILE = 64;

    for (int r = 0; r < num_tiles_M; r++) {
        for (int c = 0; c < num_tiles_P; c++) {
            
            // Calculate Block Index for Output
            int block_idx = r * num_tiles_P + c;
            int offset = block_idx * PACKETS_PER_TILE;

            for (int i = 0; i < PACKETS_PER_TILE; i++) {
                #pragma HLS PIPELINE II=1
                C[offset + i] = stream_C.read();
            }
        }
    }
}

// Top Level
void mmult_systolic(
    const ap_uint<512> *A,    
    const ap_uint<512> *B,    
    ap_uint<512> *C,          
    int M, int N, int P
) {
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=64 num_read_outstanding=16 depth=4096
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=64 num_read_outstanding=16 depth=4096
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_write_burst_length=64 num_write_outstanding=16 depth=4096
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=M bundle=control
    #pragma HLS INTERFACE s_axilite port=N bundle=control
    #pragma HLS INTERFACE s_axilite port=P bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW 
    hls::stream<ap_uint<512>> StreamA;
    #pragma HLS STREAM variable=StreamA depth=256
    hls::stream<ap_uint<512>> StreamB;
    #pragma HLS STREAM variable=StreamB depth=256
    hls::stream<ap_uint<512>> StreamC;
    #pragma HLS STREAM variable=StreamC depth=256

    read_A(A, StreamA, M, N, P);
    read_B(B, StreamB, M, N, P);
    compute_gemm(StreamA, StreamB, StreamC, M, N, P);
    write_C(C, StreamC, M, N, P);
}
}