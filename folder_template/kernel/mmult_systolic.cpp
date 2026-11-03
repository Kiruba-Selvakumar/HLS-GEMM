#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>

// --- Architecture Configuration ---
#define SA_DIM 32 
#define VEC_WIDTH 16 // 512 bits / 32-bit float = 16 floats per packet

extern "C" {

// --------------------------------------------------------
// 1. Read Engine A (Optimized)
// --------------------------------------------------------
void read_A_opt(const ap_uint<512>* A, hls::stream<ap_uint<512>>& stream_A, int M, int N, int P) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    int num_tiles_N = N / SA_DIM;
    
    // 32x32 floats = 4KB per tile. 
    // 4KB / 64 bytes (512-bit) = 64 packets
    const int PACKETS_PER_TILE = 64; 

    for (int r = 0; r < num_tiles_M; r++) {
        for (int c = 0; c < num_tiles_P; c++) {
            for (int k = 0; k < num_tiles_N; k++) {
                // A is Row-Major. Tile index (r, k)
                int block_idx = r * num_tiles_N + k;
                int offset = block_idx * PACKETS_PER_TILE;
                
                Read_A_Loop:
                for (int i = 0; i < PACKETS_PER_TILE; i++) {
                    #pragma HLS PIPELINE II=1
                    stream_A.write(A[offset + i]);
                }
            }
        }
    }
}

// --------------------------------------------------------
// 2. Read Engine B (FIXED: Standard Row-Major Layout)
// --------------------------------------------------------
void read_B_opt(const ap_uint<512>* B, hls::stream<ap_uint<512>>& stream_B, int M, int N, int P) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    int num_tiles_N = N / SA_DIM;
    const int PACKETS_PER_TILE = 64;

    for (int r = 0; r < num_tiles_M; r++) {
        for (int c = 0; c < num_tiles_P; c++) {
            for (int k = 0; k < num_tiles_N; k++) {
                // FIXED: Standard Row-Major Addressing for Matrix B
                // We need the tile at Row=k, Col=c. 
                // In linear memory (Row-Major), this is: k * (Total_Cols_In_Tiles) + c
                int block_idx = k * num_tiles_P + c; 
                
                int offset = block_idx * PACKETS_PER_TILE;

                Read_B_Loop:
                for (int i = 0; i < PACKETS_PER_TILE; i++) {
                    #pragma HLS PIPELINE II=1
                    stream_B.write(B[offset + i]);
                }
            }
        }
    }
}

// --------------------------------------------------------
// 3. Compute Engine (FP32 Systolic - Optimized)
// --------------------------------------------------------
void compute_gemm_opt(
    hls::stream<ap_uint<512>>& stream_A, 
    hls::stream<ap_uint<512>>& stream_B, 
    hls::stream<ap_uint<512>>& stream_C,
    int M, int N, int P
) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    int num_tiles_N = N / SA_DIM;

    // Local Buffers
    float A_local[SA_DIM][SA_DIM];
    float B_local[SA_DIM][SA_DIM]; 
    float C_local[SA_DIM][SA_DIM];

    // Partitioning for Spatial Access
    #pragma HLS ARRAY_PARTITION variable=A_local dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=B_local dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=C_local dim=0 complete

    for (int r = 0; r < num_tiles_M; r++) {
        for (int c = 0; c < num_tiles_P; c++) {
            
            // OPTIMIZATION: Unrolled Initialization (1 Cycle)
            Init_C_Loop:
            for(int i=0; i<SA_DIM; i++) {
                #pragma HLS UNROLL
                for(int j=0; j<SA_DIM; j++) {
                    #pragma HLS UNROLL
                    C_local[i][j] = 0.0f;
                }
            }

            // Compute Tile Depth (Accumulate over K tiles)
            for (int k = 0; k < num_tiles_N; k++) {
                
                // Load A: 32 rows, 2 packets per row
                Load_A_Local:
                for (int i = 0; i < SA_DIM; i++) {
                    for (int p = 0; p < 2; p++) {
                        #pragma HLS PIPELINE II=1
                        ap_uint<512> packet = stream_A.read();
                        for (int v = 0; v < 16; ++v) {
                            #pragma HLS UNROLL
                            uint32_t raw_bits = packet.range(32*(v+1)-1, 32*v);
                            A_local[i][p*16 + v] = *reinterpret_cast<float*>(&raw_bits);
                        }
                    }
                }

                // Load B: 32 rows, 2 packets per row
                Load_B_Local:
                for (int j = 0; j < SA_DIM; j++) {
                    for (int p = 0; p < 2; p++) {
                        #pragma HLS PIPELINE II=1
                        ap_uint<512> packet = stream_B.read();
                        for (int v = 0; v < 16; ++v) {
                            #pragma HLS UNROLL
                            uint32_t raw_bits = packet.range(32*(v+1)-1, 32*v);
                            B_local[j][p*16 + v] = *reinterpret_cast<float*>(&raw_bits);
                        }
                    }
                }

                // Execute Compute on Tile
                Compute_Loop:
                for (int kk = 0; kk < SA_DIM; ++kk) {
                    #pragma HLS PIPELINE II=1
                    // OPTIMIZATION: Explicit Unrolling forces 1024 DSP utilization
                    for (int i = 0; i < SA_DIM; ++i) {
                        #pragma HLS UNROLL
                        for (int j = 0; j < SA_DIM; ++j) {
                            #pragma HLS UNROLL
                            C_local[i][j] += A_local[i][kk] * B_local[kk][j];
                        }
                    }
                }
            } 

            // Write C
            Store_C_Local:
            for (int i = 0; i < SA_DIM; i++) {
                for (int p = 0; p < 2; p++) {
                    #pragma HLS PIPELINE II=1
                    ap_uint<512> out_packet;
                    for (int v = 0; v < 16; v++) {
                        #pragma HLS UNROLL
                        float val = C_local[i][p*16 + v];
                        uint32_t raw_bits = *reinterpret_cast<uint32_t*>(&val);
                        out_packet.range(32*(v+1)-1, 32*v) = raw_bits;
                    }
                    stream_C.write(out_packet);
                }
            }
        }
    }
}

// --------------------------------------------------------
// 4. Write Engine C
// --------------------------------------------------------
void write_C_opt(ap_uint<512>* C, hls::stream<ap_uint<512>>& stream_C, int M, int N, int P) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    const int PACKETS_PER_TILE = 64; 

    for (int r = 0; r < num_tiles_M; r++) {
        for (int c = 0; c < num_tiles_P; c++) {
            // Standard Row-Major Output (r, c)
            int block_idx = r * num_tiles_P + c;
            int offset = block_idx * PACKETS_PER_TILE;
            
            Write_C_Loop:
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
    // OPTIMIZATION: Outstanding transactions to hide latency
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
    
    // OPTIMIZATION: Increased FIFO Depth to 512 to prevent stalling
    hls::stream<ap_uint<512>> StreamA("StreamA");
    #pragma HLS STREAM variable=StreamA depth=512
    hls::stream<ap_uint<512>> StreamB("StreamB");
    #pragma HLS STREAM variable=StreamB depth=512
    hls::stream<ap_uint<512>> StreamC("StreamC");
    #pragma HLS STREAM variable=StreamC depth=512

    read_A_opt(A, StreamA, M, N, P);
    read_B_opt(B, StreamB, M, N, P);
    compute_gemm_opt(StreamA, StreamB, StreamC, M, N, P);
    write_C_opt(C, StreamC, M, N, P);
}
}