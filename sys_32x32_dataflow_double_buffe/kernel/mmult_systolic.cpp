#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>

// --- Architecture Configuration ---
#define SA_DIM 16 
#define VEC_WIDTH 16 // 512 bits / 32-bit float = 16 floats per packet
#define TILE_ITER 32 // 1024 / 32 = 32 tiles (For Tripcount)

extern "C" {

// --------------------------------------------------------
// Helper: Load Tile from Stream to Local Buffer
// --------------------------------------------------------
void load_tile_from_stream(
    hls::stream<ap_uint<512>>& stream_A, 
    hls::stream<ap_uint<512>>& stream_B,
    float A_buffer[SA_DIM][SA_DIM],
    float B_buffer[SA_DIM][SA_DIM]
) {
    #pragma HLS INLINE off // Force hierarchy for Dataflow
    
    // 32x32 floats = 1024 floats. 
    // 1024 / 16 floats per packet = 64 packets.
    // We load A and B sequentially here, but this function runs 
    // in parallel with the compute engine.
    
    Load_A_Internal:
    for (int i = 0; i < SA_DIM; i++) {
        for (int p = 0; p < 2; p++) { // 2 packets per row
            #pragma HLS PIPELINE II=1
            ap_uint<512> packet = stream_A.read();
            for (int v = 0; v < 16; ++v) {
                #pragma HLS UNROLL
                uint32_t raw_bits = packet.range(32*(v+1)-1, 32*v);
                A_buffer[i][p*16 + v] = *reinterpret_cast<float*>(&raw_bits);
            }
        }
    }

    Load_B_Internal:
    for (int j = 0; j < SA_DIM; j++) {
        for (int p = 0; p < 2; p++) {
            #pragma HLS PIPELINE II=1
            ap_uint<512> packet = stream_B.read();
            for (int v = 0; v < 16; ++v) {
                #pragma HLS UNROLL
                uint32_t raw_bits = packet.range(32*(v+1)-1, 32*v);
                B_buffer[j][p*16 + v] = *reinterpret_cast<float*>(&raw_bits);
            }
        }
    }
}

// --------------------------------------------------------
// Helper: Execute Systolic Compute on Local Buffer
// --------------------------------------------------------
void compute_tile_systolic(
    float A_buffer[SA_DIM][SA_DIM],
    float B_buffer[SA_DIM][SA_DIM],
    float C_buffer[SA_DIM][SA_DIM]
) {
    #pragma HLS INLINE off // Force hierarchy for Dataflow

    // Iterate over the "K" dimension of the TILE (32)
    Compute_Loop:
    for (int kk = 0; kk < SA_DIM; ++kk) {
        #pragma HLS PIPELINE II=1
        
        // Full Spatial Unroll: 32x32 = 1024 MACs per cycle
        for (int i = 0; i < SA_DIM; ++i) {
            #pragma HLS UNROLL
            for (int j = 0; j < SA_DIM; ++j) {
                #pragma HLS UNROLL
                // Systolic update
                C_buffer[i][j] += A_buffer[i][kk] * B_buffer[kk][j];
            }
        }
    }
}

// --------------------------------------------------------
// 1. Read Engine A
// --------------------------------------------------------
void read_A_opt(const ap_uint<512>* A, hls::stream<ap_uint<512>>& stream_A, int M, int N, int P) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    int num_tiles_N = N / SA_DIM;
    const int PACKETS_PER_TILE = 64; 

    Loop_RA_M: for (int r = 0; r < num_tiles_M; r++) {
        #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER
        Loop_RA_P: for (int c = 0; c < num_tiles_P; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER
            Loop_RA_N: for (int k = 0; k < num_tiles_N; k++) {
                #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER
                
                int block_idx = r * num_tiles_N + k;
                int offset = block_idx * PACKETS_PER_TILE;
                
                Read_A_Inner:
                for (int i = 0; i < PACKETS_PER_TILE; i++) {
                    #pragma HLS PIPELINE II=1
                    stream_A.write(A[offset + i]);
                }
            }
        }
    }
}

// --------------------------------------------------------
// 2. Read Engine B
// --------------------------------------------------------
void read_B_opt(const ap_uint<512>* B, hls::stream<ap_uint<512>>& stream_B, int M, int N, int P) {
    int num_tiles_M = M / SA_DIM;
    int num_tiles_P = P / SA_DIM;
    int num_tiles_N = N / SA_DIM;
    const int PACKETS_PER_TILE = 64;

    Loop_RB_M: for (int r = 0; r < num_tiles_M; r++) {
        #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER
        Loop_RB_P: for (int c = 0; c < num_tiles_P; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER
            Loop_RB_N: for (int k = 0; k < num_tiles_N; k++) {
                #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER

                // Row-major logic: Row=k, Col=c
                int block_idx = k * num_tiles_P + c; 
                int offset = block_idx * PACKETS_PER_TILE;

                Read_B_Inner:
                for (int i = 0; i < PACKETS_PER_TILE; i++) {
                    #pragma HLS PIPELINE II=1
                    stream_B.write(B[offset + i]);
                }
            }
        }
    }
}

// --------------------------------------------------------
// 3. Compute Engine (Double Buffered)
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

    // --- Double Buffers (Ping-Pong) ---
    // Dimension [2] is for Ping/Pong toggling
    float A_ping_pong[2][SA_DIM][SA_DIM];
    float B_ping_pong[2][SA_DIM][SA_DIM];
    
    // Accumulator (Output)
    float C_local[SA_DIM][SA_DIM];

    // Partitioning:
    // A: Partition Columns (dim 3) so we can read a whole row at once
    #pragma HLS ARRAY_PARTITION variable=A_ping_pong dim=3 complete
    // B: Partition Rows (dim 2) so we can read a whole column at once
    #pragma HLS ARRAY_PARTITION variable=B_ping_pong dim=2 complete
    // C: Fully partitioned for parallel MACs
    #pragma HLS ARRAY_PARTITION variable=C_local dim=0 complete

    Loop_Comp_M: for (int r = 0; r < num_tiles_M; r++) {
        #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER
        Loop_Comp_P: for (int c = 0; c < num_tiles_P; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER
            
            // 1. Initialize C
            Init_C:
            for(int i=0; i<SA_DIM; i++) {
                #pragma HLS UNROLL
                for(int j=0; j<SA_DIM; j++) {
                    #pragma HLS UNROLL
                    C_local[i][j] = 0.0f;
                }
            }

            // 2. Prologue: Load the first tile (k=0) into buffer 0
            load_tile_from_stream(stream_A, stream_B, A_ping_pong[0], B_ping_pong[0]);

            // 3. Main Loop: Compute K, while Loading K+1
            // Iterate from k=0 to N-2
            Loop_Comp_N: for (int k = 0; k < num_tiles_N - 1; k++) {
                #pragma HLS LOOP_TRIPCOUNT min=31 max=31 
                
                int compute_idx = k % 2;
                int load_idx = (k + 1) % 2;

                // DATAFLOW directive enables these two functions to run in parallel
                #pragma HLS DATAFLOW
                compute_tile_systolic(A_ping_pong[compute_idx], B_ping_pong[compute_idx], C_local);
                load_tile_from_stream(stream_A, stream_B, A_ping_pong[load_idx], B_ping_pong[load_idx]);
            }

            // 4. Epilogue: Compute the final tile (k = N-1)
            // No more loading needed
            compute_tile_systolic(A_ping_pong[(num_tiles_N - 1) % 2], B_ping_pong[(num_tiles_N - 1) % 2], C_local);

            // 5. Stream out result C
            Store_C:
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

    Loop_WC_M: for (int r = 0; r < num_tiles_M; r++) {
        #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER
        Loop_WC_P: for (int c = 0; c < num_tiles_P; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=TILE_ITER max=TILE_ITER
            
            int block_idx = r * num_tiles_P + c;
            int offset = block_idx * PACKETS_PER_TILE;
            
            Write_C_Inner:
            for (int i = 0; i < PACKETS_PER_TILE; i++) {
                #pragma HLS PIPELINE II=1
                C[offset + i] = stream_C.read();
            }
        }
    }
}

// --------------------------------------------------------
// Top Level
// --------------------------------------------------------
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
    
    // Increased Depth to accommodate double buffering jitter
    hls::stream<ap_uint<512>> StreamA("StreamA");
    #pragma HLS STREAM variable=StreamA depth=1024
    hls::stream<ap_uint<512>> StreamB("StreamB");
    #pragma HLS STREAM variable=StreamB depth=1024
    hls::stream<ap_uint<512>> StreamC("StreamC");
    #pragma HLS STREAM variable=StreamC depth=512

    read_A_opt(A, StreamA, M, N, P);
    read_B_opt(B, StreamB, M, N, P);
    compute_gemm_opt(StreamA, StreamB, StreamC, M, N, P);
    write_C_opt(C, StreamC, M, N, P);
}
}
