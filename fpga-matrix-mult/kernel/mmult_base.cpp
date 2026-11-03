extern "C" {
void mmult_base(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int P)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=M bundle=control
#pragma HLS INTERFACE s_axilite port=N bundle=control
#pragma HLS INTERFACE s_axilite port=P bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    Row: for (int i = 0; i < M; i++) {
        Col: for (int j = 0; j < P; j++) {
            float sum = 0.0f;
            Prod: for (int k = 0; k < N; k++) {
	    #pragma HLS PIPELINE II=1 
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}
}
