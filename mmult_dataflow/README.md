# mmult_dataflow — Weight-stationary INT8 dataflow GEMM

A paper-style GEMM engine where **matrix A stays resident on-chip** and B is
streamed in column-blocks. This is one of the two curated "final" kernels (also
copied into `../final_codes_submission/matrix_mult_paper_imp/`).

- **Kernel:** `kernel/mmult_dataflow.cpp` · top function `mmult`
- **Precision:** INT8 (`char`) × INT8 → **INT32**
- **Compute tile:** 32×32, `B` column block `BLOCK_M = 256`

## Architecture

Signature: `mmult(a, b, c, N, K, M, update_A)`.

- **`update_A` flag / weight-stationary A.** When `update_A` is true, A is loaded
  once into a persistent on-chip buffer `bram_a[MAX_N][MAX_K]` (partitioned on
  dim 1 for high-bandwidth reads). Subsequent calls with the same A skip the
  reload — useful when one weight matrix multiplies many activation batches.
- **Streaming B in blocks.** `mmult_dataflow()` is a `#pragma HLS DATAFLOW`
  region: it loads a `K × BLOCK_M` block of B into `bram_b`, then walks 32×32
  tiles calling `compute_tile()`.
- **DSP-forced compute.** `compute_tile()` is a 32×32 unrolled MAC array with
  `#pragma HLS BIND_OP op=mul impl=dsp` forcing multiplies onto DSP48 slices.

## Build & run

See the [root README](../README.md). Note the extra `update_A` kernel argument in
the host call.

```bash
make kernel TARGET=hw && make host
```
