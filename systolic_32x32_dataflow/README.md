# systolic_32x32_dataflow — Streaming systolic GEMM (FP32)

Restructures the systolic kernel into a **four-stage dataflow pipeline** so that
memory reads, compute, and writeback all run concurrently.

- **Kernel:** `kernel/mmult_systolic.cpp` · top function `mmult_systolic`
- **Precision:** FP32
- **Array / tile:** `SA_DIM = 32`

## Architecture

The top level is a `#pragma HLS DATAFLOW` region of four functions connected by
deep `hls::stream<ap_uint<512>>` FIFOs:

```
read_A_opt  ─┐
             ├─► compute_gemm_opt ─► write_C_opt
read_B_opt  ─┘
```

- **`read_A_opt` / `read_B_opt`** burst 512-bit packets of each 32×32 tile from
  global memory into the streams, using a **tiled memory layout** (A indexed
  `[Tile_M][Tile_N]`, B row-major `[Tile_N][Tile_P]`).
- **`compute_gemm_opt`** unpacks tiles into partitioned local buffers and runs a
  32-deep MAC loop (`C_local[i][j] += A_local[i][kk] * B_local[kk][j]`), fully
  unrolled across the 32×32 grid.
- **`write_C_opt`** streams finished tiles back to DDR/HBM.

Splitting read/compute/write into separate dataflow stages keeps the compute
engine fed continuously instead of stalling on memory.

## Build & run

See the [root README](../README.md).

```bash
make kernel TARGET=hw && make host
./host/build/host -x kernel/build/mmult_systolic.xclbin -d 1 -m 256 -n 256 -p 256
```
