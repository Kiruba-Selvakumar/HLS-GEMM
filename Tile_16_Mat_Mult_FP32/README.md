# Tile_16_Mat_Mult_FP32 — Blocked GEMM, 16×16 tiles (FP32)

First step beyond the naive baseline: **blocked (tiled) matrix multiply** with
on-chip data reuse.

- **Kernel:** `kernel/mmult.cpp` · top function `mmult`
- **Precision:** FP32
- **Tile size:** 16×16 (kept small to fit FP32 DSP/BRAM cost)

## Architecture

The output matrix is walked in 16×16 blocks. For each output tile the kernel:

1. Zeroes a local `C_local[16][16]` accumulator.
2. Sweeps the K dimension in 16-wide steps, staging `A_local` and `B_local` tiles
   into partitioned BRAM.
3. Computes the tile with the reduction loop **fully unrolled** (16 FP32 MACs per
   cycle), pipelined at `II=1`.
4. Writes the finished tile back to global memory.

Array partitioning (`A_local` on dim 2, `B_local` on dim 1, `C_local` cyclic ×4)
exposes the memory ports needed to feed the unrolled MACs. Compared to the
baseline, each loaded tile is reused across the whole block, cutting global-memory
traffic substantially.

## Build & run

See the [root README](../README.md).

```bash
make kernel TARGET=hw && make host
./host/build/host -x kernel/build/mmult.xclbin -d 1 -m 256 -n 256 -p 256
```
