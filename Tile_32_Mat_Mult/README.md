# Tile_32_Mat_Mult — Blocked GEMM, 32×32 tiles (INT16 → INT32)

The tiled design scaled up to **32×32 blocks** on an **integer datapath**.

- **Kernel:** `kernel/mmult.cpp` · top function `mmult`
- **Precision:** INT16 × INT16 → **INT32** accumulation (avoids overflow)
- **Tile size:** 32×32

## Architecture

Same blocked structure as `Tile_16_Mat_Mult_FP32`, but:

- Tile buffers `A_local` / `B_local` / `C_local` are **fully partitioned**
  (`dim=0`) so an entire 32×32 tile is available in registers.
- The reduction loop is fully unrolled, and multiplication is explicitly pushed
  onto DSP slices with `#pragma HLS RESOURCE variable=prod core=Mul_DSP`.
- INT16 inputs let two operands pack more densely than FP32 and map efficiently to
  DSP48 multipliers, while the INT32 accumulator prevents overflow across the
  K-dimension sum.

This variant explores the accuracy/throughput trade-off of moving from FP32 to a
fixed-point datapath at a larger tile size.

## Build & run

See the [root README](../README.md).

```bash
make kernel TARGET=hw && make host
./host/build/host -x kernel/build/mmult.xclbin -d 1 -m 256 -n 256 -p 256
```
