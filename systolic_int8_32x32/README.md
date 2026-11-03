# systolic_int8_32x32 — Quantized INT8 dataflow systolic GEMM

The streaming systolic engine on an **INT8 datapath**, packing far more operands
per memory beat than the FP32 variants.

- **Kernel:** `kernel/mmult_systolic.cpp` · top function `mmult_systolic`
- **Precision:** INT8 (`ap_int<8>`) × INT8 → **INT32** (`ap_int<32>`)
- **Array / tile:** `SA_DIM = 32`

## Architecture

Four-stage `#pragma HLS DATAFLOW` pipeline (`read_A / read_B / compute_gemm /
write_C`) over 512-bit streams, like `systolic_32x32_dataflow`, but:

- **Dense packing.** A 512-bit packet carries **64 INT8 values**, i.e. **two full
  32-element rows per beat** — 16 packets move an entire 32×32 INT8 tile (vs. 64
  packets for an FP32 tile).
- **Tiled memory layout.** A is tiled `[Tile_M][Tile_N]`, B is transposed and
  tiled `[Tile_P][Tile_N]`, and C is written back as INT32 tiles. The host
  performs the matching **pack/unpack** (`pack_A`, `pack_B`, `unpack_C` in
  `host/host_aryan.cpp`).
- **Integer MACs.** The 32×32 compute grid accumulates `ap_int<32>` to hold the
  full-precision dot product without overflow.

INT8 roughly quadruples the effective memory bandwidth and lets more MACs share
each DSP, at the cost of quantization — the throughput-oriented end of the design
space.

## Build & run

See the [root README](../README.md). The INT8 host validates against an exact
integer golden reference (no epsilon).

```bash
make kernel TARGET=hw && make host
```
