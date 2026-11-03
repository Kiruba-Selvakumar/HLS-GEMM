# accelerated-matrix-mul — Tiled GEMM with load/compute overlap (FP32)

Adds **dataflow overlap** to the 32×32 tiled design: DMA loads of the next tile
run concurrently with the compute of the current one.

- **Kernel:** `kernel/mmult.cpp` · top function `mmult`
- **Precision:** FP32
- **Tile size:** 32×32

## Architecture

Inside each output-tile iteration, a `#pragma HLS DATAFLOW` region connects three
concurrent tasks through `hls::stream` FIFOs (`A_str`, `B_str`):

1. **LoadA_stream** — streams all A sub-tiles for the current row block.
2. **LoadB_stream** — streams all B sub-tiles for the current column block.
3. **Compute_tile** — drains the streams into local buffers and accumulates
   `C_local` with a fully unrolled inner product.

Because the loader and compute tasks run as a dataflow pipeline, memory latency is
hidden behind computation rather than serialized before it. The finished 32×32
`C_local` tile is written back after the dataflow region closes.

## Build & run

See the [root README](../README.md).

```bash
make kernel TARGET=hw && make host
./host/build/host -x kernel/build/mmult.xclbin -d 1 -m 256 -n 256 -p 256
```
