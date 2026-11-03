# fpga-matrix-mult — Naive GEMM baseline (FP32)

The **reference implementation**. A textbook triple-loop matrix multiply used as
the correctness and performance baseline for every other design in this repo.

- **Kernel:** `kernel/mmult_base.cpp` · top function `mmult_base`
- **Precision:** FP32
- **Interfaces:** three `float*` AXI master ports (`gmem0/1/2`) + `s_axilite` control

## Architecture

```
for i in 0..M            # rows of C
  for j in 0..P          # cols of C
    sum = 0
    for k in 0..N        # inner product, PIPELINE II=1
      sum += A[i*N+k] * B[k*P+j]
    C[i*P+j] = sum
```

Only the inner reduction is pipelined. There is **no on-chip tiling or reuse**, so
every element of B is re-read from global memory for each row of A — the design is
memory-bound and uses few DSPs. It exists to prove correctness and to quantify how
much the tiled/systolic/dataflow variants improve over the obvious approach.

## Build & run

See the [root README](../README.md). In short:

```bash
export PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1
make kernel TARGET=hw
make host
./host/build/host -x kernel/build/mmult_base.xclbin -d 1 -m 256 -n 256 -p 256
```
