# sys_32x32_dataflow_double_buffe — Dataflow GEMM with double buffering (FP32)

Extends the streaming systolic design with **ping-pong (double) buffering** to
hide tile-load latency behind compute — classic software pipelining in hardware.

- **Kernel:** `kernel/mmult_systolic.cpp` · top function `mmult_systolic`
- **Precision:** FP32
- **Array / tile:** `SA_DIM = 16`

## Architecture

Same four-stage `read_A / read_B / compute / write_C` dataflow skeleton as
`systolic_32x32_dataflow`, but the compute engine keeps **two tile buffers**
(`A_ping_pong[2][…]`, `B_ping_pong[2][…]`) and overlaps loading with computing:

```
prologue : load tile 0
loop k=0..K-2 :            # HLS DATAFLOW
    compute(tile k)   ||   load(tile k+1)   # into the alternate buffer
epilogue : compute last tile
```

While tile *k* is being multiplied, tile *k+1* is already being fetched into the
other buffer, so the MAC array rarely waits on memory. The tile dimension is
kept at 16 here to keep the doubled buffers within on-chip memory budget.

## Build & run

See the [root README](../README.md).

```bash
make kernel TARGET=hw && make host
./host/build/host -x kernel/build/mmult_systolic.xclbin -d 1 -m 256 -n 256 -p 256
```
