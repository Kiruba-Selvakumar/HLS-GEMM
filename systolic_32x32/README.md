# systolic_32x32 — True systolic PE array (FP32)

The first design with a **genuine systolic array**: a 32×32 grid of
processing elements fed by 2D shift-register meshes, backed by wide 512-bit
memory ports.

- **Kernel:** `kernel/mmult_systolic.cpp` · top function `mmult_systolic`
- **Precision:** FP32
- **Array:** `SA_DIM = 32` (32×32 ≈ 1024 DSP MACs)

## Architecture

- **Wide memory (`ap_uint<512>`).** A, B and C are accessed 512 bits at a time —
  **16 floats per beat** — with `latency=64` tuned for HBM.
- **Transposed B.** The host must transpose B (`Bᵀ`) so that columns of B become
  contiguous rows, enabling fast burst reads.
- **On-chip storage.** `A_local` is placed in **URAM**
  (`BIND_STORAGE ... impl=uram`) and both A/B buffers use `ARRAY_RESHAPE cyclic
  factor=16` to avoid BRAM/URAM explosion while keeping 16-wide access.
- **Systolic compute.** `val_A` / `val_B` shift-register arrays stream operands
  across the PE grid; every cycle each PE performs one MAC into `C_local`, with
  `BIND_OP ... impl=dsp` forcing DSP usage. The run loop iterates
  `N + 2·SA_DIM` cycles to fill and drain the pipeline.

## Host data preparation

`host/host.cpp` initializes A/B with a deterministic pattern, computes a CPU
golden reference, and validates FPGA output within an epsilon tolerance.
**Remember B must be transposed** to match the kernel's read layout.

## Build & run

See the [root README](../README.md).

```bash
make kernel TARGET=hw && make host
./host/build/host -x kernel/build/mmult_systolic.xclbin -d 1 -m 256 -n 256 -p 256
```
