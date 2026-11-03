# folder_template — Scaffold for a new GEMM variant

A clean, buildable starting point for adding a new design to the exploration. It
mirrors the standard layout (`host/`, `kernel/`, `hls/`, `Makefile`,
`profile_config.cfg`, `xrt.ini`) and ships with a working FP32 32×32 systolic
kernel identical in shape to `systolic_32x32`.

- **Kernel:** `kernel/mmult_systolic.cpp` · top function `mmult_systolic`
- **Precision:** FP32
- **Array:** `SA_DIM = 32`

## How to use it

1. Copy this folder to a new name describing your variant, e.g.
   `cp -r folder_template systolic_64x64`.
2. Edit `kernel/mmult_systolic.cpp` — change `SA_DIM`, the datatype, the
   memory/compute strategy, etc.
3. Update `host/host.cpp` if the data layout or kernel signature changes
   (transpose, tiling, pack/unpack, precision).
4. Build and validate with the standard flow (see the [root README](../README.md)).

Keeping a template folder makes each new point in the design space start from a
known-good baseline instead of from scratch.
