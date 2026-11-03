# Repository Structure

```text
fpga-vector-add/
├── README.md                 # Quick start and build/run steps
├── fpga_setup.md             # IIIT-H-specific environment and platform instructions
├── host/
│   ├── host.cpp              # Host application using XRT C++ API
│   └── CMakeLists.txt        # Build host binary
├── kernel/
│   ├── vadd.cpp              # HLS C kernel (vector addition)
│   └── CMakeLists.txt        # Build XO and link XCLBIN via v++
├── scripts/
│   ├── run.sh                # Helper to run the host with a given xclbin and N
│   └── env.sh                # (Optional) convenience environment sourcing
├── data/                     # Sample inputs/outputs if needed
├── docs/
│   └── STRUCTURE.md          # This file
└── build/                    # Out-of-source build outputs (git-ignored)
    └── (generated)
```

## Notes
- `host/` and `kernel/` are decoupled — you can build them independently.
- The kernel build uses `v++` under the hood (invoked via CMake custom commands).
- Set `$PLATFORM` to your `.xpfm` before building the kernel.
- `scripts/run.sh` expects: `./scripts/run.sh <path-to-xclbin> [N]`.

## Makefile Support
- Top-level `Makefile` orchestrates kernel and host builds and provides a `run` target.
- `host/Makefile` builds the host binary using `g++` and XRT libraries.
- `kernel/Makefile` builds the kernel with `v++` (uses `PLATFORM` and `TARGET`).

### Examples
```bash
# Build both with Make
make all

# Build only kernel
make -C kernel xclbin

# Build only host
make -C host

# Run (after build)
make run N=2048
```

---

## HLS Simulation (csim / cosim)
- Use `make csim HLS_CFG=./hls/csim.cfg WORK_DIR=./hls_csim` for C-simulation.
- Use `make cosim HLS_CFG=./hls/cosim.cfg WORK_DIR=./hls_cosim` for C/RTL Co-simulation.
- Config files live under `hls/` (sample `csim.cfg` and `cosim.cfg` provided). Edit `syn.file`, `syn.top`, and `tb.file` as needed for your project.
