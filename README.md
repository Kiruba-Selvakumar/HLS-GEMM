# HLS GEMM Accelerators on AMD/Xilinx FPGA

A collection of **General Matrix Multiply (GEMM / `C = A × B`) kernels** written in
**C/C++ for AMD Vitis HLS** and run on an **Alveo U50** FPGA via the **XRT** runtime.

The repository documents an incremental design-space exploration: starting from a
naive triple-loop kernel and progressively adding **tiling**, **wide (512-bit)
memory interfaces**, **true systolic processing-element (PE) arrays**,
**dataflow streaming**, **double buffering**, and **INT8 quantization**. Each
top-level directory is a self-contained variant of the same kernel with its own
host program, build files, and (for most) HLS simulation configs.

> Toolchain: **Vitis / Vitis HLS 2025.1**, **XRT** · Target board: **Alveo U50**
> (`xilinx_u50_gen3x16_xdma_5_202210_1`) · Data path: **HBM**.

---

## Table of Contents

- [Why so many folders?](#why-so-many-folders)
- [Design catalogue](#design-catalogue)
- [Architectural concepts](#architectural-concepts)
- [Repository layout (per design)](#repository-layout-per-design)
- [Building and running](#building-and-running)
- [HLS simulation (csim / cosim)](#hls-simulation-csim--cosim)
- [Environment setup](#environment-setup)
- [Results](#results)
- [Notes on repository hygiene](#notes-on-repository-hygiene)
- [Improvement ideas](#improvement-ideas)

---

## Why so many folders?

Each folder is one point in the design-space exploration. They all compute the
same thing — a dense matrix product — but trade off **DSP usage**, **on-chip
memory (BRAM/URAM)**, **memory bandwidth**, **numerical precision**, and
**routing/timing closure** differently. The naming encodes the two main knobs:

- **Tile / array dimension** — `16`, `32`, `48` (the size of the on-chip
  compute block / systolic array edge).
- **Micro-architecture** — `Tile` (blocked MAC), `systolic` (true PE array with
  shift registers), `dataflow` (streaming producer/consumer), `double_buffer`
  (ping-pong overlap), `int8` (quantized).

---

## Design catalogue

| Directory | Kernel top | Precision | Compute style | Array / Tile | Key idea |
|---|---|---|---|---|---|
| `fpga-matrix-mult/` | `mmult_base` | FP32 | Naive triple loop, pipelined inner product | — | Baseline / reference correctness |
| `Tile_16_Mat_Mult_FP32/` | `mmult` | FP32 | Blocked (tiled) MAC | 16×16 | First on-chip tiling; conservative FP32 resource use |
| `Tile_32_Mat_Mult/` | `mmult` | INT16 → INT32 | Blocked (tiled) MAC, fully partitioned tiles | 32×32 | Larger tile, integer path, `Mul_DSP` forcing |
| `accelerated-matrix-mul/` | `mmult` | FP32 | Tiled with `#pragma HLS DATAFLOW` + `hls::stream` load/compute overlap | 32×32 | Overlap DMA load with compute inside a tile |
| `mmult_dataflow/` | `mmult` | INT8 → INT32 | Paper-style dataflow, persistent on-chip A, blocked B streaming | 32×32 | "Weight-stationary" A held in BRAM; `update_A` flag |
| `systolic_32x32/` | `mmult_systolic` | FP32 | **True systolic array** (2D shift registers), 512-bit ports, URAM for A | 32×32 | Wide-memory + real PE array; transposed B |
| `systolic_48x48/` | `mmult_systolic` | FP32 | Systolic array with **hoisted A-strip load** + URAM | 32×32 (target 48, reduced) | Attempt at 48×48; scaled back to 32 to fix routing congestion |
| `systolic_32x32_dataflow/` | `mmult_systolic` | FP32 | Systolic compute inside a 4-stage `DATAFLOW` (read A / read B / compute / write C) | 32×32 | Tiled streaming GEMM with per-tile MAC unroll |
| `sys_32x32_dataflow_double_buffe/` | `mmult_systolic` | FP32 | Dataflow + **ping-pong double buffering** of tiles | 16×16 | Prologue/epilogue software pipelining to hide load latency |
| `systolic_int8_32x32/` | `mmult_systolic` | INT8 → INT32 | Dataflow systolic, tiled memory layout, 2 rows/512-bit packet | 32×32 | Quantized throughput variant |
| `folder_template/` | `mmult_systolic` | FP32 | Clean scaffold (same as `systolic_32x32`) | 32×32 | Starting point for new variants |
| `final_codes_submission/` | `mmult_base`, `mmult` | FP32 / INT8 | Curated copies of the naive baseline + the paper dataflow kernel | — | The two "final" reference implementations |

### One-line summary of each variant

- **`fpga-matrix-mult` (baseline).** Textbook `for i / for j / for k` with the
  inner reduction pipelined at `II=1`. Correct but memory-bound and DSP-light —
  the reference every other design is measured against.

- **`Tile_16_Mat_Mult_FP32`.** Introduces classic **blocked matrix multiply**:
  16×16 tiles of A, B, C are staged in partitioned BRAM, the reduction loop is
  fully unrolled (16 FP32 MACs/cycle), and the tile is written back. Tile size is
  deliberately kept at 16 to fit FP32 DSP cost.

- **`Tile_32_Mat_Mult`.** Same tiling idea at 32×32 but on an **INT16×INT16 →
  INT32** datapath with fully partitioned tile buffers and explicit
  `#pragma HLS RESOURCE ... core=Mul_DSP` to force DSP mapping.

- **`accelerated-matrix-mul`.** Keeps the 32×32 tiling but wraps the per-tile
  **load and compute in a `DATAFLOW` region** connected by `hls::stream` FIFOs,
  so loading the next A/B tile overlaps with computing the current one.

- **`mmult_dataflow` / `final_codes_submission/matrix_mult_paper_imp`.** A
  paper-style **INT8** engine. Matrix A is loaded **once** into a persistent
  on-chip buffer (`bram_a`, gated by an `update_A` argument — weight-stationary
  behavior), B is streamed in column-blocks, and a 32×32 unrolled `compute_tile`
  (with `BIND_OP op=mul impl=dsp`) accumulates into INT32.

- **`systolic_32x32`.** The first **genuine systolic array**: 2D `val_A` / `val_B`
  shift-register meshes feed a 32×32 grid of MACs. Uses **512-bit (`ap_uint<512>`)
  AXI ports** loading 16 floats/cycle, assumes **B is transposed on the host**,
  stores A in **URAM** (`BIND_STORAGE ... impl=uram`) and reshapes buffers to
  avoid BRAM/URAM blow-up.

- **`systolic_48x48`.** Same systolic style, targeting a 48×48 array for more MACs,
  but the on-chip comment records that it was **reduced back to 32×32 to escape
  "Congestion Level 7" routing failures**. Adds a **hoisted A-strip load** (load
  a full row-strip of A once, reuse across all column tiles).

- **`systolic_32x32_dataflow`.** Restructures the systolic kernel into four
  concurrent dataflow stages — `read_A`, `read_B`, `compute_gemm`, `write_C` —
  communicating over deep `hls::stream`s, with tiled memory addressing and a
  32-deep MAC unroll per tile.

- **`sys_32x32_dataflow_double_buffe`.** Adds **ping-pong (double) buffering**:
  a prologue loads tile 0, then each iteration computes tile *k* while
  loading tile *k+1* into the alternate buffer (`A_ping_pong[2][…]`), followed by
  an epilogue for the last tile — classic software pipelining to hide load latency.

- **`systolic_int8_32x32`.** The dataflow systolic engine on an **INT8×INT8 →
  INT32** datapath, packing **two 32-element rows per 512-bit packet** and using a
  tiled memory layout that the host must pack/unpack (`pack_A`, `pack_B`,
  `unpack_C`).

---

## Architectural concepts

Several techniques recur across the variants:

- **Tiling / blocking.** The output matrix is partitioned into `TILE_SIZE ×
  TILE_SIZE` blocks; A/B sub-tiles are staged on-chip so the inner MACs read fast
  local memory instead of DDR/HBM.

- **Array partitioning & reshaping.** `#pragma HLS ARRAY_PARTITION` (and
  `ARRAY_RESHAPE`) expose enough memory ports to feed a fully-unrolled MAC grid in
  a single cycle; `BIND_STORAGE ... impl=uram` moves large A buffers to URAM.

- **Wide memory interfaces.** The systolic variants use `ap_uint<512>` AXI master
  ports to move **16 FP32** (or **64 INT8**) elements per beat, with
  `max_read_burst_length`, `num_read_outstanding`, etc., tuned for HBM bandwidth.

- **Systolic PE array.** `val_A`/`val_B` 2D shift registers stream operands across
  a grid of accumulators; each cycle every PE does one MAC, giving `SA_DIM²`
  MACs/cycle at `II=1`.

- **Dataflow streaming.** `#pragma HLS DATAFLOW` + `hls::stream` split the kernel
  into concurrent read/compute/write tasks so memory movement and computation
  overlap.

- **Double buffering.** Ping-pong tile buffers overlap the load of the next tile
  with the compute of the current one.

- **DSP forcing.** `BIND_OP op=mul impl=dsp` / `RESOURCE core=Mul_DSP` push
  multiplies onto DSP48 slices rather than LUT fabric.

- **Precision trade-offs.** FP32 for accuracy; INT16/INT8 with INT32 accumulation
  to pack more MACs per DSP and per memory beat.

---

## Repository layout (per design)

Most design folders follow the same structure:

```text
<design>/
├── README.md            # Quick start (some are template stubs — see notes below)
├── Makefile             # Orchestrates kernel + host build, and csim/cosim
├── fpga_setup.md        # IIIT-H node / Vitis environment guide
├── profile_config.cfg   # v++ profiling config (data/memory/stall/HBM trace)
├── xrt.ini              # XRT runtime + trace settings
├── host/
│   ├── host.cpp         # XRT Native C++ host: alloc BOs, sync, run, verify vs CPU
│   ├── cmdlineparser.*  # CLI parsing helpers
│   └── logger.*         # Logging helpers
├── kernel/
│   ├── mmult*.cpp       # The HLS kernel (the interesting file)
│   ├── Makefile         # v++ compile (.xo) + link (.xclbin)
│   └── CMakeLists.txt
├── hls/
│   └── hls_test.cpp     # C/RTL testbench for csim/cosim
├── scripts/             # run.sh helpers
└── docs/STRUCTURE.md    # Folder description
```

The **host program** (XRT Native C++ API) is the same shape everywhere: open the
device, load the `.xclbin`, allocate buffer objects, initialize A/B with a
deterministic pattern, compute a **CPU golden reference**, run the kernel, sync
results back, and **validate** (FP variants use an epsilon tolerance; INT
variants check exact equality). The INT8/tiled-layout hosts additionally
**pack/unpack** matrices into the tiled on-chip layout the kernel expects, and
the systolic FP32 hosts **transpose B**.

---

## Building and running

Each design builds independently. From inside a design directory:

```bash
# 0) Environment (see below)
source /opt/xilinx/xrt/setup.sh
source /home/Xilinx/2025.1/Vitis/settings64.sh
export PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1
export TARGET=hw          # or hw_emu for emulation

# 1) Build the kernel  ->  kernel/build/<kernel>.xclbin
make kernel TARGET=$TARGET

# 2) Build the host
make host

# 3) Run
./host/build/host -x kernel/build/<kernel>.xclbin -d 1 -m 256 -n 256 -p 256
```

Host command-line flags (systolic host):

| Flag | Meaning | Default |
|---|---|---|
| `-x, --xclbin_file` | path to the built `.xclbin` (required) | — |
| `-d, --device_id`   | XRT device index | `1` |
| `-m, --M`           | rows of A | `256` |
| `-n, --N`           | cols of A / rows of B | `256` |
| `-p, --P`           | cols of B | `256` |

You can also drive `v++` directly through the per-directory kernel Makefile:

```bash
make -C kernel xclbin TARGET=hw PLATFORM=$PLATFORM
```

---

## HLS simulation (csim / cosim)

`sw_emu` is deprecated in this toolchain; correctness is checked with **HLS
C-simulation** and **C/RTL co-simulation** using the testbench in `hls/`:

```bash
# Fast C-level simulation
make csim  HLS_CFG=./hls/csim.cfg  WORK_DIR=./hls_csim

# RTL co-simulation against the same testbench
make cosim HLS_CFG=./hls/cosim.cfg WORK_DIR=./hls_cosim
```

Under the hood these call
`vitis-run --mode hls --csim|--cosim --config <cfg> --work_dir <dir>`.

---

## Environment setup

Targets the **AMD/Xilinx Alveo U50** on the IIIT-H FPGA nodes. See any design's
[`fpga_setup.md`](systolic_32x32/fpga_setup.md) for the full IIIT-H VPN / RCE /
license-server instructions. The essentials:

```bash
export LM_LICENSE_FILE=2100@rce.iiit.ac.in
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export PLATFORM_REPO_PATHS=/opt/xilinx/platform
source /opt/xilinx/xrt/setup.sh
source /home/Xilinx/2025.1/Vitis/settings64.sh
export PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1
```

Verify the board is visible with `xbutil examine`.

---

## Results

Per-design **resource usage, clock, and latency** (extracted from the Vitis HLS
2025.1 synthesis reports for the U50) and the workflow for collecting **on-board
throughput** live in **[RESULTS.md](RESULTS.md)**.

Quick regeneration:

```bash
# Synthesis-estimate table, straight from the build reports:
python3 scripts/collect_hls_results.py

# Measured on-board GFLOP/s for a design (needs the U50 + XRT):
scripts/run_bench.sh systolic_32x32 -d 1 -s "128 256 512 1024"
```

Headline synthesis numbers (Alveo U50, 300 MHz target): the 32×32 fully-unrolled
arrays use ~1,000 DSPs (~17%); the FP32 systolic array is the most LUT- and
timing-intensive (~29% LUT, 366 MHz); `systolic_48x48` is the only URAM user
(and was scaled back to 32×32 to close routing). See RESULTS.md for the full
table and important caveats on comparing latency across designs.

---

## Notes on repository hygiene

This repo currently also contains **large generated build artifacts** committed to
git — synthesis/implementation logs, `_x/` build trees, `.run/` emulation
sandboxes, `*.xclbin` bitstreams (~33 MB each), and IP `*.zip` files. There is
also a **777 MB `hls_example.zip`** in the working tree (untracked). Before
publishing to GitHub you almost certainly want to strip these, since GitHub
rejects files over 100 MB and the bitstreams/logs are not source.

Suggested `.gitignore` additions:

```gitignore
# Vitis / HLS build outputs
**/_x/
**/build/
**/.run/
**/.Xil/
**/.ipcache/
**/*.xclbin
**/*.xo
**/*.log
**/*.zip
**/xsim.dir/
```

The kernel `.cpp`, host `.cpp`, Makefiles, `CMakeLists.txt`, `*.cfg`, and the
`docs/` are the parts worth keeping under version control.

Each design folder now has its own accurate `README.md` describing that specific
variant; this top-level README is the map to all of them.

---

## Improvement ideas

See **[IMPROVEMENTS.md](IMPROVEMENTS.md)** for a prioritized roadmap — measured
results tables, arbitrary-size support, shared host code, multi-CU scaling, the
48×48 congestion investigation, datatype exploration, and CI via HLS csim.
```

---

## License

Add a license of your choice (e.g. MIT) before publishing.
