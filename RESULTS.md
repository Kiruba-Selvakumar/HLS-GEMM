# Results

Synthesis and performance data for the GEMM designs in this repository.

There are two kinds of numbers here, kept strictly separate:

1. **HLS synthesis estimates** (below) — resource usage, achievable clock, and
   estimated latency reported by Vitis HLS during synthesis. These are produced
   automatically by the tool and are extracted here with
   [`scripts/collect_hls_results.py`](scripts/collect_hls_results.py).
2. **On-board measured throughput** — real wall-clock kernel time on the Alveo
   U50, collected with [`scripts/run_bench.sh`](scripts/run_bench.sh). This
   section is a template to be filled in on the FPGA host (see below).

---

## 1. HLS synthesis estimates

<!-- Regenerate with: python3 scripts/collect_hls_results.py -o RESULTS_table.md -->

Target device: **xcu50-fsvh2104-2-e** (Alveo U50) · Target clock: **3.33 ns
(300 MHz)** · Tool: **Vitis HLS 2025.1**

| Design | Est. clock (ns / MHz) | Latency (worst) | DSP | BRAM_18K | URAM | FF | LUT | DSP% | LUT% |
|---|---|---|---|---|---|---|---|---|---|
| `fpga-matrix-mult` (naive FP32) | 2.470 / 405 | runtime-dependent | 7 | 2 | 0 | 4,141 | 5,162 | 0 | 0 |
| `Tile_16_Mat_Mult_FP32` | 2.433 / 411 | 10.254 ms (3,076,385 cyc) | 88 | 2 | 0 | 21,886 | 12,591 | 1 | 1 |
| `Tile_32_Mat_Mult` (INT16) | 2.433 / 411 | 2.406 ms (722,001 cyc) | 1,032 | 2 | 0 | 141,047 | 77,158 | 17 | 8 |
| `accelerated-matrix-mul` | 2.433 / 411 | 10.512 ms (3,153,853 cyc) | 1,044 | 74 | 0 | 107,888 | 73,505 | 17 | 8 |
| `mmult_dataflow` (INT8) | 2.433 / 411 | 5.897 ms (1,769,297 cyc) | 168 | 2 | 0 | 161,023 | 70,711 | 2 | 8 |
| `systolic_32x32` (FP32) | 2.729 / 366 | 0.120 s (36,098,177 cyc) | 864 | 16 | 0 | 282,769 | 253,237 | 14 | 29 |
| `systolic_48x48` (FP32) | 2.729 / 366 | 0.113 ms (33,907 cyc) | 858 | 272 | 256 | 250,879 | 176,702 | 14 | 20 |
| `systolic_32x32_dataflow` | 2.470 / 405 | 32.018 ms (9,606,291 cyc) | 788 | 62 | 0 | 285,316 | 204,978 | 13 | 23 |
| `sys_32x32_dataflow_double_buffe` | 2.470 / 405 | 17.605 ms (5,281,940 cyc) | 406 | 76 | 0 | 194,612 | 146,913 | 6 | 16 |
| `systolic_int8_32x32` | 2.433 / 411 | runtime-dependent | 1,044 | 62 | 0 | 175,215 | 99,158 | 17 | 11 |

Percentages are against the **full U50 device** (5,952 DSP · 2,688 BRAM_18K ·
640 URAM · 1,743,360 FF · 871,680 LUT). Absolute resource counts are directly
comparable across designs; see the caveats below before comparing latency.

### ⚠️ How to read the latency column

The latency numbers are **not directly comparable across designs**, for two
reasons:

- **Different synthesized problem sizes.** Each kernel's latency reflects the
  `LOOP_TRIPCOUNT` / interface `depth` pragmas it was last synthesized with — some
  were built for 256×256, others for 1024×1024 or a small co-sim size. For
  example `systolic_48x48` shows a tiny 0.113 ms because it was synthesized for a
  96×96 co-sim, **not** because it is faster than `systolic_32x32`.
- **Runtime-dependent latency.** The naive baseline and the INT8 dataflow kernel
  have no fixed tripcount, so HLS reports no bounded latency ("runtime-dependent").

To compare designs fairly, use the **on-board throughput** in Section 2, where
every design runs the *same* matrix sizes.

### What the resource numbers already tell us

- **The naive baseline is nearly free but useless for throughput** — 7 DSPs, no
  reuse. It exists only as a correctness reference.
- **Full spatial unroll is what costs DSPs.** The 32×32 fully-unrolled arrays
  (`Tile_32`, `accelerated`, `systolic_int8`) all land around **1,000–1,044 DSPs**
  (~17%), i.e. roughly the 32×32 = 1,024 MACs you'd expect.
- **FP32 systolic is LUT-hungry.** `systolic_32x32` uses ~29% of device LUTs and
  the lowest clock (366 MHz) — floating-point add/mul on a large mesh stresses
  both routing and timing. This is the practical ceiling that motivated the INT8
  and dataflow variants.
- **`systolic_48x48` is the memory-footprint outlier** — it's the only design
  using **URAM (256 blocks) and heavy BRAM (272)** because it buffers a full A
  row-strip on-chip. It's also the design that had to be scaled back from a 48×48
  to a 32×32 array to escape routing congestion.
- **Double buffering trades DSPs for overlap** — `sys_32x32_dataflow_double_buffe`
  (16×16 array) uses ~406 DSPs, about half the 32×32 variants, by keeping the
  array smaller while hiding load latency behind compute.

---

## 2. On-board measured throughput (to be collected)

This is the definitive comparison. Because it requires the physical U50, it is
run on the FPGA host, not in synthesis.

### How to collect

```bash
# Build the design (kernel + host), then benchmark it across a size sweep:
make -C systolic_32x32 kernel TARGET=hw
make -C systolic_32x32 host
scripts/run_bench.sh systolic_32x32 -d 1 -s "128 256 512 1024"
```

`run_bench.sh` runs the host at each size, parses the kernel execution time it
prints, computes throughput, records PASS/FAIL, and writes `results/<design>.csv`
and `results/<design>.md`.

**Throughput definition.** A dense `M×N×P` GEMM performs `2·M·N·P` floating-point
operations (one multiply + one add per MAC), so:

```
GFLOP/s = (2 · M · N · P) / kernel_time_seconds / 1e9
```

For reference, a fully-utilized 32×32 FP32 array at 366 MHz has a theoretical
peak of `1024 MAC × 2 FLOP × 366e6 ≈ 750 GFLOP/s`; measured numbers show how
close each design's memory system and scheduling get to that ceiling.

### Results table (fill in from `results/*.csv`)

| Design | Precision | 256³ GFLOP/s | 512³ GFLOP/s | 1024³ GFLOP/s | Speedup vs. naive | Verified |
|---|---|---|---|---|---|---|
| `fpga-matrix-mult` | FP32 | _tbd_ | _tbd_ | _tbd_ | 1.0× (baseline) | _tbd_ |
| `Tile_16_Mat_Mult_FP32` | FP32 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `Tile_32_Mat_Mult` | INT16 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `accelerated-matrix-mul` | FP32 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `mmult_dataflow` | INT8 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `systolic_32x32` | FP32 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `systolic_32x32_dataflow` | FP32 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `sys_32x32_dataflow_double_buffe` | FP32 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `systolic_int8_32x32` | INT8 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

> Tip: also record the host's CPU golden-reference time (the hosts already print
> it) to report FPGA-vs-CPU speedup, and note the achieved fraction of peak.

---

## Reproducing / updating this file

```bash
# Regenerate the synthesis-estimate table (Section 1) from build reports:
python3 scripts/collect_hls_results.py

# Collect measured throughput (Section 2) for a design on the U50:
scripts/run_bench.sh <design_dir> -d <device> -s "128 256 512 1024"
```
