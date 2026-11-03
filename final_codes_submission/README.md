# final_codes_submission — Curated reference kernels

The two implementations selected as the "final" deliverables of the exploration,
kept together as clean, standalone kernel + host pairs.

| Subfolder | Kernel | Precision | What it is |
|---|---|---|---|
| `naive_impl/` | `mmult_base.cpp` | FP32 | The naive triple-loop baseline (same as [`../fpga-matrix-mult`](../fpga-matrix-mult)) — the correctness/perf reference. |
| `matrix_mult_paper_imp/` | `mmult.cpp` | INT8 → INT32 | The weight-stationary dataflow engine (same as [`../mmult_dataflow`](../mmult_dataflow)) — A held on-chip via `update_A`, B streamed in blocks, 32×32 DSP-forced compute. |

These are the "before and after" of the project: the obvious implementation, and
the optimized dataflow design that improves on it. Each `host.cpp` sets up the
matrices, runs the kernel, and validates against a CPU reference.

See the [root README](../README.md) for the full design catalogue, build flow,
and environment setup.
