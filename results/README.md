# results/

Measured on-board benchmark output, one file per design, produced by
[`../scripts/run_bench.sh`](../scripts/run_bench.sh):

- `<design>.csv` — machine-readable: `design,M,N,P,kernel_us,gflops,status`
- `<design>.md`  — human-readable Markdown table

These are committed so the repository carries real measured numbers. Regenerate
them on a host with the Alveo U50 and XRT:

```bash
scripts/run_bench.sh systolic_32x32 -d 1 -s "128 256 512 1024"
```

Roll the headline figures up into [`../RESULTS.md`](../RESULTS.md) (Section 2).
The synthesis-estimate table in RESULTS.md (Section 1) is generated separately by
`scripts/collect_hls_results.py` and needs no board.
