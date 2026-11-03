#!/usr/bin/env bash
# run_bench.sh — Measure on-board GEMM throughput for one design.
#
# Runs a design's host binary across a sweep of square matrix sizes, parses the
# kernel execution time the host prints, computes GFLOP/s, records PASS/FAIL, and
# appends the results to results/<design>.csv (and a Markdown table).
#
# This produces REAL measured numbers (unlike the HLS estimates in RESULTS.md),
# so it must be run on a host with the Alveo U50 and XRT available.
#
# Usage:
#   scripts/run_bench.sh <design_dir> [-x xclbin] [-d device] [-s "sizes..."]
#
# Examples:
#   scripts/run_bench.sh systolic_32x32
#   scripts/run_bench.sh systolic_32x32 -d 1 -s "256 512 1024"
#   scripts/run_bench.sh Tile_32_Mat_Mult -x Tile_32_Mat_Mult/kernel/build/mmult.xclbin
#
# The host is expected to accept: -x <xclbin> -d <dev> -m <M> -n <N> -p <P>
# and to print a line containing "execution time:" followed by microseconds.

set -euo pipefail

DESIGN="${1:?usage: run_bench.sh <design_dir> [-x xclbin] [-d device] [-s sizes]}"
shift || true

DEVICE=1
SIZES="128 256 512 1024"
XCLBIN=""

while getopts "x:d:s:" opt; do
  case "$opt" in
    x) XCLBIN="$OPTARG" ;;
    d) DEVICE="$OPTARG" ;;
    s) SIZES="$OPTARG" ;;
    *) echo "unknown option"; exit 1 ;;
  esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

HOST_BIN="$DESIGN/host/build/host"
[ -x "$HOST_BIN" ] || HOST_BIN="$DESIGN/host/hw"
if [ ! -x "$HOST_BIN" ]; then
  echo "ERROR: host binary not found (looked for $DESIGN/host/build/host and $DESIGN/host/hw)."
  echo "Build it first:  make -C $DESIGN host"
  exit 1
fi

# Auto-detect the xclbin if not given.
if [ -z "$XCLBIN" ]; then
  XCLBIN="$(ls "$DESIGN"/kernel/build/*.xclbin 2>/dev/null | head -n1 || true)"
fi
if [ -z "$XCLBIN" ] || [ ! -f "$XCLBIN" ]; then
  echo "ERROR: no xclbin found under $DESIGN/kernel/build/. Pass one with -x."
  exit 1
fi

mkdir -p results
CSV="results/$(basename "$DESIGN").csv"
MD="results/$(basename "$DESIGN").md"
echo "design,M,N,P,kernel_us,gflops,status" > "$CSV"
{
  echo "### $(basename "$DESIGN") — measured on-board throughput"
  echo ""
  echo "xclbin: \`$XCLBIN\` · device: $DEVICE · $(date -u +%Y-%m-%dT%H:%MZ)"
  echo ""
  echo "| M×N×P | Kernel time | GFLOP/s | Status |"
  echo "|---|---|---|---|"
} > "$MD"

echo "Benchmarking $(basename "$DESIGN")  (xclbin: $XCLBIN, device: $DEVICE)"
for S in $SIZES; do
  M=$S; N=$S; P=$S
  OUT="$("$HOST_BIN" -x "$XCLBIN" -d "$DEVICE" -m "$M" -n "$N" -p "$P" 2>&1 || true)"

  # Microseconds from a line like "FPGA execution time: 1234 us"
  US="$(echo "$OUT" | grep -ioE 'execution time:[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -n1 || true)"
  if echo "$OUT" | grep -qiE 'TEST PASSED|PASSED'; then STATUS=PASS
  elif echo "$OUT" | grep -qiE 'TEST FAILED|FAILED|Mismatch'; then STATUS=FAIL
  else STATUS="?"; fi

  if [ -n "${US:-}" ] && [ "$US" -gt 0 ]; then
    # GFLOP/s = 2*M*N*P / (time_seconds) / 1e9 = 2*M*N*P / (us*1e3)
    GF="$(awk -v m="$M" -v n="$N" -v p="$P" -v us="$US" 'BEGIN{printf "%.2f", (2.0*m*n*p)/(us*1e3)}')"
    echo "  ${M}x${N}x${P}:  ${US} us   ${GF} GFLOP/s   ${STATUS}"
    echo "$(basename "$DESIGN"),$M,$N,$P,$US,$GF,$STATUS" >> "$CSV"
    echo "| ${M}×${N}×${P} | ${US} µs | ${GF} | ${STATUS} |" >> "$MD"
  else
    echo "  ${M}x${N}x${P}:  no timing parsed (${STATUS})"
    echo "$(basename "$DESIGN"),$M,$N,$P,,,${STATUS}" >> "$CSV"
    echo "| ${M}×${N}×${P} | — | — | ${STATUS} |" >> "$MD"
  fi
done

echo "Wrote $CSV and $MD"
