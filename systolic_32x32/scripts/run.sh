#!/usr/bin/env bash
set -euo pipefail

XCLBIN_PATH=${1:-kernel/build/vadd.xclbin}

if [[ ! -f "${XCLBIN_PATH}" ]]; then
  echo "Error: XCLBIN not found at ${XCLBIN_PATH}"
  echo "Usage: $0 <path-to-xclbin> -d 1"
  exit 1
fi

if [[ -z "${XILINX_XRT:-}" ]]; then
  echo "Warning: XILINX_XRT not set. Make sure XRT is sourced."
fi

if [[ -z "${PLATFORM:-}" ]]; then
  echo "Warning: PLATFORM not set. Kernel must be built against the correct .xpfm."
fi

echo "[INFO] Running host with xclbin=${XCLBIN_PATH}"
./host/build/host.exe -x "${XCLBIN_PATH}" -d 1 
