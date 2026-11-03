#!/usr/bin/env bash
set -euo pipefail

# Default path to the hardware binary. Can be overridden from the command line.
XCLBIN_PATH=${1:-../kernel/build/mmult.xclbin}

if [[ ! -f "${XCLBIN_PATH}" ]]; then
  echo "Error: XCLBIN not found at ${XCLBIN_PATH}"
  echo "Usage: $0 [path-to-xclbin]"
  exit 1
fi

if [[ -z "${XILINX_XRT:-}" ]]; then
  echo "Warning: XILINX_XRT not set. Sourcing 'env.sh' is recommended."
fi

echo "[INFO] Running host with xclbin=${XCLBIN_PATH}"
# The host executable is called with the binary path and a device ID
XCL_EMULATION_MODE=hw_emu ../host/build/host.exe -x "${XCLBIN_PATH}" -d 1
