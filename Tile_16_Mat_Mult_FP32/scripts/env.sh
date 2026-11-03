#!/usr/bin/env bash
# Convenience script: source XRT and Vitis if needed (adjust to your site)
# For IIIT-H U50 default (Vitis 2025.1):
if [ -f /opt/xilinx/xrt/setup.sh ]; then
  source /opt/xilinx/xrt/setup.sh
fi
if [ -f /home/Xilinx/2025.1/Vitis/settings64.sh ]; then
  source /home/Xilinx/2025.1/Vitis/settings64.sh
fi
# Example platform (adjust if needed):
export PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1
echo "XRT, Vitis, and PLATFORM set (if available)."
