# Top-level Makefile for the Tiled Matrix Multiplication project
#
# Usage:
#   make all           # Builds both kernel and host
#   make kernel        # Builds mmult.xclbin
#   make host          # Builds host binary
#   make run           # Runs the application after building
#
# Environment:
#   PLATFORM=<your .xpfm name> (required for kernel)
#   TARGET=hw|hw_emu|sw_emu   (optional, default is hw)

TARGET ?= hw_emu
PLATFORM?=/opt/xilinx/platforms/xilinx_u50_gen3x16_xdma_5_202210_1/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm

.PHONY: all kernel host run clean

all: kernel host

kernel:
	$(MAKE) -C kernel all TARGET=$(TARGET)

host:
	$(MAKE) -C host DEBUG=1 all

run: all
	./scripts/run.sh

clean:
	$(MAKE) -C kernel clean || true
	$(MAKE) -C host clean || true

.PHONY: csim cosim
csim:
	$(MAKE) -C kernel csim HLS_CFG=$(HLS_CFG) WORK_DIR=$(WORK_DIR)

cosim:
	$(MAKE) -C kernel cosim HLS_CFG=$(HLS_CFG) WORK_DIR=$(WORK_DIR)
