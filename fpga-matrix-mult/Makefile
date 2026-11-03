# Top-level Makefile to build kernel and host
# Usage:
#   make kernel        # builds vadd.xclbin
#   make host          # builds host binary
#   make all           # builds both
#   make run N=1024    # runs with N elements (after build)
#
# Environment:
#   PLATFORM=<your .xpfm name> (required for kernel)
#   TARGET=hw|hw_emu|sw_emu (optional, default hw)

TARGET ?= hw
N ?= 32

.PHONY: all kernel host run clean

all: kernel host

kernel:
	$(MAKE) -C kernel TARGET=$(TARGET) xclbin

host:
	$(MAKE) -C host all

run:
	./scripts/run.sh kernel/build/vadd.xclbin $(N)

clean:
	$(MAKE) -C kernel clean || true
	$(MAKE) -C host clean || true
	rm -rf build

.PHONY: csim cosim
csim:
	$(MAKE) -C kernel csim HLS_CFG=$(HLS_CFG) WORK_DIR=$(WORK_DIR)

cosim:
	$(MAKE) -C kernel cosim HLS_CFG=$(HLS_CFG) WORK_DIR=$(WORK_DIR)
