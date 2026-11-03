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
WORK_HLS ?= build/hls
HLS_CFG ?= ./hls/csim.cfg
HLS_TEST_CFG ?= csim_temp.cfg

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
	@echo "=== HLS C-simulation ==="
	@mkdir -p $(WORK_HLS)
	# 1. Create temp config in ROOT dir
	cat $(HLS_CFG) > $(HLS_TEST_CFG)
	@echo "" >> $(HLS_TEST_CFG)
	@echo "[hls]" >> $(HLS_TEST_CFG)
	@echo "csim.cflags=-DTEST_MODE" >> $(HLS_TEST_CFG)
	# 2. Run
	vitis-run --mode hls --csim --config $(HLS_TEST_CFG) --work_dir $(WORK_HLS)
	# 3. Cleanup
	rm -f $(HLS_TEST_CFG)


csynth:
	@echo "=== HLS Synthesis ==="
	@mkdir -p $(WORK_HLS)
	v++ -c --mode hls --config $(HLS_CFG) --work_dir $(WORK_HLS)


cosim:
	@echo "=== HLS Co-simulation ==="
	@mkdir -p $(WORK_HLS)
	# 1. Create temp config in ROOT dir
	cat $(HLS_CFG) > $(HLS_TEST_CFG)
	@echo "" >> $(HLS_TEST_CFG)
	@echo "[hls]" >> $(HLS_TEST_CFG)
	@echo "syn.cflags=-DTEST_MODE" >> $(HLS_TEST_CFG)
	@echo "csim.cflags=-DTEST_MODE" >> $(HLS_TEST_CFG)
	@echo "cosim.cflags=-DTEST_MODE" >> $(HLS_TEST_CFG)
	
	# 2. Run Synthesis first (REQUIRED for cosim)
	v++ -c --mode hls --config $(HLS_TEST_CFG) --work_dir $(WORK_HLS)
	
	# 3. Run Co-simulation
	vitis-run --mode hls --cosim --config $(HLS_TEST_CFG) --work_dir $(WORK_HLS)
	
	# 4. Cleanup
	rm -f $(HLS_TEST_CFG)
