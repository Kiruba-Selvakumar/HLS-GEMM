# ============================================================
# Top-Level Makefile for GNN Project
# ============================================================
 
TARGET ?= hw  # hw_emu | hw
PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1
XCLBIN ?= kernel/build/gcn_layer_hls.$(TARGET).xclbin
HLS_CFG ?= hls_csim.cfg
WORK_HLS ?= build/hls
 
# Place the temp config in the ROOT directory so relative paths work
HLS_TEST_CFG := test_mode.cfg 
 
# --- LOGIC TO FORCE TEST MODE FOR EMULATION ---
COMMON_FLAGS :=
ifeq ($(TARGET),hw_emu)
    COMMON_FLAGS += -DTEST_MODE
endif
 
.PHONY: all csim csynth cosim kernel host run clean
 
# -------------------------
# HLS: C-simulation
# -------------------------
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
 
# -------------------------
# HLS: Synthesis
# -------------------------
csynth:
	@echo "=== HLS Synthesis ==="
	@mkdir -p $(WORK_HLS)
	v++ -c --mode hls --config $(HLS_CFG) --work_dir $(WORK_HLS)
 
# -------------------------
# HLS: Co-simulation
# -------------------------
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
 
# -------------------------
# Vitis kernel build
# -------------------------
kernel:
	@echo "=== Building Kernel XCLBIN for TARGET=$(TARGET) ==="
	# Pass COMMON_FLAGS to kernel makefile (-D is supported by standard v++ kernel compiler)
	$(MAKE) -C kernel TARGET=$(TARGET) PLATFORM=$(PLATFORM) EXTRA_FLAGS="$(COMMON_FLAGS)" xclbin
 
# -------------------------
# Host build
# -------------------------
host:
	@echo "=== Building Host ==="
	$(MAKE) -C host EXTRA_FLAGS="$(COMMON_FLAGS)" all
 
# -------------------------
# Full build
# -------------------------
all: kernel host
 
# -------------------------
# Run (hw or hw_emu)
# -------------------------
run: all
	@echo "=== Running Host for TARGET=$(TARGET) ==="
ifeq ($(strip $(TARGET)),hw_emu)
	@echo "=== Starting Hardware Emulation ==="
	emconfigutil --platform $(PLATFORM) --od build
	export XCL_EMULATION_MODE=hw_emu; \
	./host/build/host.exe --xclbin_file $(XCLBIN) --device_id 0 -s
else
	@echo "=== Starting Hardware Execution ==="
	env -u XCL_EMULATION_MODE \
	./host/build/host.exe --xclbin_file $(XCLBIN) --device_id 0
endif
 
# -------------------------
# Clean
# -------------------------
clean:
	@echo "=== Cleaning ==="
	$(MAKE) -C kernel clean || true
	$(MAKE) -C host clean || true
	rm -rf build .Xil .vitis-run
