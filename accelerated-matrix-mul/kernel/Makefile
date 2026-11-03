# Kernel build with Make using v++
# Requires Vitis (v++ in PATH) and PLATFORM env set to .xpfm name

VPP ?= v++
TARGET ?= hw
PLATFORM ?= $(PLATFORM) # expect from env
KERNEL := mmult
SRC := mmult.cpp

BUILD_DIR := build
XO := $(BUILD_DIR)/$(KERNEL).xo
XCLBIN := $(BUILD_DIR)/$(KERNEL).xclbin
CONFIG_FILE := /home/sajiv.singh/PanCakes/accelerated-matrix-mul/profile_config.cfg

VPP_CFLAGS := -c -t $(TARGET) --platform $(PLATFORM) -k $(KERNEL) -g --config $(CONFIG_FILE)
VPP_LFLAGS := -l -t $(TARGET) --platform $(PLATFORM) -g --config $(CONFIG_FILE)

.PHONY: all xo xclbin clean

all: xclbin

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

xo: $(XO)
$(XO): $(SRC) | $(BUILD_DIR)
	@if [ -z "$(PLATFORM)" ]; then echo "Error: PLATFORM not set"; exit 1; fi
	$(VPP) $(VPP_CFLAGS) -o $@ $<

xclbin: $(XCLBIN)
$(XCLBIN): $(XO) | $(BUILD_DIR)
	$(VPP) $(VPP_LFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)


# --- HLS simulation (C-sim / Co-sim) with vitis-run (Vitis 2025.1) ---
# Provide a config describing sources, top, and test bench.
# Usage:
#   make csim  HLS_CFG=./hls/csim.cfg  WORK_DIR=./hls_csim
#   make cosim HLS_CFG=./hls/cosim.cfg WORK_DIR=./hls_cosim
HLS_CFG ?=
WORK_DIR ?= ./hls_work

.PHONY: csim cosim
csim:
	@if [ -z "$(HLS_CFG)" ]; then echo "Error: specify HLS_CFG=<path to csim cfg>"; exit 1; fi
	vitis-run --mode hls --csim --config $(HLS_CFG) --work_dir $(WORK_DIR)

cosim:
	@if [ -z "$(HLS_CFG)" ]; then echo "Error: specify HLS_CFG=<path to cosim cfg>"; exit 1; fi
	vitis-run --mode hls --cosim --config $(HLS_CFG) --work_dir $(WORK_DIR)
