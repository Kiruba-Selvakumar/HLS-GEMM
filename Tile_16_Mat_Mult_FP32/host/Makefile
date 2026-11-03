# Host build with Make (XRT C++ API)
# Files: host.cpp, cmdlineparser.cpp/.h, logger.cpp/.h
# Requirements:
#   - XRT installed and sourced: source /opt/xilinx/xrt/setup.sh
#   - XILINX_XRT env var set (points to XRT root)
# Usage:
#   make            # build release
#   make DEBUG=1    # build debug (-O0 -g)
#   make clean
#   make run XCLBIN=../kernel/build/mmult.xclbin DEV=0

# ====== Config ======
CXX      ?= g++
CXXSTD   ?= -std=c++17

# Toggle debug build: make DEBUG=1
ifeq ($(DEBUG),1)
  OPTFLAGS ?= -O0 -g
else
  OPTFLAGS ?= -O2
endif

WARNFLAGS ?= -Wall -Wextra -Wpedantic
CPPFLAGS  ?= -MMD -MP

# XRT include/lib from environment
ifndef XILINX_XRT
  $(error XILINX_XRT is not set. Please 'source /opt/xilinx/xrt/setup.sh')
endif
INCLUDES  := -I$(XILINX_XRT)/include
LDFLAGS   := -L$(XILINX_XRT)/lib

# Some systems need both xrt_coreutil and xrt_core; link both safely.
LDLIBS    := -lxrt_coreutil -lxrt_core -lOpenCL -lpthread -ldl

# ====== Paths ======
BUILD_DIR := build
TARGET    := $(BUILD_DIR)/host.exe

SRC := \
  host.cpp \
  cmdlineparser.cpp \
  logger.cpp

OBJ := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC))
DEP := $(OBJ:.o=.d)

# ====== Rules ======
.PHONY: all clean run

all: $(TARGET)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Pattern rule to compile .cpp -> build/.o
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXSTD) $(OPTFLAGS) $(WARNFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

# Link
$(TARGET): $(OBJ)
	$(CXX) $(OPTFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Run helper
# Example:
#   make run XCLBIN=../kernel/build/mmult.xclbin DEV=0
XCLBIN ?= ../kernel/build/mmult.xclbin
DEV    ?= 0

run: $(TARGET)
	@echo "[RUN] dev=$(DEV) xclbin=$(XCLBIN)"
	$(TARGET) --xclbin_file $(XCLBIN) --device_id $(DEV)

clean:
	@rm -rf $(BUILD_DIR)

# Auto-include dependency files
-include $(DEP)