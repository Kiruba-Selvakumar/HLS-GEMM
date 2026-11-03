cmake_minimum_required(VERSION 3.16)
project(VectorAddKernel)

# Expect PLATFORM (.xpfm) in environment
if(NOT DEFINED ENV{PLATFORM})
  message(FATAL_ERROR "PLATFORM env var is not set. Export PLATFORM to your .xpfm name (e.g., xilinx_u50_gen3x16_xdma_5_202210_1).")
endif()

# Target can be hw, hw_emu, or sw_emu
if(NOT DEFINED TARGET)
  set(TARGET "hw")
endif()

set(KERNEL_SRC vadd.cpp)
set(KERNEL_NAME vadd)
set(XO_FILE ${CMAKE_BINARY_DIR}/${KERNEL_NAME}.xo)
set(XCLBIN_FILE ${CMAKE_BINARY_DIR}/${KERNEL_NAME}.xclbin)

# Compile kernel to XO
add_custom_command(
  OUTPUT ${XO_FILE}
  COMMAND v++ -c -t ${TARGET} --platform $ENV{PLATFORM} -k ${KERNEL_NAME} -o ${XO_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_SRC}
  DEPENDS ${KERNEL_SRC}
  VERBATIM
)

# Link XO to XCLBIN
add_custom_command(
  OUTPUT ${XCLBIN_FILE}
  COMMAND v++ -l -t ${TARGET} --platform $ENV{PLATFORM} ${XO_FILE} -o ${XCLBIN_FILE}
  DEPENDS ${XO_FILE}
  VERBATIM
)

add_custom_target(build_kernel ALL DEPENDS ${XCLBIN_FILE})
