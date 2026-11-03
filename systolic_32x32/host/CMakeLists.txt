cmake_minimum_required(VERSION 3.16)
project(VectorAddHost CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# XRT include and library paths from environment
if(DEFINED ENV{XILINX_XRT})
  include_directories($ENV{XILINX_XRT}/include)
  link_directories($ENV{XILINX_XRT}/lib)
else()
  message(FATAL_ERROR "XILINX_XRT environment variable not set. Please source XRT setup.")
endif()

add_executable(host host.cpp)
target_link_libraries(host xrt_coreutil OpenCL pthread)

# Install/Copy convenience
set_target_properties(host PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
