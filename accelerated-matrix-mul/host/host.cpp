/**
* Host code for the tiled matrix multiplication accelerator.
* Based on the architecture from arXiv:2503.16731v2.
*/

#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <random>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// Define data type to match the paper's int8 quantization
typedef char T;

// Define matrix dimensions for a test case
// Corresponds to the DistilBERT attention case in the paper
const int N = 32;
const int K = 32;
const int M = 32;

// Function to initialize a matrix with random int8 values
void init_matrix(std::vector<T>& matrix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-10, 10);
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<T>(distrib(gen));
    }
}

// Function to perform CPU-based matrix multiplication for verification
void cpu_mmult(const std::vector<T>& a, const std::vector<T>& b, std::vector<int>& c) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < M; ++col) {
            int result = 0;
            for (int k = 0; k < K; ++k) {
                result += static_cast<int>(a[row * K + k]) * static_cast<int>(b[k * M + col]);
            }
            c[row * M + col] = result;
        }
    }
}

int main(int argc, char** argv) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "1");
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));
    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    auto krnl = xrt::kernel(device, uuid, "mmult");

    // --- Allocate and Initialize Host-Side Matrices ---
    std::cout << "Initializing host-side matrices...\n";
    std::vector<T> matrix_a(N * K);
    std::vector<T> matrix_b1(K * M); // First B matrix
    std::vector<T> matrix_b2(K * M); // Second B matrix to test data persistence
    std::vector<int> fpga_result(N * M);
    std::vector<int> cpu_ref_result(N * M);

    init_matrix(matrix_a);
    init_matrix(matrix_b1);
    init_matrix(matrix_b2);

    // --- Allocate Device-Side Buffers ---
    size_t size_a_bytes = matrix_a.size() * sizeof(T);
    size_t size_b_bytes = matrix_b1.size() * sizeof(T);
    size_t size_c_bytes = fpga_result.size() * sizeof(int);

    std::cout << "Allocate Buffers in Global Memory\n";
    auto bo_a = xrt::bo(device, size_a_bytes, krnl.group_id(0));
    auto bo_b = xrt::bo(device, size_b_bytes, krnl.group_id(1));
    auto bo_c = xrt::bo(device, size_c_bytes, krnl.group_id(2));

    // --- First Run: Load Matrix A and compute A x B1 ---
    std::cout << "\n--- KERNEL RUN 1: Loading Matrix A and computing A * B1 ---\n";
    bool update_A = true; // Set flag to true to load A

    // Write data to device buffers
    bo_a.write(matrix_a.data());
    bo_b.write(matrix_b1.data());
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Execution of the kernel...\n";
    auto run1 = krnl(bo_a, bo_b, bo_c, N, K, M, update_A);
    run1.wait();

    // Get the output and verify
    std::cout << "Get the output data from the device\n";
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_c.read(fpga_result.data());

    std::cout << "Calculating CPU reference for A * B1...\n";
    cpu_mmult(matrix_a, matrix_b1, cpu_ref_result);

    if (memcmp(fpga_result.data(), cpu_ref_result.data(), size_c_bytes)) {
        throw std::runtime_error("RUN 1 FAILED: Value read back does not match reference");
    }
    std::cout << "RUN 1 PASSED!\n";

    // --- Second Run: Reuse Matrix A and compute A x B2 ---
    std::cout << "\n--- KERNEL RUN 2: Reusing Matrix A and computing A * B2 ---\n";
    update_A = false; // Set flag to false to reuse A on-chip

    // Only write the new Matrix B
    bo_b.write(matrix_b2.data());
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Execution of the kernel...\n";
    auto run2 = krnl(bo_a, bo_b, bo_c, N, K, M, update_A);
    run2.wait();

    // Get the output and verify
    std::cout << "Get the output data from the device\n";
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_c.read(fpga_result.data());

    std::cout << "Calculating CPU reference for A * B2...\n";
    cpu_mmult(matrix_a, matrix_b2, cpu_ref_result);

    if (memcmp(fpga_result.data(), cpu_ref_result.data(), size_c_bytes)) {
        throw std::runtime_error("RUN 2 FAILED: Value read back does not match reference");
    }
    std::cout << "RUN 2 PASSED!\n";

    std::cout << "\nTEST PASSED\n";
    return 0;
}
