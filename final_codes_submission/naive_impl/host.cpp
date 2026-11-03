// This code relies on a non-standard "cmdlineparser.h" file.
// You must have this file available in your include path to compile.
#include "cmdlineparser.h" 

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>    // <--- ADDED: For std::ifstream and std::ofstream
#include <algorithm>  // <--- ADDED: For std::fill
#include <unistd.h> // For usleep

// XRT includes
// NOTE: The 'experimental/' API is deprecated.
// Consider using the modern API (e.g., #include "xrt/xrt_bo.h")
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

/// Maximum dimensions supported by kernel 
constexpr int MAX_M = 256;
constexpr int MAX_N = 256;
constexpr int MAX_P = 256;

// Utility to read a matrix from a text file into a vector
// FIXED: Changed std::vector& to std::vector<float>&
bool read_matrix_from_file(const std::string& filename, std::vector<float>& data, int expected_size) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open file " << filename << "\n";
        return false;
    }
    float val;
    while (fin >> val) {
        data.push_back(val);
    }
    fin.close();
    if ((int)data.size() != expected_size) {
        std::cerr << "Warning: expected " << expected_size << " elements, got " << data.size() << "\n";
        return false;
    }
    return true;
}

// Utility to write a matrix to a file
// FIXED: Changed const std::vector& to const std::vector<float>&
void write_matrix_to_file(const std::string& filename, const std::vector<float>& data, int rows, int cols) {
    std::ofstream fout(filename);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fout << data[i * cols + j] << " ";
        }
        fout << "\n";
    }
    fout.close();
}

int main(int argc, char** argv) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "1");
    parser.addSwitch("--m", "", "Matrix dimension M (rows of A, C)", "32");
    parser.addSwitch("--n", "", "Matrix dimension N (cols of A / rows of B)", "32");
    parser.addSwitch("--p", "", "Matrix dimension P (cols of B, C)", "32");
    parser.addSwitch("--a_file", "", "Path to matrix A text file", "A.txt");
    parser.addSwitch("--b_file", "", "Path to matrix B text file", "B.txt");
    parser.addSwitch("--c_ref", "", "Path to reference C text file", "C_ref.txt");
    parser.addSwitch("--c_out", "", "Path to output C text file", "C_out.txt");
    parser.parse(argc, argv);

    std::string binaryFile = parser.value("xclbin_file");
    int device_index = std::stoi(parser.value("device_id"));
    int M = std::stoi(parser.value("m"));
    int N = std::stoi(parser.value("n"));
    int P = std::stoi(parser.value("p"));
    std::string a_file = parser.value("a_file");
    std::string b_file = parser.value("b_file");
    std::string c_ref_file = parser.value("c_ref");
    std::string c_out_file = parser.value("c_out");

    if (M > MAX_M || N > MAX_N || P > MAX_P) {
        std::cerr << "Error: matrix size exceeds kernel maximum ("
                  << MAX_M << "," << MAX_N << "," << MAX_P << ")\n";
        return EXIT_FAILURE;
    }

    std::cout << "Running " << M << "x" << N << " * " << N << "x" << P << " matrix multiply\n";

    auto device = xrt::device(device_index);
    auto uuid = device.load_xclbin(binaryFile);
    auto krnl = xrt::kernel(device, uuid, "mmult_base");

    size_t size_a_bytes = sizeof(float) * M * N;
    size_t size_b_bytes = sizeof(float) * N * P;
    size_t size_c_bytes = sizeof(float) * M * P;

    auto boA = xrt::bo(device, size_a_bytes, krnl.group_id(0));
    auto boB = xrt::bo(device, size_b_bytes, krnl.group_id(1));
    auto boC = xrt::bo(device, size_c_bytes, krnl.group_id(2));

    auto A = boA.map<float*>();
    auto B = boB.map<float*>();
    auto C = boC.map<float*>();

    // Read matrices from files
    std::vector<float> A_vec, B_vec;
    if (!read_matrix_from_file(a_file, A_vec, M * N)) return EXIT_FAILURE;
    if (!read_matrix_from_file(b_file, B_vec, N * P)) return EXIT_FAILURE;

    std::memcpy(A, A_vec.data(), size_a_bytes);
    std::memcpy(B, B_vec.data(), size_b_bytes);
    std::fill(C, C + (M * P), 0.0f); // Requires <algorithm>

    // Sync to device
    boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Launch kernel
    auto t1 = std::chrono::high_resolution_clock::now();
    auto run = krnl(boA, boB, boC, M, N, P);
    run.wait();
    auto t2 = std::chrono::high_resolution_clock::now();

    auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Kernel execution time: " << elapsed_us << " us\n";

    // Get output from device
    boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<float> C_vec(C, C + (M * P));

    // Write output to file
    write_matrix_to_file(c_out_file, C_vec, M, P);
    std::cout << "Output matrix written to " << c_out_file << "\n";

    // Optional verification from reference file
    std::vector<float> C_ref;
    if (read_matrix_from_file(c_ref_file, C_ref, M * P)) {
        int errors = 0;
        const float eps = 1e-3f;
        for (int i = 0; i < M * P; ++i) {
            float diff = std::fabs(C_vec[i] - C_ref[i]);
            if (diff > eps) {
                if (errors < 10)
                    std::cerr << "Mismatch at index " << i
                              << " FPGA=" << C_vec[i] << " REF=" << C_ref[i]
                              << " diff=" << diff << "\n";
                ++errors;
            }
        }
        if (errors)
            std::cerr << "TEST FAILED with " << errors << " mismatches\n";
        else
            std::cout << "TEST PASSED\n";
    } else {
        std::cerr << "No reference file found — skipping verification.\n";
    }

    // --- ADD THIS ---
// Give XRT a moment (e.g., 100ms) to finalize
// profile data before the host program exits.
usleep(100000);

    return 0;
}
