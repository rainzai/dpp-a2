/*
 * Scaled Dot-Product Attention Implementation
 * This file handles dataset loading, memory management, and the execution
 * CUDA kernels for Transformer attention mechanisms.
 *
 * Flags:
 * -b <runs> : Run benchmark mode
 * -v        : Enable verbose output for debugging results
 */

#include "attention.hh"
#include "timer.hh"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

constexpr size_t BATCH        = 1;
constexpr size_t SEQ_LEN      = 128;
constexpr size_t HEAD_DIM     = 768;
constexpr size_t NUM_ELEMENTS = BATCH * SEQ_LEN * HEAD_DIM;
constexpr size_t BLOCK_SIZE   = 32;

/*
 * Loads binary data as floats given a file path and number of elements.
 */
std::vector<float> load_binary(const std::string &path, size_t num_elements) {
    std::vector<float> data(num_elements);
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    auto bytes = std::as_writable_bytes(std::span(data));
    file.read(reinterpret_cast<char *>(bytes.data()), bytes.size_bytes());
    return data;
}

/*
 * Saves data of type float given a filename and the data.
 */
void save_to_binary(std::string_view filename, std::span<const float> data) {
    auto file = std::ofstream(filename.data(), std::ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open " << filename << "\n";
        return;
    }

    file.write(reinterpret_cast<const char *>(data.data()), data.size_bytes());
    file.close();
    if (file.good()) {
        std::cout << "Saved " << data.size() << " elements to " << filename << "\n";
    }
}

int main(int argc, char *argv[]) {
    bool run_benchmark = false;
    bool verbose       = false;
    int runs           = 1000;

    for (int i = 1; i < argc; i++) {
        std::string_view arg(argv[i]);
        if (arg == "-b") {
            run_benchmark = true;
            if (i + 1 < argc && std::isdigit(argv[i + 1][0])) {
                runs = std::atoi(argv[++i]);
            }
        } else if (arg == "-v") {
            verbose = true;
        }
    }
    timer t("Attention Benchmark");

    std::vector<float> h_Q = load_binary("../data/Q.bin", NUM_ELEMENTS);
    std::vector<float> h_K = load_binary("../data/K.bin", NUM_ELEMENTS);
    std::vector<float> h_V = load_binary("../data/V.bin", NUM_ELEMENTS);

    if (run_benchmark) {
        float total_kernel_ms = 0.0f;
        for (int i = 0; i < runs; i++) {
            t.start();
            total_kernel_ms +=
            run_attention_on_gpu(h_Q, h_K, h_V, HEAD_DIM, SEQ_LEN, BLOCK_SIZE).kernel_time_ms;
            t.stop();
        }
        t.print(std::cout);
        std::cout << "Average kernel runtime: " << (total_kernel_ms / runs) << " ms\n";
    }

    AttentionResult result =
    run_attention_on_gpu(h_Q, h_K, h_V, HEAD_DIM, SEQ_LEN, BLOCK_SIZE);

    if (verbose) {
        // Can be used for debug purposes / printing utilities
    }

    save_to_binary("result.bin", result.output);
    return EXIT_SUCCESS;
}
