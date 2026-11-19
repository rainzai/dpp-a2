/*
 * attention.hh
 *
 * Header file for the CUDA Scaled Dot-Product Attention Implementation
 *
 */

#ifndef ATTENTION_HH
#define ATTENTION_HH
#include <cstddef>
#include <vector>

/*
 * Output and GPU execution time of the attention operation.
 * output: Output vector on host [seq_len x head_dim]
 * kernel_time_ms: Pure kernel runtime in milliseconds
 */
struct AttentionResult {
    std::vector<float> output;
    float kernel_time_ms{ 0.0f };
};

/*
 * Manages the attention operation: memory allocation/transfers
 * and kernel launches (QK -> Softmax -> SV).
 * h_q, h_k, h_v: Input feature vectors from host
 * head_dim: Feature dimension size (d_k)
 * seq_len: Number of tokens in the sequence
 * block_size: The number of threads per block dimension
 */
AttentionResult run_attention_on_gpu(const std::vector<float> &h_q,
                                     const std::vector<float> &h_k,
                                     const std::vector<float> &h_v,
                                     const size_t head_dim,
                                     const size_t seq_len,
                                     const size_t block_size);

#endif
