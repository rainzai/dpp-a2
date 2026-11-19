/*
 * Implementation of Scaled Dot-Product Attention, parallelized on the GPU using
 * CUDA.
 *
 * Names:
 * Student IDs:
 */

#include "attention.hh"
#include <iostream>
#include <math_constants.h>
#include <vector>

/* Utility function, use to do error checking for CUDA calls
 *
 * Use this function like this:
 *     checkCudaCall(<cuda_call>);
 *
 * For example:
 *     checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
 *
 * Special case to check the result of the last kernel invocation:
 *     kernel<<<...>>>(...);
 *     checkCudaCall(cudaGetLastError());
 **/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "cuda error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// YOUR CODE HERE: CUDA kernels for the operations

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
                                     const size_t block_size) {

    // YOUR CODE HERE: Call your CUDA kernels and return resulting array
    std::vector<float> h_out(h_q.size());
    float elapsed;
    AttentionResult res = { h_out, elapsed };
    return res;
}
