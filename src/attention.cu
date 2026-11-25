/*
 * Implementation of Scaled Dot-Product Attention, parallelized on the GPU using
 * CUDA.
 *
 * Names:
 * Ronald Hendrik Matthew Koning
 * Sean Zhu Liong Li

 * Student IDs: 16437284, 16068912
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

    // output variables
    std::vector<float> h_out(h_q.size());
    float elapsed = 0.0f;

    // amount of bytes each matrix takes
    size_t qkv_bytes = seq_len * head_dim * sizeof(float);
    size_t s_bytes   = seq_len * seq_len  * sizeof(float);

    // allocate gpu memory
    float *d_q, *d_k, *d_v, *d_s, *d_out;
    checkCudaCall(cudaMalloc(&d_q,   qkv_bytes));
    checkCudaCall(cudaMalloc(&d_k,   qkv_bytes));
    checkCudaCall(cudaMalloc(&d_v,   qkv_bytes));
    checkCudaCall(cudaMalloc(&d_s,   s_bytes));
    checkCudaCall(cudaMalloc(&d_out, qkv_bytes));

    // copy the inputs from cpu to gpu (H2D)
    checkCudaCall(cudaMemcpy(d_q, h_q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_k, h_k.data(), qkv_bytes, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_v, h_v.data(), qkv_bytes, cudaMemcpyHostToDevice));

    // TODO: kernel launch

    // copy the result back from gpu to cpu (D2H)
    checkCudaCall(cudaMemcpy(h_out.data(), d_out, qkv_bytes, cudaMemcpyDeviceToHost));

    // free gpu memory
    checkCudaCall(cudaFree(d_q));
    checkCudaCall(cudaFree(d_k));
    checkCudaCall(cudaFree(d_v));
    checkCudaCall(cudaFree(d_s));
    checkCudaCall(cudaFree(d_out));

    AttentionResult res = { h_out, elapsed };
    return res;
}
