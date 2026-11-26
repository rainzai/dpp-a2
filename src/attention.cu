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

#define USE_TILED 1   // 0 = baseline, 1 = optimized

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

// scores kernel
// S = Q * K^T / sqrt(d_k)
__global__ void scores_kernel(const float *q, const float *k, float *s,
                              size_t seq_len, size_t head_dim) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < seq_len) {
        // dot product of row 'row' of Q with row 'col' of K
        float sum = 0.0f;
        for (size_t d = 0; d < head_dim; d++)
            sum += q[row * head_dim + d] * k[col * head_dim + d];

        // divide by sqrt(d_k)
        s[row * seq_len + col] = sum / sqrtf((float)head_dim);
    }
}

// tiled scores kernel (optimization)
// same result as scores_kernel, but Q and K are loaded in tiles into shared
// memory, so every value is read from global memory only once per block
__global__ void scores_tiled_kernel(const float *q, const float *k, float *s,
                                    size_t seq_len, size_t head_dim) {
    extern __shared__ float tile[];
    size_t b = blockDim.x;
    float *q_tile = tile;
    float *k_tile = tile + b * b;

    size_t tx = threadIdx.x, ty = threadIdx.y;
    size_t row   = blockIdx.y * b + ty;
    size_t col   = blockIdx.x * b + tx;
    size_t k_row = blockIdx.x * b + ty;

    float sum = 0.0f;
    for (size_t t = 0; t < head_dim; t += b) {
        // every thread loads one value of Q and one of K
        // this is coalesced
        // (+1 avoids bank conflicts)
        q_tile[ty * b + tx] = (row < seq_len && t + tx < head_dim)
                              ? q[row * head_dim + t + tx] : 0.0f;
        k_tile[ty * (b + 1) + tx] = (k_row < seq_len && t + tx < head_dim)
                                    ? k[k_row * head_dim + t + tx] : 0.0f;
        __syncthreads();
        // wait until the whole tile is loaded

        for (size_t d = 0; d < b; d++)
            sum += q_tile[ty * b + d] * k_tile[tx * (b + 1) + d];
        __syncthreads();
        // wait before the tile is overwritten
    }

    if (row < seq_len && col < seq_len)
        s[row * seq_len + col] = sum / sqrtf((float)head_dim);
}

// softmax kernel
// normalizes each row of S with softmax (numerically stable)
__global__ void softmax_kernel(float *s, size_t seq_len) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len) {
        float *r = s + row * seq_len;   // r[j] is S[row][j]

        // find the maximum of the row
        float max = r[0];
        for (size_t j = 1; j < seq_len; j++)
            max = fmaxf(max, r[j]);

        // exponentiate (minus max) and sum
        float sum = 0.0f;
        for (size_t j = 0; j < seq_len; j++) {
            r[j] = expf(r[j] - max);
            sum += r[j];
        }

        // divide by the sum
        for (size_t j = 0; j < seq_len; j++)
            r[j] /= sum;
    }
}

// output kernel
// O = S * V
__global__ void output_kernel(const float *s, const float *v, float *out,
                              size_t seq_len, size_t head_dim) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < head_dim) {
        // dot product of row 'row' of S with column 'col' of V
        float sum = 0.0f;
        for (size_t j = 0; j < seq_len; j++)
            sum += s[row * seq_len + j] * v[j * head_dim + col];

        out[row * head_dim + col] = sum;
    }
}

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

    // copy the inputs from cpu to gpu
    checkCudaCall(cudaMemcpy(d_q, h_q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_k, h_k.data(), qkv_bytes, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_v, h_v.data(), qkv_bytes, cudaMemcpyHostToDevice));

    // start timing
    cudaEvent_t start, stop;
    checkCudaCall(cudaEventCreate(&start));
    checkCudaCall(cudaEventCreate(&stop));
    checkCudaCall(cudaEventRecord(start));

    // launch scores kernel
    // one thread per element of S (seq_len x seq_len)
    dim3 block(block_size, block_size);
    dim3 grid_s((seq_len + block_size - 1) / block_size,
                (seq_len + block_size - 1) / block_size);
    #if USE_TILED
        size_t shmem = (block_size * block_size + block_size * (block_size + 1)) * sizeof(float);
        scores_tiled_kernel<<<grid_s, block, shmem>>>(d_q, d_k, d_s, seq_len, head_dim);
    #else
        scores_kernel<<<grid_s, block>>>(d_q, d_k, d_s, seq_len, head_dim);
    #endif
    checkCudaCall(cudaGetLastError());

    // launch softmax kernel
    // one thread per row of S
    int threads = block_size * block_size;
    int blocks  = (seq_len + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(d_s, seq_len);
    checkCudaCall(cudaGetLastError());

    // launch output kernel
    // one thread per element of O (seq_len x head_dim)
    dim3 grid_o((head_dim + block_size - 1) / block_size,
                (seq_len  + block_size - 1) / block_size);
    output_kernel<<<grid_o, block>>>(d_s, d_v, d_out, seq_len, head_dim);
    checkCudaCall(cudaGetLastError());

    // stop timing
    checkCudaCall(cudaEventRecord(stop));
    checkCudaCall(cudaEventSynchronize(stop));
    checkCudaCall(cudaEventElapsedTime(&elapsed, start, stop));
    checkCudaCall(cudaEventDestroy(start));
    checkCudaCall(cudaEventDestroy(stop));

    // copy the result back from gpu to cpu
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
