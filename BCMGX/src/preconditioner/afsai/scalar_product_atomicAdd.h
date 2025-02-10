/** @file */
#pragma once

/**
 * @brief Performs a parallelized multi-scalar product operation.
 * 
 * This CUDA kernel computes the element-wise product of two input vectors `a` and `b`, 
 * and accumulates the result into an output array `out`. The accumulation is done based on 
 * the index modulo `n`, allowing for a multi-scalar product across multiple blocks.
 * The kernel uses the `atomicAdd` function to ensure thread safety when updating the result 
 * in the output array `out`.
 * 
 * The multi-scalar product operation can be useful in various scenarios, such as 
 * calculating dot products or computing other types of reduced sums in parallel.
 * 
 * @param a A pointer to the input array `a`, where each element represents a scalar factor.
 * @param b A pointer to the input array `b`, where each element represents a scalar factor.
 * @param out A pointer to the output array where the accumulated results are stored.
 * @param n The number of elements in the vectors `a` and `b`. The output will be accumulated based on the modulo `n`.
 * @param mr A factor that expands the total number of elements processed by the kernel. The kernel processes `n * mr` elements in total.
 * 
 * @note This kernel should be called with an appropriate number of threads and blocks to ensure efficient execution.
 * @note The `atomicAdd` operation ensures that the result accumulation is thread-safe, but may introduce some performance overhead due to synchronization.
 */
__global__ void multiscalar_product(REALafsai* a, REALafsai* b, REALafsai* out, int n, int mr)
{
    REALafsai scratch;
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int rid = tid % n;
    if (tid < (n * mr)) {
        scratch = a[tid] * b[tid];
        atomicAdd(&out[rid], scratch);
    }
}
