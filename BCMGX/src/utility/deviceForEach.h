/**
 * @file
 */
#pragma once

#include "utility/utils.h"
#include <cuda.h>

/**
 * @brief Functor that fills each element of an array with its index value.
 * 
 * This functor is used in the `deviceForEach` function to set each element of an array to its index value. It is intended 
 * to be passed as a parameter to the `deviceForEach` function, which will apply it to each element of a device array in parallel.
 * 
 * @tparam T The type of the elements in the array.
 */
template <typename T>
struct FillWithIndexOperator {
    /**
     * @brief Fills the given element with its index value.
     * 
     * This function sets the given element in the device array to its index value.
     * 
     * @param index The index of the current element in the array.
     * @param item A reference to the element in the array to be filled with its index.
     */
    __device__ void operator()(int index, T& item)
    {
        item = index;
    }
};

/**
 * @brief CUDA kernel that applies an operation to each element of a device array.
 * 
 * This kernel applies the given `Operator` to each element in the input array `d_arr`. The operation is performed in parallel
 * across multiple threads, where each thread operates on a different element of the array.
 * 
 * @tparam T The type of the elements in the array.
 * @tparam Operator The type of the operation that will be applied to each element.
 * 
 * @param d_arr Pointer to the input device array.
 * @param n The number of elements in the array.
 * @param op The operation (or functor) to be applied to each element in the array.
 */
template <typename T, typename Operator>
__global__ void _deviceForEach(T* d_arr, size_t n, Operator op)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < n) {
        op(tid, d_arr[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief Applies an operation to each element of a device array.
 * 
 * This function applies the provided `Operator` to each element of the device array `d_arr`. It is executed on the GPU using 
 * a CUDA kernel, and each element is processed in parallel by multiple threads.
 * 
 * @tparam T The type of the elements in the array.
 * @tparam Operator The type of the operation (or functor) to be applied to each element.
 * 
 * @param d_arr Pointer to the input device array.
 * @param n The number of elements in the array.
 * @param op The operation (or functor) to be applied to each element in the array.
 * 
 * @throws cudaError_t If any CUDA operations fail (e.g., kernel execution or memory operations).
 */
template <typename T, typename Operator>
void deviceForEach(T* d_arr, size_t n, Operator op)
{
    if (!n) {
        return;
    }

    GridBlock gb = getKernelParams(n);
    _deviceForEach<<<gb.g, gb.b>>>(d_arr, n, op);
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_DEVICE(err);
}
