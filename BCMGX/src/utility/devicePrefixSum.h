/**
 * @file
 */
#pragma once

#define USE_THRUST 1

#if USE_THRUST
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#else
#include <cub/cub.cuh>
#endif

/**
 * @brief Computes the prefix sum (inclusive scan) of a device array.
 * 
 * This function computes the prefix sum (inclusive scan) of a device array. Depending on the compile-time flag 
 * `USE_THRUST`, it will use Thrust's `inclusive_scan` for the operation. If `USE_THRUST` is not set, this function 
 * provides a placeholder for implementing CUB's `DeviceScan::InclusiveSum` or other scan algorithms.
 * 
 * A prefix sum is a sequence of partial sums of a given sequence, where each element is replaced by the sum of 
 * all previous elements including the current one.
 * 
 * @tparam T The type of the elements in the array (e.g., `int`, `float`).
 * @param d_array Pointer to the device array to compute the prefix sum on.
 * @param size The number of elements in the device array.
 * 
 * @return Void. The input array will be modified in place with the computed prefix sums.
 * 
 * @throws std::bad_alloc If memory allocation fails during the prefix sum operation.
 * @throws cudaError_t If any CUDA memory allocation or copy fails.
 * @throws cub::CubError If a CUB operation fails when implementing the alternative scan algorithm.
 */
template <typename T>
void devicePrefixSum(T* d_array, size_t size)
{
    if (!size) {
        return;
    }

#if USE_THRUST
    thrust::inclusive_scan(thrust::device, d_array, d_array + size, d_array);
// TODO check for errors
#else
// cub::DeviceScan::InclusiveSum
#endif
}

#undef USE_THRUST
