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
 * @brief Computes the inclusive prefix sum of an array on the host.
 * 
 * This function computes the inclusive prefix sum (also known as the cumulative sum) of an array on the host.
 * The prefix sum is calculated in-place, meaning that each element at index `i` in the array is replaced by
 * the sum of all elements from index 0 to `i`. If the size of the array is zero, the function does nothing.
 * 
 * The function uses Thrust's `inclusive_scan` algorithm.
 *
 * @tparam T The type of elements in the array (e.g., `int`, `float`, etc.).
 * 
 * @param[in,out] d_array The array to compute the prefix sum. It will be modified in place.
 * @param[in] size The number of elements in the array.
 * 
 * @note The array is modified in place, with each element at index `i` representing the sum of elements from index 0 to `i`.
 * 
 * @see thrust::inclusive_scan for more information on Thrust’s `inclusive_scan`.
 */
template <typename T>
void hostPrefixSum(T* d_array, size_t size)
{
    if (!size) {
        return;
    }

#if USE_THRUST
    thrust::inclusive_scan(thrust::host, d_array, d_array + size, d_array);
// TODO check for errors
#else
// NOT IMPLEMENTED
#endif
}

#undef USE_THRUST
