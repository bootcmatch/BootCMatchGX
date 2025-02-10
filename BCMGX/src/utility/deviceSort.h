/**
 * @file
 */
#pragma once

#define USE_THRUST 0
#define USE_RADIX_SORT 0

/**
 * @brief Sorts a device array using either Thrust, CUB's Merge Sort, or Radix Sort based on compile-time flags.
 * 
 * This function sorts a device array of elements using the specified comparator. Depending on the compile-time 
 * flags (`USE_THRUST` and `USE_RADIX_SORT`), it will either use Thrust's `sort`, CUB's `MergeSort`, or CUB's `RadixSort` 
 * to perform the sorting. If `USE_THRUST` is set to 1, it will use Thrust's sorting algorithms; if `USE_RADIX_SORT` is set 
 * to 1, it will use CUB's Radix Sort; otherwise, it defaults to CUB's Merge Sort.
 * 
 * The `deviceSort` function serves as the main entry point for sorting and selects the sorting method based on the 
 * flags defined at compile time.
 * 
 * @tparam T The type of the elements in the array (e.g., `int`, `float`).
 * @tparam K The type of the keys used for sorting in Radix Sort.
 * @tparam Comparator The type of the comparator function or functor used for sorting.
 * @param d_array Pointer to the device array to be sorted.
 * @param size The number of elements in the device array.
 * @param comparator The comparator function or functor to use for sorting.
 * 
 * @return Void.
 * 
 * @throws std::bad_alloc If memory allocation fails during sorting operations.
 * @throws cudaError_t If any CUDA memory copy or allocation fails.
 * @throws cub::CubError If there is a failure in the CUB library during sorting operations.
 */
template <typename T, typename K, typename Comparator>
void deviceSort(T* d_array, size_t size, Comparator comparator);

#if USE_THRUST
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

template <typename T, typename K, typename Comparator>
void deviceSort(T* d_array, size_t size, Comparator comparator)
{
    if (!size) {
        return;
    }

    thrust::sort(
        thrust::device,
        d_array,
        d_array + size,
        comparator);
}

#else
#include "deviceMap.h"
#include "utility/memory.h"
#include <cub/cub.cuh>

template <typename T, typename K, typename Comparator>
void deviceMergeSort(T* d_array, size_t size, Comparator comparator)
{
    if (!size) {
        return;
    }

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_array, size, comparator);

    // Allocate temporary storage
    d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_array, size, comparator);

    CUDA_FREE(d_temp_storage);
}

template <typename T, typename K, typename Comparator>
void deviceRadixSort(T* d_array, size_t size, Comparator comparator)
{
    if (!size) {
        return;
    }

    K* d_keys_in = deviceMap<T, K, Comparator>(d_array, size, comparator);
    K* d_keys_out = NULL;
    T* d_array_out = NULL;
    d_keys_out = CUDA_MALLOC(K, size);
    d_array_out = CUDA_MALLOC(T, size);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_array, d_array_out, size);

    // Allocate temporary storage
    d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_array, d_array_out, size);

    CHECK_DEVICE(cudaMemcpy(d_array, d_array_out, size * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_FREE(d_temp_storage);
    CUDA_FREE(d_keys_in);
    CUDA_FREE(d_keys_out);
    CUDA_FREE(d_array_out);
}

template <typename T, typename K, typename Comparator>
void deviceSort(T* d_array, size_t size, Comparator comparator)
{
    if (!size) {
        return;
    }

#if USE_RADIX_SORT
    deviceRadixSort<T, K, Comparator>(d_array, size, comparator);
#else
    deviceMergeSort<T, K, Comparator>(d_array, size, comparator);
#endif
}

#endif

#undef USE_THRUST
#undef USE_RADIX_SORT
