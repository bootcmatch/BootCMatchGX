#pragma once

#define USE_THRUST 0
#define USE_RADIX_SORT 0

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
    CHECK_DEVICE(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run sorting operation
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_array, size, comparator);

    cudaFree(d_temp_storage);
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
    CHECK_DEVICE(cudaMalloc(&d_keys_out, size * sizeof(K)));
    CHECK_DEVICE(cudaMalloc(&d_array_out, size * sizeof(T)));

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_array, d_array_out, size);

    // Allocate temporary storage
    CHECK_DEVICE(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_array, d_array_out, size);

    CHECK_DEVICE(cudaMemcpy(d_array, d_array_out, size * sizeof(T), cudaMemcpyDeviceToDevice));
    cudaFree(d_temp_storage);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_array_out);
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
