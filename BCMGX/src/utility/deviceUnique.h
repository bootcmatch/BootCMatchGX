#pragma once

#define USE_THRUST 0

#if USE_THRUST
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#else
#include <cub/cub.cuh>
#endif

template <typename T>
T* deviceUnique(T* d_arr, size_t len, size_t* uniqueSize)
{
    if (!len) {
        *uniqueSize = 0;
        return NULL;
    }

    T* d_uniqueArr = NULL;
    CHECK_DEVICE(cudaMalloc(&d_uniqueArr, len * sizeof(T)));

#if USE_THRUST
    T* d_uniqueArrEnd = thrust::unique_copy(
        thrust::device,
        d_arr,
        d_arr + len,
        d_uniqueArr);
    *uniqueSize = d_uniqueArrEnd - d_uniqueArr;
#else
    int* d_uniqueSize = NULL;
    CHECK_DEVICE(cudaMalloc(&d_uniqueSize, sizeof(int)));

    // ---------------------------------------------------------------------
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::Unique(
        d_temp_storage,
        temp_storage_bytes,
        d_arr,
        d_uniqueArr,
        d_uniqueSize,
        len);

    CHECK_DEVICE(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceSelect::Unique(
        d_temp_storage,
        temp_storage_bytes,
        d_arr,
        d_uniqueArr,
        d_uniqueSize,
        len);
    // ---------------------------------------------------------------------

    int h_uniqueSize = 0;
    CHECK_DEVICE(cudaMemcpy(
        (void*)&h_uniqueSize,
        (void*)d_uniqueSize,
        sizeof(int),
        cudaMemcpyDeviceToHost));

    cudaFree(d_temp_storage);
    cudaFree(d_uniqueSize);

    *uniqueSize = h_uniqueSize;
#endif

    return d_uniqueArr;
}

#undef USE_THRUST
