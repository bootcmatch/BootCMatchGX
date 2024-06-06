#pragma once

#include <cub/cub.cuh>

template <typename T>
struct IsNotEqual {
    T value;

    __host__ __device__ __forceinline__
    IsNotEqual(T value)
        : value(value)
    {
    }

    __host__ __device__ __forceinline__ bool operator()(const T& a) const
    {
        return a != value;
    }
};

template <typename T, typename F>
T* deviceFilter(T* d_arr, size_t len, F filter, size_t* filteredSize)
{
    if (!len) {
        *filteredSize = 0;
        return NULL;
    }

    T* d_filteredArr = NULL;
    CHECK_DEVICE(cudaMalloc(&d_filteredArr, len * sizeof(T)));

    int* d_filteredSize = NULL;
    CHECK_DEVICE(cudaMalloc(&d_filteredSize, sizeof(int)));

    // ---------------------------------------------------------------------
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::If(
        d_temp_storage,
        temp_storage_bytes,
        d_arr,
        d_filteredArr,
        d_filteredSize,
        len,
        filter);

    CHECK_DEVICE(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceSelect::If(
        d_temp_storage,
        temp_storage_bytes,
        d_arr,
        d_filteredArr,
        d_filteredSize,
        len,
        filter);
    // ---------------------------------------------------------------------

    int h_filteredSize = 0;
    CHECK_DEVICE(cudaMemcpy(
        (void*)&h_filteredSize,
        (void*)d_filteredSize,
        sizeof(int),
        cudaMemcpyDeviceToHost));

    cudaFree(d_temp_storage);
    cudaFree(d_filteredSize);

    *filteredSize = h_filteredSize;

    return d_filteredArr;
}
