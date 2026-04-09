#pragma once

#include "utility/memory.h"
#include "utility/utils.h"
#include <cuda.h>

template <typename T>
T deviceMax(T* a, size_t n)
{
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    T h_ret = 0;
    T* d_ret = CUDA_MALLOC(T, 1, true);

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, a, d_ret, n);
    d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, a, d_ret, n);

    CUDA_FREE(d_temp_storage);
    CHECK_DEVICE(cudaMemcpy(&h_ret, d_ret, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_FREE(d_ret);

    return h_ret;
}
