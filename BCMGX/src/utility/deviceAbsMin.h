#pragma once

#include "utility/memory.h"
#include "utility/utils.h"
#include <cuda.h>

struct AbsMin {
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T& lhs, const T& rhs) const
    {
        T ab_lhs = fabs(lhs);
        T ab_rhs = fabs(rhs);
        return ab_lhs < ab_rhs ? ab_lhs : ab_rhs;
    }
};

template <typename T>
T deviceAbsMin(T* a, size_t n)
{
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    T h_ret = 0;
    T* d_ret = CUDA_MALLOC(T, 1, true);

    AbsMin absmin;
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, a, d_ret, n, absmin, DBL_MAX);
    d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes);
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, a, d_ret, n, absmin, DBL_MAX);

    CUDA_FREE(d_temp_storage);
    CHECK_DEVICE(cudaMemcpy(&h_ret, d_ret, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_FREE(d_ret);

    return h_ret;
}
