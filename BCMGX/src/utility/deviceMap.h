#pragma once

#include "utility/utils.h"
#include <cuda.h>

template <typename OriginalType, typename MappedType, typename Mapper>
__global__ void _deviceMap(OriginalType* d_src, size_t n, Mapper mapper, MappedType* d_dst)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < n) {
        d_dst[tid] = mapper(d_src[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

template <typename OriginalType, typename MappedType, typename Mapper>
MappedType* deviceMap(OriginalType* d_arr, size_t n, Mapper mapper)
{
    if (!n) {
        return NULL;
    }

    MappedType* d_ret = NULL;
    CHECK_DEVICE(cudaMalloc(&d_ret, n * sizeof(MappedType)));

    GridBlock gb = getKernelParams(n);
    _deviceMap<<<gb.g, gb.b>>>(d_arr, n, mapper, d_ret);
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_DEVICE(err);

    return d_ret;
}
