#pragma once

#include "utility/utils.h"
#include <cuda.h>

template <typename T>
struct FillWithIndexOperator {
    __device__ void operator()(int index, T& item)
    {
        item = index;
    }
};

template <typename T, typename Operator>
__global__ void _deviceForEach(T* d_arr, size_t n, Operator op)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < n) {
        op(tid, d_arr[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

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
