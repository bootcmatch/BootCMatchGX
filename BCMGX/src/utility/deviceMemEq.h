#pragma once

#include "utility/memory.h"
#include "utility/utils.h"
#include <cuda.h>

template <typename T>
__global__ void _deviceMemEq(const T* a, const T* b, int* result, size_t size)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        if (a[idx] != b[idx]) {
            atomicExch(result, 1); // scrive solo se diverso
        }
    }
}

template <typename T>
bool deviceMemEq(const T* a, const T* b, size_t size)
{
    int* d_result;
    int h_result = 0;

    CHECK_DEVICE(cudaMalloc(&d_result, sizeof(int)));
    CHECK_DEVICE(cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice));

    const int threadsPerBlock = MAX_THREADS;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    _deviceMemEq<T><<<blocks, threadsPerBlock>>>(a, b, d_result, size);
    CHECK_DEVICE(cudaDeviceSynchronize());

    CHECK_DEVICE(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_DEVICE(cudaFree(d_result));

    return (h_result == 0); // true se uguali
}
