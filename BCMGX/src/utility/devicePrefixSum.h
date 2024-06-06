#pragma once

#define USE_THRUST 1

#if USE_THRUST
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#else
#include <cub/cub.cuh>
#endif

template <typename T>
void devicePrefixSum(T* d_array, size_t size)
{
    if (!size) {
        return;
    }

#if USE_THRUST
    thrust::inclusive_scan(thrust::device, d_array, d_array + size, d_array);
// TODO check for errors
#else
// cub::DeviceScan::InclusiveSum
#endif
}

#undef USE_THRUST
