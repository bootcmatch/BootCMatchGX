/**
 * @file
 */
#pragma once

#include "utility/memory.h"
#include "utility/utils.h"
#include <cuda.h>

/**
 * @brief Maps each element of a device array to a new value using a provided mapping function.
 * 
 * This function applies a transformation to each element in the input device array `d_arr` by using a user-defined 
 * mapping function `mapper`. The result of applying `mapper` to each element of the input array is stored in the 
 * output device array `d_ret`.
 * 
 * The mapping is performed in parallel across multiple threads using a CUDA kernel. Each thread applies the mapping 
 * function to a single element of the input array.
 * 
 * @tparam OriginalType The type of the elements in the input array `d_arr`.
 * @tparam MappedType The type of the elements in the output array `d_ret`.
 * @tparam Mapper The type of the mapping function used to transform each element of `d_arr`.
 * 
 * @param d_arr Pointer to the input device array.
 * @param n The number of elements in the input array.
 * @param mapper A function (or functor) that takes an element of type `OriginalType` and returns an element of type 
 *        `MappedType`.
 * 
 * @return Pointer to the output device array, which contains the mapped elements.
 * 
 * @throws std::bad_alloc If memory allocation for the output array fails.
 * @throws cudaError_t If any CUDA operations fail (e.g., kernel execution or memory operations).
 */
template <typename OriginalType, typename MappedType, typename Mapper>
__global__ void _deviceMap(OriginalType* d_src, size_t n, Mapper mapper, MappedType* d_dst)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < n) {
        d_dst[tid] = mapper(d_src[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief CUDA kernel that applies a mapping function to each element of a device array.
 * 
 * This kernel applies the given `mapper` function to each element in the input array `d_src` and stores the results
 * in the output array `d_dst`. The kernel is designed to run in parallel across multiple threads, with each thread 
 * processing one element of the array.
 * 
 * @tparam OriginalType The type of the elements in the input array `d_src`.
 * @tparam MappedType The type of the elements in the output array `d_dst`.
 * @tparam Mapper The type of the mapping function used to transform each element of `d_src`.
 * 
 * @param d_src Pointer to the input device array.
 * @param n The number of elements in the input array.
 * @param mapper A function (or functor) that transforms an element of type `OriginalType` to `MappedType`.
 * @param d_dst Pointer to the output device array where the mapped results will be stored.
 */
template <typename OriginalType, typename MappedType, typename Mapper>
MappedType* deviceMap(OriginalType* d_arr, size_t n, Mapper mapper)
{
    if (!n) {
        return NULL;
    }

    MappedType* d_ret = CUDA_MALLOC(MappedType, n);
    GridBlock gb = getKernelParams(n);
    _deviceMap<<<gb.g, gb.b>>>(d_arr, n, mapper, d_ret);
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_DEVICE(err);

    return d_ret;
}
