/**
 * @file
 */
#pragma once

#define USE_THRUST 0

#if USE_THRUST
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#else
#include <cub/cub.cuh>
#endif

#include "utility/memory.h"

/**
 * @brief Removes duplicate elements from a device array and returns a unique array.
 * 
 * This function removes duplicates from a given device array `d_arr` and returns a new device array 
 * containing only the unique elements. It leverages Thrust or CUB libraries for efficient device-level 
 * unique element extraction. The result is stored in `d_uniqueArr`, and the size of the unique array 
 * is returned in the `uniqueSize` pointer.
 * 
 * The function checks whether the `len` (size of the input array) is zero and returns `NULL` and sets 
 * `uniqueSize` to zero if true.
 * 
 * @tparam T The type of the elements in the array (e.g., `int`, `float`).
 * @param d_arr Pointer to the input device array to extract unique elements from.
 * @param len The number of elements in the input array `d_arr`.
 * @param uniqueSize Pointer to the variable that will hold the size of the unique array.
 * 
 * @return A pointer to a new device array containing the unique elements from the input array.
 * 
 * @throws std::bad_alloc If memory allocation fails for any CUDA memory operations.
 * @throws cudaError_t If any CUDA memory copy or allocation fails.
 * @throws cub::CubError If there is a failure in the CUB library during unique element extraction.
 */
template <typename T>
T* deviceUnique(T* d_arr, size_t len, size_t* uniqueSize)
{
    if (!len) {
        *uniqueSize = 0;
        return NULL;
    }

    T* d_uniqueArr = CUDA_MALLOC(T, len, true);

#if USE_THRUST
    T* d_uniqueArrEnd = thrust::unique_copy(
        thrust::device,
        d_arr,
        d_arr + len,
        d_uniqueArr);
    *uniqueSize = d_uniqueArrEnd - d_uniqueArr;
#else
    int* d_uniqueSize = CUDA_MALLOC(int, 1, true);

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

    d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes);

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

    CUDA_FREE(d_temp_storage);
    CUDA_FREE(d_uniqueSize);

    *uniqueSize = h_uniqueSize;
#endif

    return d_uniqueArr;
}

#undef USE_THRUST
