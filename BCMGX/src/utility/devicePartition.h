/**
 * @file
 */
#pragma once

#include "utility/utils.h"
#include <cub/cub.cuh>
#include <cuda.h>

/**
 * @brief Partitions a device array based on a condition provided by the operator.
 * 
 * This function partitions the input device array into two parts: one that satisfies the provided condition (using 
 * the operator `select_op`) and the other that does not. It uses CUB's `DevicePartition::If` for the partitioning 
 * operation. The elements satisfying the condition are moved to the output array `d_out`. The function returns a 
 * pointer to the partitioned array, where the first part contains the selected elements.
 * 
 * The operator `select_op` is a functor or lambda that should define the condition for selecting an element. 
 * If `select_op(element)` evaluates to `true`, the element will be placed in the "selected" portion of the output array.
 * 
 * @tparam T The type of the elements in the array (e.g., `int`, `float`).
 * @tparam Operator The type of the operator used for the selection condition.
 * @param d_in Pointer to the input device array to partition.
 * @param len The number of elements in the input array.
 * @param select_op The condition operator used to partition the array.
 * @param selectedLen Pointer to a variable that will be updated with the number of selected elements.
 * 
 * @return Pointer to the output device array that contains the selected elements.
 * 
 * @throws std::bad_alloc If memory allocation fails during the partitioning process.
 * @throws cudaError_t If any CUDA memory operations fail (e.g., memory copy, allocation, etc.).
 * @throws cub::CubError If the CUB partition operation fails.
 */
template <typename T, typename Operator>
T* devicePartition(T* d_in, size_t len, Operator select_op, size_t* selectedLen)
{
    if (!len) {
        *selectedLen = 0;
        return NULL;
    }

    T* d_out = CUDA_MALLOC(T, len);
    int* d_num_selected_out = CUDA_MALLOC(int, 1);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    CHECK_DEVICE(cub::DevicePartition::If(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        d_num_selected_out,
        len,
        select_op));

    // Allocate temporary storage
    d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes);

    // Run selection
    CHECK_DEVICE(cub::DevicePartition::If(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        d_num_selected_out,
        len,
        select_op));

    // TODO check this: we are copying an int to a size_t
    CHECK_DEVICE(cudaMemcpy(
        selectedLen,
        d_num_selected_out,
        sizeof(int),
        cudaMemcpyDeviceToHost));

    CUDA_FREE(d_temp_storage);
    CUDA_FREE(d_num_selected_out);

    return d_out;
}
