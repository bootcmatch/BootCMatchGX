#pragma once

#include "utility/utils.h"
#include <cub/cub.cuh>
#include <cuda.h>

template <typename T, typename Operator>
T* devicePartition(T* d_in, size_t len, Operator select_op, size_t* selectedLen)
{
    if (!len) {
        *selectedLen = 0;
        return NULL;
    }

    T* d_out = NULL;
    CHECK_DEVICE(cudaMalloc(&d_out, len * sizeof(T)));

    int* d_num_selected_out = NULL;
    CHECK_DEVICE(cudaMalloc(&d_num_selected_out, sizeof(int)));

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
    CHECK_DEVICE(cudaMalloc(&d_temp_storage, temp_storage_bytes));

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

    CHECK_DEVICE(cudaFree(d_temp_storage));
    CHECK_DEVICE(cudaFree(d_num_selected_out));

    return d_out;
}
