/**
 * @file
 */
#pragma once

#include "utility/memory.h"
#include <cub/cub.cuh>

/**
 * @brief Functor that checks if a value is not equal to a given value.
 * 
 * This functor is used for filtering elements in an array that are not equal to a specified value. It is intended to be passed 
 * to the `deviceFilter` function, which will apply it to each element of a device array in parallel. If an element is not equal 
 * to the specified value, it will be included in the filtered output array.
 * 
 * @tparam T The type of the element to compare.
 */
template <typename T>
struct IsNotEqual {
    T value; ///< The value to compare elements against.

    /**
     * @brief Constructor that initializes the functor with the value to compare against.
     * 
     * @param value The value to compare elements with.
     */
    __host__ __device__ __forceinline__
    IsNotEqual(T value)
        : value(value)
    {
    }

    /**
     * @brief Compares the input element against the stored value.
     * 
     * This operator returns `true` if the input element `a` is not equal to the stored `value`, and `false` otherwise.
     * 
     * @param a The element to compare with the stored value.
     * 
     * @return `true` if `a` is not equal to `value`, otherwise `false`.
     */
    __host__ __device__ __forceinline__ bool operator()(const T& a) const
    {
        return a != value;
    }
};

/**
 * @brief Filters the elements of a device array based on a predicate function.
 * 
 * This function filters the elements of the input device array `d_arr` using the provided `filter` function. The filtered 
 * elements are stored in a new device array, and the number of filtered elements is returned in `filteredSize`. Only the elements 
 * that satisfy the predicate (i.e., for which `filter(element)` returns `true`) are included in the output array.
 * 
 * The function uses the CUB library's `DeviceSelect::If` to perform the filtering operation.
 * 
 * @tparam T The type of the elements in the array.
 * @tparam F The type of the filter function (a functor or lambda).
 * 
 * @param d_arr Pointer to the input device array.
 * @param len The number of elements in the array.
 * @param filter The filter function to be applied to each element in the array.
 * @param filteredSize Pointer to store the number of elements that satisfy the filter.
 * 
 * @return A pointer to the device array containing the filtered elements.
 * 
 * @throws cudaError_t If any CUDA operations fail (e.g., memory allocation or kernel execution).
 */
template <typename T, typename F>
T* deviceFilter(T* d_arr, size_t len, F filter, size_t* filteredSize)
{
    if (!len) {
        *filteredSize = 0;
        return NULL;
    }

    T* d_filteredArr = CUDA_MALLOC(T, len);

    int* d_filteredSize = CUDA_MALLOC(int, 1);

    // ---------------------------------------------------------------------
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::If(
        d_temp_storage,
        temp_storage_bytes,
        d_arr,
        d_filteredArr,
        d_filteredSize,
        len,
        filter);

    d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes);

    cub::DeviceSelect::If(
        d_temp_storage,
        temp_storage_bytes,
        d_arr,
        d_filteredArr,
        d_filteredSize,
        len,
        filter);
    // ---------------------------------------------------------------------

    int h_filteredSize = 0;
    CHECK_DEVICE(cudaMemcpy(
        (void*)&h_filteredSize,
        (void*)d_filteredSize,
        sizeof(int),
        cudaMemcpyDeviceToHost));

    CUDA_FREE(d_temp_storage);
    CUDA_FREE(d_filteredSize);

    *filteredSize = h_filteredSize;

    return d_filteredArr;
}
