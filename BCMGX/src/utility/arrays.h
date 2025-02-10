/**
 * @file
 * @brief Utility functions for memory management and debugging of arrays on CPU and GPU.
 */
#pragma once

#include "utility/memory.h"
#include "utility/utils.h"
#include <cuda.h>

/**
 * @brief Copies an array from device (GPU) memory to host (CPU) memory.
 * 
 * @tparam T The type of elements in the array.
 * @param arr Pointer to the array in device memory.
 * @param len Number of elements in the array.
 * @return A pointer to the copied array in host memory.
 */
template <typename T>
T* copyArrayToHost(T* arr, size_t len)
{
    if (len == 0) {
        return NULL;
    }

    T* hArr = MALLOC(T, len);

    cudaError_t err = cudaMemcpy(
        hArr, arr, len * sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    return hArr;
}

/**
 * @brief Copies an array from host (CPU) memory to device (GPU) memory.
 * 
 * @tparam T The type of elements in the array.
 * @param arr Pointer to the array in host memory.
 * @param len Number of elements in the array.
 * @return A pointer to the copied array in device memory.
 */
template <typename T>
T* copyArrayToDevice(T* arr, size_t len)
{
    if (len == 0) {
        return NULL;
    }

    T* dArr = CUDA_MALLOC(T, len);
    cudaError_t err = cudaMemcpy(
        dArr, arr, len * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);

    return dArr;
}

/**
 * @brief Prints the contents of an array to a file for debugging.
 * 
 * @tparam T The type of elements in the array.
 * @param fmt The format string for printing elements.
 * @param arr Pointer to the array.
 * @param len Number of elements in the array.
 * @param isOnDevice True if the array is on the device (GPU), false if on the host (CPU).
 * @param f File pointer to which the output is written.
 */
template <typename T>
void debugArray(const char* fmt, T* arr, size_t len, bool isOnDevice, FILE* f)
{
    T* hArr = arr;
    if (isOnDevice && arr != NULL && len > 0) {
        hArr = copyArrayToHost(arr, len);
    }

    for (int i = 0; hArr != NULL && i < len; i++) {
        fprintf(f, fmt, i, hArr[i]);
    }

    if (isOnDevice && hArr != NULL) {
        FREE(hArr);
    }
}

/**
 * @brief Concatenates two arrays, which may be located in host or device memory.
 * 
 * This function merges two arrays and returns a new array, either in device (GPU) memory 
 * or host (CPU) memory, based on the `retOnDevice` flag.
 * 
 * @tparam T The type of elements in the arrays.
 * @param arr1 Pointer to the first array.
 * @param len1 Number of elements in the first array.
 * @param isOnDevice1 True if `arr1` is on the device (GPU), false if on the host (CPU).
 * @param arr2 Pointer to the second array.
 * @param len2 Number of elements in the second array.
 * @param isOnDevice2 True if `arr2` is on the device (GPU), false if on the host (CPU).
 * @param retOnDevice True if the concatenated array should be returned on the device (GPU), 
 *                    false if it should be returned on the host (CPU).
 * @return A pointer to the concatenated array, allocated either on the host or device.
 */
template <typename T>
T* concatArrays(T* arr1, size_t len1, bool isOnDevice1,
    T* arr2, size_t len2, bool isOnDevice2,
    bool retOnDevice)
{

    /*
    printf("concatArrays(arr1 = %x, len1 = %ld, isOnDevice1 = %d,"
            "arr2 = %x, len2 = %ld, isOnDevice2 = %d,"
            "retOnDevice = %d)\n",
            arr1, len1, isOnDevice1,
            arr2, len2, isOnDevice2,
            retOnDevice);*/

    T* concatenated = NULL;
    size_t concatenatedSize = len1 + len2;
    if (!concatenatedSize) {
        return concatenated;
    }

    if (retOnDevice) {
        concatenated = CUDA_MALLOC(T, concatenatedSize);
    } else {
        concatenated = MALLOC(T, concatenatedSize);
        CHECK_HOST(concatenated);
    }

    if (len1) {
        if (!retOnDevice && !isOnDevice1) {
            memcpy(concatenated, arr1, len1 * sizeof(T));
        } else {
            CHECK_DEVICE(cudaMemcpy(
                concatenated,
                arr1,
                len1 * sizeof(T),
                getMemcpyKind(retOnDevice, isOnDevice1)));
        }
    }

    if (len2) {
        // printf("concatenated + len1: %x, arr2: %x, len2: %ld, memcpyKind: %d\n", concatenated + len1, arr2, len2, getMemcpyKind(retOnDevice, isOnDevice2));
        if (!retOnDevice && !isOnDevice2) {
            memcpy(concatenated + len1, arr2, len2 * sizeof(T));
        } else {
            CHECK_DEVICE(cudaMemcpy(
                concatenated + len1,
                arr2,
                len2 * sizeof(T),
                getMemcpyKind(retOnDevice, isOnDevice2)));
        }
    }

    return concatenated;
}
