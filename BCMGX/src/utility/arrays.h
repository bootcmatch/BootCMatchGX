#pragma once

#include "utility/utils.h"
#include <cuda.h>

template <typename T>
T* copyArrayToHost(T* arr, size_t len)
{
    if (len == 0) {
        return NULL;
    }

    T* hArr = (T*)Malloc(len * sizeof(T));
    CHECK_HOST(hArr);

    cudaError_t err = cudaMemcpy(
        hArr, arr, len * sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    return hArr;
}

template <typename T>
T* copyArrayToDevice(T* arr, size_t len)
{
    if (len == 0) {
        return NULL;
    }

    T* dArr = NULL;
    cudaError_t err = cudaMalloc(&dArr, len * sizeof(T));
    CHECK_DEVICE(err);

    err = cudaMemcpy(
        dArr, arr, len * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);

    return dArr;
}

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
        free(hArr);
    }
}

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
        CHECK_DEVICE(cudaMalloc(
            &concatenated,
            concatenatedSize * sizeof(T)));
    } else {
        concatenated = (T*)Malloc(concatenatedSize * sizeof(T));
        CHECK_HOST(concatenated);
    }

    if (len1) {
        CHECK_DEVICE(cudaMemcpy(
            concatenated,
            arr1,
            len1 * sizeof(T),
            getMemcpyKind(retOnDevice, isOnDevice1)));
    }

    if (len2) {
        // printf("concatenated + len1: %x, arr2: %x, len2: %ld, memcpyKind: %d\n", concatenated + len1, arr2, len2, getMemcpyKind(retOnDevice, isOnDevice2));

        CHECK_DEVICE(cudaMemcpy(
            concatenated + len1,
            arr2,
            len2 * sizeof(T),
            getMemcpyKind(retOnDevice, isOnDevice2)));
    }

    return concatenated;
}
