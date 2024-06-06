#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <float.h>
#include <fstream>
#include <iostream>
// #include <mpi.h>
#include "setting.h"

#define DEFAULT_STREAM 0
#define WARP_SIZE 32
#define FULL_WARP 32

// #ifndef FULL_MASK
//   #define FULL_MASK 0xFFFFFFFF
// #endif

#define MINI_WARP_THRESHOLD_2 3
#define MINI_WARP_THRESHOLD_4 6
#define MINI_WARP_THRESHOLD_8 12
#define MINI_WARP_THRESHOLD_16 24

void* Malloc(size_t sz);

void* Realloc(void* ptr, size_t sz);

const char* cublasGetStatusString(cublasStatus_t status);
void CHECK_CUBLAS(cublasStatus_t err);

namespace Eval {
void printMetaData(const char* name, double value, int type);
}

#define CHECK_DEVICE(X)                                                                                   \
    {                                                                                                     \
        cudaError_t status = X;                                                                           \
        if (status != cudaSuccess) {                                                                      \
            const char* err_str = cudaGetErrorString(status);                                             \
            fprintf(stderr, "[ERROR DEVICE] :\n\t%s; LINE: %d; FILE: %s\n", err_str, __LINE__, __FILE__); \
            exit(1);                                                                                      \
        }                                                                                                 \
    }

#define CHECK_HOST(X)                                                                       \
    {                                                                                       \
        if (X == NULL) {                                                                    \
            fprintf(stderr, "[ERROR HOST] :\n\t LINE: %d; FILE: %s\n", __LINE__, __FILE__); \
            exit(1);                                                                        \
        }                                                                                   \
    }

template <typename T>
void Free(T*& ptr)
{
    if (ptr) {
        free(ptr);
        ptr = NULL;
    }
}

template <typename T>
void _CudaFree(T*& ptr, const char* file, int line)
{
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error in file %s at line %d\n", file, line);
        }
        CHECK_DEVICE(err);
        ptr = NULL;
    }
}

#define CudaFree(ptr) _CudaFree(ptr, __FILE__, __LINE__)

__device__ __inline__ unsigned int getMaskByWarpID(unsigned int m_size, unsigned int m_id)
{
    if (m_size == 32) {
        return FULL_MASK;
    }
    unsigned int m = (1 << (m_size)) - 1;
    return (m << (m_size * m_id));
}

struct GridBlock {
    dim3 g;
    dim3 b;
};

GridBlock gb1d(const unsigned n, const unsigned block_size, const bool is_warp_agg = false, int MINI_WARP_SIZE = 32);

GridBlock _getKernelParams(int desiredThreads, const char* file, int line);

#define getKernelParams(desiredThreads) _getKernelParams(desiredThreads, __FILE__, __LINE__)

cudaMemcpyKind getMemcpyKind(bool dstOnDevice, bool srcOnDevice);

#define IS_ZERO(a) (fabs(a) < 0.0000001)

#define MAX(a, b) (a > b ? a : b)
