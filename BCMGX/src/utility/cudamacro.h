/**
 * @file
 */
 
#pragma once

#include "setting.h"

/**
 * @brief Macro to check the result of a CUDA API call.
 * 
 * This macro is used to check the result of a CUDA API call. If the call results in an error (i.e., not `cudaSuccess`), 
 * the macro prints an error message to `stderr` with details of the file, line number, and the error string, and then 
 * exits the program with a failure status.
 * 
 * This is a convenient tool for handling errors in CUDA code, allowing for easier debugging and faster identification of issues.
 * 
 * @param call The CUDA API call to check for errors.
 */
#define MY_CUDA_CHECK(call)                                               \
    {                                                                     \
        cudaError err = call;                                             \
        if (cudaSuccess != err) {                                         \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

/**
 * @brief Macro to check the last CUDA error with a custom error message.
 * 
 * This macro checks the last CUDA error using `cudaGetLastError()`. If the last CUDA error is not `cudaSuccess`, 
 * it prints a custom error message to `stderr` with details of the file, line number, the error string, and other 
 * relevant information such as the number of threads, blocks, and other provided parameters. It then exits the program 
 * with a failure status.
 * 
 * @param errorMessage The custom error message to be printed if an error is encountered.
 */
#define MY_CHECK_ERROR(errorMessage)                                                                                    \
    {                                                                                                                   \
        cudaError_t err = cudaGetLastError();                                                                           \
        if (cudaSuccess != err) {                                                                                       \
            fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s, nthreads=%d, nblocks=%d, n=%d, number=%d.\n", \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString(err), nThreads, nBlocks, n, number);               \
            exit(EXIT_FAILURE);                                                                                         \
        }                                                                                                               \
    }

/**
 * @brief Macro to check the result of a cuBLAS API call.
 * 
 * This macro checks the result of a cuBLAS API call. If the call does not return `CUBLAS_STATUS_SUCCESS`, 
 * it prints an error message to `stderr` with the file, line number, and the cuBLAS error code, then exits 
 * the program with a failure status.
 * 
 * @param call The cuBLAS API call to check for errors.
 */
#define MY_CUBLAS_CHECK(call)                                               \
    {                                                                       \
        cublasStatus_t err = call;                                          \
        if (CUBLAS_STATUS_SUCCESS != err) {                                 \
            fprintf(stderr, "Cublas error in file '%s' in line %i : %d.\n", \
                __FILE__, __LINE__, err);                                   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

extern itype* iPtemp1;
extern vtype* vPtemp1;
extern itype* iAtemp1;
extern vtype* vAtemp1;
extern itype* idevtemp1;
extern itype* idevtemp2;
extern vtype* vdevtemp1;
// ------------ TEST --------------
extern itype* dev_rcvprow_stat;
extern vtype* completedP_stat_val;
extern itype* completedP_stat_col;
extern itype* completedP_stat_row;
// ------------ AH glob -----------
extern itype* AH_glob_row;
extern itype* AH_glob_col;
extern vtype* AH_glob_val;
// --------------------------------
extern int* buffer_4_getmct;
extern int sizeof_buffer_4_getmct;
extern unsigned int* idx_4shrink;
extern bool alloced_idx;
// --------- cuCompactor ---------
extern int* glob_d_BlocksCount;
extern int* glob_d_BlocksOffset;
// --------------------------------
