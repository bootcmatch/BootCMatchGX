/**
 * @file cuda_error_check.h
 * 
 * @brief Provides macros and functions for error checking in CUDA API calls and kernel launches.
 * 
 * These macros and functions help ensure the correctness of CUDA operations by checking for errors
 * after each CUDA API call and kernel launch. They print detailed error messages and terminate the
 * program if any error occurs.
 * 
 * The error checking mechanism is enabled by defining `CUDA_ERROR_CHECK`.
 */
#pragma once

#include <stdio.h>
#include <stdlib.h>

/**
 * @def CUDA_ERROR_CHECK
 * 
 * @brief Macro to enable or disable error checking globally in CUDA code.
 * 
 * When defined, this macro activates error checking after CUDA API calls and kernel launches.
 * It helps ensure that no errors occur during the execution of CUDA functions.
 */
#define CUDA_ERROR_CHECK

/**
 * @def CUDASAFECALL
 * 
 * @brief Macro for checking errors after a CUDA API call.
 * 
 * This macro calls the `__cudaSafeCall` function to check for errors after a CUDA API call.
 * It automatically provides the current file name and line number for detailed error reporting.
 * 
 * @param err The error code returned by the CUDA function.
 */
#define CUDASAFECALL(err) __cudaSafeCall(err, __FILE__, __LINE__)

/**
 * @def CUDACHECKERROR
 * 
 * @brief Macro for checking errors after a CUDA kernel launch or sequence of operations.
 * 
 * This macro calls the `__cudaCheckError` function to check for errors after kernel launches or
 * CUDA operations. It also performs synchronization to detect errors that occur during kernel
 * execution.
 */
#define CUDACHECKERROR() __cudaCheckError(__FILE__, __LINE__)

/**
 * @brief Checks the result of a CUDA API call and exits if an error occurred.
 * 
 * This function checks whether the given `err` code is equal to `cudaSuccess`. If the error
 * code is anything other than `cudaSuccess`, it prints an error message with the file, line
 * number, and error string, and exits the program.
 * 
 * @param err The error code returned by the CUDA API call.
 * @param file The source file in which the error occurred.
 * @param line The line number in the source file where the error occurred.
 */
inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));

        fprintf(stdout, "cudaSafeCall() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}

/**
 * @brief Checks the last CUDA error and exits if an error occurred.
 * 
 * This function checks for errors after a kernel launch or a series of CUDA operations.
 * It retrieves the last error with `cudaGetLastError()` and performs a device synchronization
 * check with `cudaDeviceSynchronize()` to catch errors that may not be immediately visible.
 * 
 * @param file The source file in which the error occurred.
 * @param line The line number in the source file where the error occurred.
 */
inline void __cudaCheckError(const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));

        fprintf(stdout, "cudaCheckError() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
        exit(-1);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));

        fprintf(stdout, "cudaCheckError() with sync failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));

        exit(-1);
    }
#endif

    return;
}
