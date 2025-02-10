/**
 * @file
 */
#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "setting.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <float.h>
#include <fstream>
#include <iostream>

#define DEFAULT_STREAM 0
#define WARP_SIZE 32
#define FULL_WARP 32

#define MINI_WARP_THRESHOLD_2 3
#define MINI_WARP_THRESHOLD_4 6
#define MINI_WARP_THRESHOLD_8 12
#define MINI_WARP_THRESHOLD_16 24

/**
 * @brief Converts a cuBLAS status code to a human-readable string.
 * 
 * This function takes a `cublasStatus_t` status code and returns a string 
 * describing the corresponding cuBLAS error or success status.
 * 
 * @param status The cuBLAS status code to convert.
 * @return A constant character pointer to the corresponding status description.
 */
const char* cublasGetStatusString(cublasStatus_t status);

/**
 * @brief Checks the return status of a cuBLAS function and handles errors.
 * 
 * This function evaluates the status code returned by a cuBLAS function.
 * If the status is not `CUBLAS_STATUS_SUCCESS`, it prints an error message
 * and terminates the program.
 * 
 * @param err The cuBLAS status code of type `cublasStatus_t` to check.
 */
void CHECK_CUBLAS(cublasStatus_t err);

/**
 * @namespace Eval
 * @brief Namespace containing utility functions for evaluation and debugging.
 */
namespace Eval {

/**
 * @brief Prints metadata information in a standardized format.
 * 
 * This function prints metadata with a specific format, prefixing it with `#META`.
 * The metadata includes a name, a value, and its type (integer or floating-point).
 * 
 * @param name The name of the metadata.
 * @param value The numerical value associated with the metadata.
 * @param type The type of the value:
 *             - `0` for integer (value is cast to int).
 *             - `1` for floating-point (printed in scientific notation).
 */
void printMetaData(const char* name, double value, int type);

}

/**
 * @brief Macro to check the result of a CUDA function call on the device.
 * 
 * This macro checks the return status of a CUDA function executed on the device.
 * If the status indicates an error, it prints the error message, the line, 
 * and the file where the error occurred, then terminates the program.
 * 
 * @param X The CUDA function call whose result is to be checked.
 */
#define CHECK_DEVICE(X)                                                                                   \
    {                                                                                                     \
        cudaError_t status = X;                                                                           \
        if (status != cudaSuccess) {                                                                      \
            const char* err_str = cudaGetErrorString(status);                                             \
            fprintf(stderr, "[ERROR DEVICE] :\n\t%s; LINE: %d; FILE: %s\n", err_str, __LINE__, __FILE__); \
            exit(1);                                                                                      \
        }                                                                                                 \
    }

/**
 * @brief Macro to check if a pointer is NULL on the host.
 * 
 * This macro checks if a pointer is NULL in host memory. If the pointer is NULL,
 * it prints an error message including the line number and the file where the error occurred,
 * then terminates the program.
 * 
 * @param X The pointer to check for NULL.
 */
#define CHECK_HOST(X)                                                                       \
    {                                                                                       \
        if (X == NULL) {                                                                    \
            fprintf(stderr, "[ERROR HOST] :\n\t LINE: %d; FILE: %s\n", __LINE__, __FILE__); \
            exit(1);                                                                        \
        }                                                                                   \
    }

/**
 * @brief Computes the bitmask corresponding to a specific warp ID for a given size.
 * 
 * This device function generates a bitmask based on the warp ID and the size of the warp (`m_size`).
 * For `m_size == 32`, it returns a full warp mask (i.e., `FULL_MASK`).
 * Otherwise, it calculates a bitmask with the appropriate number of bits shifted according to the warp ID.
 * 
 * @param m_size The size of the warp (in terms of the number of threads, typically 32).
 * @param m_id The ID of the warp for which the bitmask is being calculated.
 * @return The bitmask for the given warp ID.
 * 
 * @note This function assumes that the warp size is either 32 or less.
 */
__device__ __inline__ unsigned int getMaskByWarpID(unsigned int m_size, unsigned int m_id)
{
    if (m_size == 32) {
        return FULL_MASK;
    }
    unsigned int m = (1 << (m_size)) - 1;
    return (m << (m_size * m_id));
}

/**
 * @brief Structure representing CUDA grid and block dimensions.
 * 
 * This struct holds the grid (`g`) and block (`b`) dimensions for a CUDA kernel launch.
 */
struct GridBlock {
    dim3 g; ///< Grid dimensions.
    dim3 b; ///< Block dimensions.
};

/**
 * @brief Computes 1D grid and block configurations for a CUDA kernel launch.
 * 
 * This function calculates the optimal 1D grid (`g`) and block (`b`) sizes based on 
 * the given problem size (`n`), desired block size, and warp aggregation settings.
 * 
 * @param n The total number of elements to process.
 * @param block_size The number of threads per block.
 * @param is_warp_agg Boolean flag indicating whether warp aggregation is used.
 * @param MINI_WARP_SIZE The size of a mini warp when warp aggregation is enabled.
 * @return A `GridBlock` structure containing the computed grid (`g`) and block (`b`) sizes.
 * 
 * @note If `n` is zero, the function returns a grid and block size of zero.
 */
GridBlock gb1d(const unsigned n, const unsigned block_size, const bool is_warp_agg = false, int MINI_WARP_SIZE = 32);

/**
 * @brief Computes the optimal grid and block sizes for a CUDA kernel launch.
 * 
 * This function determines the appropriate grid (`nb`) and block (`nt`) sizes based 
 * on the desired number of threads. If the desired number of threads exceeds 
 * `MAX_THREADS`, it adjusts the grid and block sizes accordingly.
 * 
 * @param desiredThreads The total number of desired threads.
 * @param file The filename where this function is called (for error reporting).
 * @param line The line number where this function is called (for error reporting).
 * @return A `GridBlock` structure containing the computed grid (`g`) and block (`b`) sizes.
 * 
 * @note If the computed block size or grid size is zero, the function prints an error message and exits.
 */
GridBlock _getKernelParams(int desiredThreads, const char* file, int line);

/**
 * @brief Wrapper macro to call `_getKernelParams` with file and line information.
 * 
 * This macro simplifies calling `_getKernelParams` by automatically passing 
 * the current file name (`__FILE__`) and line number (`__LINE__`) for error reporting.
 * 
 * @param desiredThreads The total number of desired threads for kernel execution.
 * @return A `GridBlock` structure containing the computed grid (`g`) and block (`b`) sizes.
 * 
 * @see _getKernelParams
 */
#define getKernelParams(desiredThreads) _getKernelParams(desiredThreads, __FILE__, __LINE__)

/**
 * @brief Determines the appropriate cudaMemcpyKind based on source and destination locations.
 * 
 * This function returns the correct `cudaMemcpyKind` based on whether the source
 * and destination are located on the device (GPU) or the host (CPU).
 * 
 * @param dstOnDevice Boolean indicating if the destination is on the device (GPU).
 * @param srcOnDevice Boolean indicating if the source is on the device (GPU).
 * @return The corresponding `cudaMemcpyKind` value:
 *         - `cudaMemcpyDeviceToDevice` if both source and destination are on the GPU.
 *         - `cudaMemcpyHostToDevice` if the source is on the host and the destination is on the GPU.
 *         - `cudaMemcpyDeviceToHost` if the source is on the GPU and the destination is on the host.
 *         - `cudaMemcpyHostToHost` if both source and destination are on the CPU.
 */
cudaMemcpyKind getMemcpyKind(bool dstOnDevice, bool srcOnDevice);

/**
 * @brief Macro to check if a floating-point number is approximately zero.
 * 
 * This macro compares the absolute value of a number to a very small threshold (1e-7)
 * to determine if it is effectively zero.
 * 
 * @param a The floating-point number to check.
 * @return True if the absolute value of `a` is less than 1e-7, otherwise false.
 */
#define IS_ZERO(a) (fabs(a) < 0.0000001)

/**
 * @brief Macro to compute the minimum of two values.
 * 
 * This macro evaluates the two inputs and returns the smaller one.
 * 
 * @param x First value.
 * @param y Second value.
 * @return The minimum of `x` and `y`.
 */
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/**
 * @brief Macro to compute the maximum of two values.
 * 
 * This macro evaluates the two inputs and returns the larger one.
 * 
 * @param x First value.
 * @param y Second value.
 * @return The maximum of `x` and `y`.
 */
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
