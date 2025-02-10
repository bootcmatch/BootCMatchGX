/**
 * @file
 */
#pragma once

#include "utility/utils.h"

/**
 * @struct handles
 * @brief Structure to hold CUDA stream and cuBLAS handle resources.
 *
 * This structure contains handles for multiple CUDA streams and a cuBLAS handle. These resources are used
 * for asynchronous operations and matrix operations within the CUDA environment. The `init` function initializes
 * these resources, and the `free` function is used to properly clean up and release them.
 */
struct handles {
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;
    cudaStream_t stream4;
    cudaStream_t stream_free;

    cublasHandle_t cublas_h;
};

/**
 * @namespace Handles
 * @brief Namespace for managing CUDA streams and cuBLAS handle resources.
 * 
 * The `Handles` namespace encapsulates functions and structures related to the initialization, 
 * management, and cleanup of CUDA streams and cuBLAS handles. It provides a convenient way to 
 * initialize and free resources required for matrix operations and asynchronous tasks in CUDA.
 * 
 * The `init` function within the namespace creates and initializes a set of CUDA streams and 
 * a cuBLAS handle. The `free` function releases these resources when they are no longer needed.
 */
namespace Handles {

/**
 * @brief Initializes CUDA streams and cuBLAS handle.
 * 
 * This function allocates and initializes a `handles` structure, creates multiple CUDA streams, and initializes
 * a cuBLAS handle for matrix operations. These resources are necessary for performing asynchronous tasks
 * and efficient matrix computations in CUDA.
 * 
 * @return A pointer to a newly allocated and initialized `handles` structure.
 * 
 * @throws std::bad_alloc If memory allocation for the `handles` structure fails.
 * @throws cudaError_t If any CUDA or cuBLAS operation fails.
 */
handles* init();

/**
 * @brief Frees CUDA streams and cuBLAS handle resources.
 * 
 * This function releases the resources allocated for the CUDA streams and cuBLAS handle in a `handles` structure.
 * It is important to call this function when the streams and cuBLAS handle are no longer needed to avoid memory leaks.
 * 
 * @param h The `handles` structure containing the CUDA streams and cuBLAS handle to be freed.
 * 
 * @throws cudaError_t If any CUDA operation fails during stream destruction.
 * @throws cublasStatus_t If the cuBLAS handle destruction fails.
 */
void free(handles* h);
}
