/**
 * @file
 */
#pragma once

#include <cuda.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @def DEFAULT_INIT_MEMORY
 * 
 * A macro that defines whether to initialize memory by default. Set to `false` to disable
 * memory initialization unless specified otherwise.
 */
#define DEFAULT_INIT_MEMORY false

/**
 * @def PRINT_INFO_ALLOC
 * 
 * A macro to control whether memory allocation info should be printed during the allocation and deallocation
 * process. Set to `1` to enable, or `0` to disable.
 */
#define PRINT_INFO_ALLOC 0
// #define PRINT_INFO_ALLOC 1

/**
 * @brief Allocates memory on the device (GPU) for an array of type T.
 * 
 * This function uses `cudaMalloc` to allocate memory on the GPU. If the allocation is successful, it optionally
 * initializes the allocated memory using `cudaMemset` (based on the `initMemory` flag). The memory is allocated for
 * `nItems` elements of type `T`. If the allocation fails, the program prints an error message and exits.
 * 
 * @param file The file where the allocation was called.
 * @param line The line number where the allocation was called.
 * @param nItems The number of elements of type `T` to allocate.
 * @param initMemory Whether to initialize the allocated memory to zero (defaults to `DEFAULT_INIT_MEMORY`).
 * 
 * @return A pointer to the allocated memory on the GPU.
 */
template <typename T>
inline T* myCudaMalloc(const char* file, const int& line, size_t nItems, bool initMemory = DEFAULT_INIT_MEMORY)
{
    T* ptr = nullptr;
    if (nItems) {
        // cudaMalloc_CNT;
        cudaError_t err = cudaMalloc(&ptr, nItems * sizeof(T));
        if (err != cudaSuccess) {
            printf("Error allocating memory at %s:%d.\n", file, line);
            exit(1);
        }
        if (initMemory) {
            err = cudaMemset(ptr, 0, nItems * sizeof(T));
            if (err != cudaSuccess) {
                printf("Error initializing memory (cudaMemset) at %s:%d.\n", file, line);
                exit(1);
            }
        }
    } else {
        printf("Error allocating memory at %s:%d: nItems == 0.\n", file, line);
        exit(1);
    }
#if PRINT_INFO_ALLOC
    fprintf(stdout, "[PTR MALLOC] CUDA_MALLOC of %p at %s:%d.\n", ptr, file, line);
#endif
    return ptr;
}

/**
 * @brief Macro to simplify CUDA memory allocation.
 * 
 * A shorthand macro for calling `myCudaMalloc` with the current file and line information automatically passed.
 * 
 * @param T The type of the elements to allocate.
 * @param ... The number of elements to allocate and optional flags for memory initialization.
 */
#define CUDA_MALLOC(T, ...) myCudaMalloc<T>(__FILE__, __LINE__, __VA_ARGS__);

/**
 * @brief Allocates memory on the device (GPU) for a given number of bytes.
 * 
 * This function uses `cudaMalloc` to allocate a specified number of bytes on the GPU. It optionally initializes
 * the allocated memory using `cudaMemset` (based on the `initMemory` flag). If the allocation fails, the program
 * prints an error message and exits.
 * 
 * @param file The file where the allocation was called.
 * @param line The line number where the allocation was called.
 * @param nBytes The number of bytes to allocate.
 * @param initMemory Whether to initialize the allocated memory to zero (defaults to `DEFAULT_INIT_MEMORY`).
 * 
 * @return A pointer to the allocated memory on the GPU.
 */
template <typename T>
inline T* myCudaMallocBytes(const char* file, const int& line, size_t nBytes, bool initMemory = DEFAULT_INIT_MEMORY)
{
    T* ptr = nullptr;
    if (nBytes) {
        // cudaMalloc_CNT;
        cudaError_t err = cudaMalloc(&ptr, nBytes);
        if (err != cudaSuccess) {
            printf("Error allocating memory at %s:%d.\n", file, line);
            exit(1);
        }
        if (initMemory) {
            err = cudaMemset(ptr, 0, nBytes);
            if (err != cudaSuccess) {
                printf("Error initializing memory (cudaMemset) at %s:%d.\n", file, line);
                exit(1);
            }
        }
    } else {
        printf("Error allocating memory at %s:%d: nBytes == 0.\n", file, line);
        exit(1);
    }
#if PRINT_INFO_ALLOC
    fprintf(stdout, "[PTR MALLOC] CUDA_MALLOC_BYTES of %p at %s:%d.\n", ptr, file, line);
#endif
    return ptr;
}

/**
 * @brief Macro to simplify CUDA memory allocation by bytes.
 * 
 * A shorthand macro for calling `myCudaMallocBytes` with the current file and line information automatically passed.
 * 
 * @param T The type of the elements to allocate.
 * @param ... The number of bytes to allocate and optional flags for memory initialization.
 */
#define CUDA_MALLOC_BYTES(T, ...) myCudaMallocBytes<T>(__FILE__, __LINE__, __VA_ARGS__);

/**
 * @brief Allocates memory on the host (CPU) for an array of type T.
 * 
 * This function uses `cudaMallocHost` to allocate pinned memory on the host. If the allocation is successful,
 * it optionally initializes the allocated memory using `memset` (based on the `initMemory` flag). The memory is
 * allocated for `nItems` elements of type `T`. If the allocation fails, the program prints an error message and exits.
 * 
 * @param file The file where the allocation was called.
 * @param line The line number where the allocation was called.
 * @param nItems The number of elements of type `T` to allocate.
 * @param initMemory Whether to initialize the allocated memory to zero (defaults to `DEFAULT_INIT_MEMORY`).
 * 
 * @return A pointer to the allocated memory on the host.
 */
template <typename T>
inline T* myCudaMallocHost(const char* file, const int& line, size_t nItems, bool initMemory = DEFAULT_INIT_MEMORY)
{
    T* ptr = nullptr;
    if (nItems) {
        // cudaMalloc_CNT;
        cudaError_t err = cudaMallocHost(&ptr, nItems * sizeof(T));
        if (err != cudaSuccess) {
            printf("Error allocating memory at %s:%d.\n", file, line);
            exit(1);
        }
        if (initMemory) {
            memset(ptr, 0, nItems * sizeof(T));
        }
    } else {
        printf("Error allocating memory at %s:%d: nItems == 0.\n", file, line);
        exit(1);
    }
#if PRINT_INFO_ALLOC
    fprintf(stdout, "[PTR MALLOC] CUDA_MALLOC_HOST of %p at %s:%d.\n", ptr, file, line);
#endif
    return ptr;
}

/**
 * @brief Macro to simplify host memory allocation.
 * 
 * A shorthand macro for calling `myCudaMallocHost` with the current file and line information automatically passed.
 * 
 * @param T The type of the elements to allocate.
 * @param ... The number of elements to allocate and optional flags for memory initialization.
 */
#define CUDA_MALLOC_HOST(T, ...) myCudaMallocHost<T>(__FILE__, __LINE__, __VA_ARGS__);

/**
 * @brief Frees memory on the device (GPU).
 * 
 * This function frees memory previously allocated on the GPU with `cudaFree`. If the pointer is `nullptr`, it does nothing.
 * It also prints allocation information for debugging purposes (if `PRINT_INFO_ALLOC` is enabled).
 * 
 * @param file The file where the deallocation was called.
 * @param line The line number where the deallocation was called.
 * @param ptr The pointer to the memory to free.
 */
template <typename T>
inline void myCudaFree(const char* file, const int& line, T*& ptr)
{
#if PRINT_INFO_ALLOC
    fprintf(stdout, "[PTR FREE] CUDA_FREE of %p at %s:%d.\n", ptr, file, line);
#endif
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        // cudaError_t err = cudaFreeAsync(ptr);
        if (err != cudaSuccess) {
            printf("Error releasing memory at %s:%d. (%s)\n", file, line, cudaGetErrorString(err));
            exit(1);
        }
    }
    ptr = nullptr;
}

/**
 * @brief Macro to simplify CUDA memory deallocation.
 * 
 * A shorthand macro for calling `myCudaFree` with the current file and line information automatically passed.
 * 
 * @param ptr The pointer to the memory to free.
 */
#define CUDA_FREE(ptr) myCudaFree(__FILE__, __LINE__, ptr);

/**
 * @brief Frees memory on the host (CPU).
 * 
 * This function frees memory previously allocated on the host with `cudaFreeHost`. If the pointer is `nullptr`, it does nothing.
 * It also prints allocation information for debugging purposes (if `PRINT_INFO_ALLOC` is enabled).
 * 
 * @param file The file where the deallocation was called.
 * @param line The line number where the deallocation was called.
 * @param ptr The pointer to the memory to free.
 */
template <typename T>
inline void myCudaFreeHost(const char* file, const int& line, T*& ptr)
{
#if PRINT_INFO_ALLOC
    fprintf(stdout, "[PTR FREE] CUDA_FREE_HOST of %p at %s:%d.\n", ptr, file, line);
#endif
    if (ptr) {
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess) {
            printf("Error releasing memory at %s:%d.\n", file, line);
            exit(1);
        }
    }
    ptr = nullptr;
}

#define CUDA_FREE_HOST(ptr) myCudaFreeHost(__FILE__, __LINE__, ptr);

/**
 * @brief Frees memory on the device (GPU) asynchronously.
 * 
 * This function frees memory previously allocated on the GPU with `cudaFreeAsync`. If the pointer is `nullptr`, it does nothing.
 * It also prints allocation information for debugging purposes (if `PRINT_INFO_ALLOC` is enabled).
 * 
 * @param file The file where the deallocation was called.
 * @param line The line number where the deallocation was called.
 * @param ptr The pointer to the memory to free.
 */
template <typename T>
inline void myCudaFreeAsync(const char* file, const int& line, T*& ptr)
{
#if PRINT_INFO_ALLOC
    fprintf(stdout, "[PTR FREE] CUDA_FREE_ASYNC of %p at %s:%d.\n", ptr, file, line);
#endif
    if (ptr) {
        cudaError_t err = cudaFreeAsync(ptr);
        if (err != cudaSuccess) {
            printf("Error releasing memory at %s:%d.\n", file, line);
            exit(1);
        }
    }
    ptr = nullptr;
}

/**
 * @brief Macro to simplify asynchronous CUDA memory deallocation.
 * 
 * A shorthand macro for calling `myCudaFreeAsync` with the current file and line information automatically passed.
 * 
 * @param ptr The pointer to the memory to free.
 */
#define CUDA_FREE_ASYNC(ptr) myCudaFreeAsync(__FILE__, __LINE__, ptr);

/**
 * @brief Frees memory on the device (GPU) asynchronously, with a stream.
 * 
 * This function frees memory previously allocated on the GPU with `cudaFreeAsync`, but it allows specifying a CUDA stream
 * for asynchronous execution. If the pointer is `nullptr`, it does nothing.
 * 
 * @param file The file where the deallocation was called.
 * @param line The line number where the deallocation was called.
 * @param ptr The pointer to the memory to free.
 * @param stream The CUDA stream to use for asynchronous deallocation.
 */
template <typename T>
inline void myCudaFreeAsyncStream(const char* file, const int& line, T*& ptr, cudaStream_t& stream)
{
#if PRINT_INFO_ALLOC
    fprintf(stdout, "[PTR FREE] CUDA_FREE_ASYNC_STREAM of %p at %s:%d.\n", ptr, file, line);
#endif
    if (ptr) {
        cudaError_t err = cudaFreeAsync(ptr, stream);
        if (err != cudaSuccess) {
            printf("Error releasing memory at %s:%d.\n", file, line);
            exit(1);
        }
    }
    ptr = nullptr;
}

/**
 * @brief Macro to simplify asynchronous CUDA memory deallocation with a stream.
 * 
 * A shorthand macro for calling `myCudaFreeAsyncStream` with the current file and line information automatically passed.
 * 
 * @param ptr The pointer to the memory to free.
 * @param stream The CUDA stream to use for asynchronous deallocation.
 */
#define CUDA_FREE_ASYNC_STREAM(ptr, stream) myCudaFreeAsyncStream(__FILE__, __LINE__, ptr, stream);

/**
 * @brief Allocates memory on the host (CPU).
 * 
 * This function uses `malloc` to allocate memory on the CPU. If the allocation is successful, it optionally
 * initializes the allocated memory using `memset` (based on the `initMemory` flag). The memory is allocated for
 * `nItems` elements of type `T`. If the allocation fails, the program prints an error message and exits.
 * 
 * @param file The file where the allocation was called.
 * @param line The line number where the allocation was called.
 * @param nItems The number of elements of type `T` to allocate.
 * @param initMemory Whether to initialize the allocated memory to zero (defaults to `DEFAULT_INIT_MEMORY`).
 * 
 * @return A pointer to the allocated memory on the host.
 */
template <typename T>
inline T* myMalloc(const char* file, const int& line, size_t nItems, bool initMemory = DEFAULT_INIT_MEMORY)
{
    T* ptr = nullptr;
    if (nItems) {
        ptr = (T*)malloc(nItems * sizeof(T));
        if (ptr == nullptr) {
            printf("Error allocating memory at %s:%d.\n", file, line);
            exit(1);
        }
        if (initMemory) {
            memset(ptr, 0, nItems * sizeof(T));
        }
    } else {
        printf("Error allocating memory at %s:%d: nItems == 0.\n", file, line);
        exit(1);
    }
    return ptr;
}

/**
 * @brief Macro to simplify host memory allocation.
 * 
 * A shorthand macro for calling `myMalloc` with the current file and line information automatically passed.
 * 
 * @param T The type of the elements to allocate.
 * @param ... The number of elements to allocate and optional flags for memory initialization.
 */
#define MALLOC(T, ...) myMalloc<T>(__FILE__, __LINE__, __VA_ARGS__);

/**
 * @brief Allocates a given number of bytes on the host (CPU).
 * 
 * This function uses `malloc` to allocate memory for a specific number of bytes on the CPU. It optionally initializes
 * the allocated memory using `memset` (based on the `initMemory` flag). If the allocation fails, the program prints
 * an error message and exits.
 * 
 * @param file The file where the allocation was called.
 * @param line The line number where the allocation was called.
 * @param nBytes The number of bytes to allocate.
 * @param initMemory Whether to initialize the allocated memory to zero (defaults to `DEFAULT_INIT_MEMORY`).
 * 
 * @return A pointer to the allocated memory on the host.
 */
template <typename T>
inline T* myMallocBytes(const char* file, const int& line, size_t nBytes, bool initMemory = DEFAULT_INIT_MEMORY)
{
    T* ptr = nullptr;
    if (nBytes) {
        ptr = (T*)malloc(nBytes);
        if (ptr == nullptr) {
            printf("Error allocating memory at %s:%d.\n", file, line);
            exit(1);
        }
        if (initMemory) {
            memset(ptr, 0, nBytes);
        }
    } else {
        printf("Error allocating memory at %s:%d: nItems == 0.\n", file, line);
        exit(1);
    }
    return ptr;
}

/**
 * @brief Macro to simplify host memory allocation by bytes.
 * 
 * A shorthand macro for calling `myMallocBytes` with the current file and line information automatically passed.
 * 
 * @param T The type of the elements to allocate.
 * @param ... The number of bytes to allocate and optional flags for memory initialization.
 */
#define MALLOC_BYTES(T, ...) myMallocBytes<T>(__FILE__, __LINE__, __VA_ARGS__);

/**
 * @brief Allocates memory on the host (CPU) and copies data from another pointer.
 * 
 * This function uses `malloc` to allocate memory on the CPU and copies the contents from an existing memory location
 * (`src`) into the new allocated memory. The memory is allocated for `nItems` elements of type `T`. If the allocation
 * fails, the program prints an error message and exits.
 * 
 * @param file The file where the allocation was called.
 * @param line The line number where the allocation was called.
 * @param nItems The number of elements of type `T` to allocate.
 * @param src The source pointer to copy data from.
 * 
 * @return A pointer to the allocated memory on the host with copied data.
 */
template <typename T>
inline T* myClone(const char* file, const int& line, size_t nItems, T* src)
{
    T* ptr = nullptr;
    if (nItems) {
        ptr = (T*)malloc(nItems * sizeof(T));
        if (ptr == nullptr) {
            printf("Error allocating memory at %s:%d.\n", file, line);
            exit(1);
        }
        memcpy(ptr, src, nItems * sizeof(T));
    } else {
        printf("Error allocating memory at %s:%d: nItems == 0.\n", file, line);
        exit(1);
    }
    return ptr;
}

/**
 * @brief Macro to simplify memory cloning.
 * 
 * A shorthand macro for calling `myClone` with the current file and line information automatically passed.
 * 
 * @param T The type of the elements to clone.
 * @param ... The number of elements to allocate and the source pointer.
 */
#define CLONE(T, ...) myClone<T>(__FILE__, __LINE__, __VA_ARGS__);

/**
 * @brief Reallocates memory on the host (CPU).
 * 
 * This function uses `realloc` to change the size of a previously allocated memory block. If the memory pointer is
 * not `nullptr`, the function reallocates it and ensures the memory is initialized to zero if requested.
 * 
 * @param file The file where the reallocation was called.
 * @param line The line number where the reallocation was called.
 * @param ptr The pointer to the memory block to reallocate.
 * @param nItemsOld The current number of items in the allocated block.
 * @param nItemsNew The new number of items after reallocation.
 * @param initMemory Whether to initialize the newly allocated memory to zero (defaults to `DEFAULT_INIT_MEMORY`).
 * 
 * @return A pointer to the newly allocated memory block.
 */
template <typename T>
inline T* myRealloc(const char* file, const int& line, T* ptr, size_t nItemsOld, size_t nItemsNew, bool initMemory = DEFAULT_INIT_MEMORY)
{
    if (!nItemsNew) {
        printf("Error allocating memory at %s:%d: nItemsNew == 0.\n", file, line);
        exit(1);
    }

    T* oldPtr = ptr;
    ptr = (T*)realloc(ptr, nItemsNew * sizeof(T));
    if (ptr == nullptr) {
        printf("Error allocating memory at %s:%d.\n", file, line);
        exit(1);
    }
    if (initMemory) {
        if (oldPtr != ptr) {
            memset(ptr, 0, nItemsNew * sizeof(T));
        } else if (nItemsNew - nItemsOld > 0) {
            memset(ptr + nItemsOld, 0, (nItemsNew - nItemsOld) * sizeof(T));
        }
    }

    return ptr;
}

/**
 * @brief Macro to simplify memory reallocation.
 * 
 * A shorthand macro for calling `myRealloc` with the current file and line information automatically passed.
 * 
 * @param T The type of the elements to reallocate.
 * @param ... The pointer to reallocate, old size, new size, and optional flags for memory initialization.
 */
#define REALLOC(T, ...) myRealloc<T>(__FILE__, __LINE__, __VA_ARGS__);

/**
 * @brief Frees memory on the host (CPU).
 * 
 * This function frees memory previously allocated with `malloc`. If the pointer is `nullptr`, it does nothing.
 * 
 * @param file The file where the deallocation was called.
 * @param line The line number where the deallocation was called.
 * @param ptr The pointer to the memory to free.
 */
template <typename T>
inline void myFree(const char* file, const int& line, T*& ptr)
{
    if (ptr) {
        free(ptr);
    }
    ptr = nullptr;
}

/**
 * @brief Macro to simplify host memory deallocation.
 * 
 * A shorthand macro for calling `myFree` with the current file and line information automatically passed.
 * 
 * @param ptr The pointer to the memory to free.
 */
#define FREE(ptr) myFree(__FILE__, __LINE__, ptr);
