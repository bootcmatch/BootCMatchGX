/**
 * @file
 * @brief This file defines structures and utility functions for handling vectors and vector collections,
 *        as well as functions to perform operations such as copying, filling, scaling, and performing basic
 *        linear algebra operations (e.g., dot product, norm, etc.) on vectors.
 * 
 * The code supports operations on vectors stored in device memory (GPU) and host memory. It also includes 
 * functionality for vector operations in a distributed context, with MPI support.
 */
#pragma once

#include "utility/mpi.h"
#include "utility/setting.h"
#include "utility/utils.h"
#include <curand.h>
#include <iostream>
#include <stdarg.h>
#include <stdio.h>

/**
 * @struct vector
 * @brief A templated structure representing a sparse vector.
 * 
 * This structure stores a sparse vector with non-zero values on either the host or device memory. It tracks
 * the number of non-zero entries and whether the vector resides in device memory.
 * 
 * @tparam T The data type of the vector elements (e.g., `float`, `int`, `double`).
 */
template <typename T>
struct vector {
    int n = 0; /**< The number of non-zero elements in the vector */
    bool on_the_device = false; /**< Flag indicating whether the vector is stored on the device */
    T* val = NULL; /**< Pointer to the array of non-zero values */
};

/**
 * @struct vectordh
 * @brief A structure to manage vector data in both host and device memory.
 * 
 * This structure is designed to handle vector data that can be accessed in both host and device memory
 * (dual-host). The structure contains two pointers to the vector's data: one for the host and one for the device.
 * 
 * @tparam T The data type of the vector elements.
 */
template <typename T>
struct vectordh {
    int n = 0; /**< The number of elements in the vector */
    T* val = NULL; /**< Pointer to the vector stored in host memory */
    T* val_ = NULL; /**< Pointer to the vector stored in device memory */
};

/**
 * @struct vectorCollection
 * @brief A structure to represent a collection of vectors.
 * 
 * This structure holds an array of pointers to individual vectors, enabling operations on multiple vectors at once.
 * 
 * @tparam T The data type of the vectors in the collection.
 */
template <typename T>
struct vectorCollection {
    unsigned int n = 0; /**< The number of vectors in the collection */
    vector<T>** val = NULL; /**< Pointer to an array of vectors */
};

/**
 * @namespace Vector
 * @brief A namespace containing utility functions for handling vector operations.
 * 
 * This namespace provides functions to initialize vectors, copy data between host and device, perform basic
 * vector operations such as scaling and dot products, and manage memory allocation and deallocation for vectors.
 */
namespace Vector {

/**
 * @brief Initializes a vector with the specified size and memory allocation behavior.
 * 
 * This function creates a vector of size `n` and optionally allocates memory for the vector on either
 * the host or device based on the `on_the_device` flag.
 * 
 * @tparam T The data type of the vector elements.
 * @param n The number of non-zero elements in the vector.
 * @param allocate_mem Flag indicating if memory should be allocated for the vector.
 * @param on_the_device Flag indicating whether to allocate the vector on the device or not.
 * @return A pointer to the initialized vector.
 */
template <typename T>
vector<T>* init(unsigned int n, bool allocate_mem, bool on_the_device);

/**
 * @brief Initializes a dual-host vector with the specified size.
 * 
 * This function creates a vector with the given size that has both host and device memory allocated.
 * 
 * @tparam T The data type of the vector elements.
 * @param n The number of elements in the vector.
 * @return A pointer to the initialized dual-host vector.
 */
template <typename T>
vectordh<T>* initdh(int n);

 /**
  * @brief Copies a vector from host to device memory.
  * 
  * This function copies the vector data from host memory to device memory.
  * 
  * @tparam T The data type of the vector elements.
  * @param v The vector to be copied to the device.
  */
template <typename T>
void copydhToD(vectordh<T>* v);

/**
 * @brief Copies a vector from device to host memory.
 * 
 * This function copies the vector data from device memory to host memory.
 * 
 * @tparam T The data type of the vector elements.
 * @param v The vector to be copied to the host.
 */
template <typename T>
void copydhToH(vectordh<T>* v);

/**
 * @brief Frees memory allocated for a dual-host vector.
 * 
 * This function frees the memory for a dual-host vector, including both host and device memory.
 * 
 * @tparam T The data type of the vector elements.
 * @param v The dual-host vector to be freed.
 */
template <typename T>
void freedh(vectordh<T>* v);

/**
 * @brief Fills a vector with a specific value.
 * 
 * This function sets all elements in a vector to the specified value.
 * 
 * @tparam T The data type of the vector elements.
 * @param v The vector to be filled with the value.
 * @param value The value to fill the vector with.
 */
template <typename T>
void fillWithValue(vector<T>* v, T value);

/**
 * @brief Fills a portion of a vector with a specific value.
 * 
 * This function sets a sub-section of a vector to the specified value, defined by the range of indices.
 * 
 * @tparam T The data type of the vector elements.
 * @param v The vector to be filled with the value.
 * @param value The value to fill the vector with.
 * @param start The start index of the range.
 * @param end The end index of the range.
 */
template <typename T>
void fillWithValueWithOff(vector<T>* v, T value, itype, itype);

/**
 * @brief Creates a clone of an existing vector.
 * 
 * This function creates a new vector that is a copy of the provided vector.
 * 
 * @tparam T The data type of the vector elements.
 * @param a The vector to clone.
 * @return A pointer to the cloned vector.
 */
template <typename T>
vector<T>* clone(vector<T>* a);

/**
 * @brief Localizes a global vector by extracting a subset.
 * 
 * This function extracts a portion of a global vector, based on the local length and shift.
 * 
 * @tparam T The data type of the vector elements.
 * @param global_v The global vector to localize.
 * @param local_len The length of the local vector to extract.
 * @param shift The index shift for the local vector.
 * @return A pointer to the localized vector.
 */
template <typename T>
vector<T>* localize_global_vector(vector<T>* global_v, int local_len, int shift);

/**
 * @brief Kernel to copy data between device memory locations.
 * 
 * This function is a CUDA kernel to copy data from a source vector to a destination vector.
 * 
 * @tparam T The data type of the vector elements.
 * @param n The number of elements to copy.
 * @param dest The destination vector.
 * @param source The source vector.
 */
template <typename T>
__global__ void _copy_kernel(itype n, T* dest, T* source);

/**
 * @brief Copies a vector from host to device memory.
 * 
 * This function copies a vector from host memory to device memory.
 * 
 * @tparam T The data type of the vector elements.
 * @param v The vector to copy to the device.
 * @return A pointer to the copied vector in device memory.
 */
template <typename T>
vector<T>* copyToDevice(vector<T>* v);

/**
 * @brief Copies data from one vector to another, supporting asynchronous execution with streams.
 * 
 * This function copies data from the source vector to the destination vector, with optional support for CUDA streams.
 * 
 * @tparam T The data type of the vector elements.
 * @param dest The destination vector.
 * @param source The source vector.
 * @param stream The CUDA stream to use (default is 0).
 */
template <typename T>
void copyTo(vector<T>* dest, vector<T>* source, cudaStream_t stream = 0);

/**
 * @brief Copies a portion of a vector from host to device memory.
 * 
 * This function copies a subrange of a vector from host memory to device memory.
 * 
 * @tparam T The data type of the vector elements.
 * @param dest The destination vector.
 * @param source The source vector.
 * @param start The starting index for the copy.
 * @param end The ending index for the copy.
 */
template <typename T>
void copyToWithOff(vector<T>* dest, vector<T>* source, itype, itype);

/**
 * @brief Copies a vector from device to host memory.
 * 
 * This function copies a vector from device memory to host memory.
 * 
 * @tparam T The data type of the vector elements.
 * @param v The vector to copy to the host.
 * @return A pointer to the copied vector in host memory.
 */
template <typename T>
vector<T>* copyToHost(vector<T>* v_d);

/**
 * @brief Frees memory allocated for a vector.
 * 
 * This function frees the memory allocated for a vector, either on the host or device.
 * 
 * @tparam T The data type of the vector elements.
 * @param v The vector to free.
 */
template <typename T>
void free(vector<T>* v);

/**
 * @brief Frees memory allocated for a vector asynchronously using a CUDA stream.
 * 
 * This function frees the memory allocated for a vector asynchronously using a specified CUDA stream.
 * 
 * @tparam T The data type of the vector elements.
 * @param v The vector to free.
 * @param stream The CUDA stream to use for asynchronous memory freeing.
 */
template <typename T>
void freeAsync(vector<T>* v, cudaStream_t stream);

/**
 * @brief Compares two vectors for equality.
 * 
 * This function compares two vectors to check if they are equal element-wise.
 * 
 * @tparam T The data type of the vector elements.
 * @param a The first vector.
 * @param b The second vector.
 * @return `true` if the vectors are equal, `false` otherwise.
 */
template <typename T>
bool equals(vector<T>* a, vector<T>* b);

/**
 * @brief Checks if a vector is filled with zeros.
 * 
 * This function checks if all elements in a vector are zero.
 * 
 * @tparam T The data type of the vector elements.
 * @param a The vector to check.
 * @return `true` if the vector is filled with zeros, `false` otherwise.
 */
template <typename T>
bool is_zero(vector<T>* a);

/**
 * @brief Loads a vector from a file.
 * 
 * This function loads a vector from a specified file and optionally places it on the device.
 * 
 * @tparam T The data type of the vector elements.
 * @param file_name The name of the file containing the vector data.
 * @param on_the_device Flag indicating whether the vector should be loaded into device memory.
 * @return A pointer to the loaded vector.
 */
template <typename T>
vector<T>* load(const char* file_name, bool on_the_device);

/**
 * @brief Prints a vector to the specified output stream.
 * 
 * This function prints the elements of a vector to the specified file or the default standard output.
 * 
 * @tparam T The data type of the vector elements.
 * @param v The vector to print.
 * @param n The number of elements to print (default is -1, meaning all elements).
 * @param fp The file pointer to output the vector (default is `stdout`).
 */
template <typename T>
void print(vector<T>* v, int n_ = -1, FILE* fp = stdout);

/**
 * @brief Computes the dot product of two vectors.
 * 
 * This function computes the dot product of two vectors using cuBLAS.
 * 
 * @tparam T The data type of the vector elements.
 * @param handle The cuBLAS handle.
 * @param a The first vector.
 * @param b The second vector.
 * @param stride_a The stride for vector `a` (default is 1).
 * @param stride_b The stride for vector `b` (default is 1).
 * @return The dot product of the vectors.
 */
template <typename T>
T dot(cublasHandle_t handle, vector<T>* a, vector<T>* b, int stride_a = 1, int stride_b = 1);

/**
 * @brief Computes the norm (magnitude) of a vector.
 * 
 * This function computes the Euclidean norm (2-norm) of a vector using cuBLAS.
 * 
 * @tparam T The data type of the vector elements.
 * @param handle The cuBLAS handle.
 * @param a The vector to compute the norm of.
 * @param stride_a The stride for vector `a` (default is 1).
 * @return The computed norm of the vector.
 */
template <typename T>
T norm(cublasHandle_t handle, vector<T>* a, int stride_a = 1);

/**
 * @brief Computes the MPI norm (magnitude) of a vector.
 * 
 * This function computes the norm of a vector across all processes using MPI and cuBLAS.
 * 
 * @tparam T The data type of the vector elements.
 * @param handle The cuBLAS handle.
 * @param a The vector to compute the norm of.
 * @return The computed MPI norm of the vector.
 */
template <typename T>
T norm_MPI(cublasHandle_t handle, vector<T>* a);

/**
 * @brief Performs the AXPY operation (y = alpha * x + y).
 * 
 * This function performs the AXPY operation on two vectors using cuBLAS.
 * 
 * @tparam T The data type of the vector elements.
 * @param handle The cuBLAS handle.
 * @param x The vector `x`.
 * @param y The vector `y`.
 * @param alpha The scalar multiplier for vector `x`.
 * @param inc The stride for the vectors (default is 1).
 */
template <typename T>
void axpy(cublasHandle_t handle, vector<T>* x, vector<T>* y, T alpha, int inc = 1);

/**
 * @brief Performs the AXPY operation (y = alpha * x + y) on a portion of the vectors.
 * 
 * This function performs the AXPY operation on a portion of the vectors defined by the index range.
 * 
 * @tparam T The data type of the vector elements.
 * @param handle The cuBLAS handle.
 * @param x The vector `x`.
 * @param y The vector `y`.
 * @param alpha The scalar multiplier for vector `x`.
 * @param start The start index for the operation.
 * @param end The end index for the operation.
 */
template <typename T>
void axpyWithOff(cublasHandle_t handle, vector<T>* x, vector<T>* y, T alpha, itype, itype);

/**
 * @brief Scales a vector by a constant factor.
 * 
 * This function scales the elements of a vector by a scalar multiplier using cuBLAS.
 * 
 * @tparam T The data type of the vector elements.
 * @param handle The cuBLAS handle.
 * @param x The vector to scale.
 * @param alpha The scalar multiplier.
 * @param inc The stride for the vector (default is 1).
 */
template <typename T>
void scale(cublasHandle_t handle, vector<T>* x, T alpha, int inc = 1);

/** Collection of vectors */
namespace Collection {

    /**
     * @brief Initializes a collection of vectors.
     * 
     * This function initializes a collection of vectors, each with its own memory allocation.
     * 
     * @tparam T The data type of the vector elements.
     * @param n The number of vectors in the collection.
     * @return A pointer to the initialized vector collection.
     */
    template <typename T>
    vectorCollection<T>* init(unsigned int n);

    /**
     * @brief Frees a vector collection.
     * 
     * This function frees the memory allocated for a collection of vectors.
     * 
     * @tparam T The data type of the vector elements.
     * @param c The collection to free.
     */
    template <typename T>
    void free(vectorCollection<T>* c);
}
}

/**
 * @brief Dumps a vector's contents to a file.
 * 
 * This function saves the contents of a vector to a file, with the filename formatted according to the
 * provided format string and arguments.
 * 
 * @tparam T The data type of the vector elements.
 * @param vec The vector to dump.
 * @param filename_fmt The format string for the filename.
 * @param ... The arguments to be inserted into the filename format.
 */
template <typename T>
void dump(vector<T>* vec, const char* filename_fmt, ...)
{
    char filename[1024] = { 0 };
    va_list args;
    va_start(args, filename_fmt);
    vsnprintf(filename, sizeof(filename), filename_fmt, args);
    va_end(args);
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Could not open %s\n", filename);
        exit(1);
    }
    Vector::print(vec, -1, fp);
    fclose(fp);
}

/**
 * @brief Dumps a portion of a vector's contents to a file.
 * 
 * This function saves a specified portion of a vector's contents to a file, with the filename formatted
 * according to the provided format string and arguments.
 * 
 * @tparam T The data type of the vector elements.
 * @param vec The vector to dump.
 * @param n The number of elements to dump.
 * @param filename_fmt The format string for the filename.
 * @param ... The arguments to be inserted into the filename format.
 */
template <typename T>
void dump(vector<T>* vec, int n, const char* filename_fmt, ...)
{
    char filename[1024] = { 0 };
    va_list args;
    va_start(args, filename_fmt);
    vsnprintf(filename, sizeof(filename), filename_fmt, args);
    va_end(args);
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Could not open %s\n", filename);
        exit(1);
    }
    Vector::print(vec, n, fp);
    fclose(fp);
}
