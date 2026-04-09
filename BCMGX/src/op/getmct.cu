/** @file */

#include "utility/cuCompactor.cuh"
#include "utility/cudamacro.h"
#include "utility/globals.h"
#include "utility/memory.h"
#include "utility/setting.h"
#include "utility/utils.h"
#include "utility/arrays.h"
#include "utility/devicePartition.h"
#include "utility/deviceMemEq.h"

#include <stdlib.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/functional.h> 

using namespace std;

/**
 * @brief Predicate to check if an integer is outside a specified range.
 */
struct int_predicate {
    __host__ __device__ bool operator()(const int x, const int f, const int l)
    {
        return ((x < f) || (x > l));
    }
};

struct long_predicate {
    __host__ __device__ bool operator()(const long x, const long f, const long l)
    {
        return ((x < f) || (x > l));
    }
};

/**
 * @brief Predicate to check if two integers are equal.
 */
struct int_equal_to {
    __host__ __device__ bool operator()(int a, int b)
    {
        return a == b;
    }
};

/**
 * @brief Predicate to check if two integers are equal.
 */
struct long_equal_to {
    __host__ __device__ bool operator()(long a, long b)
    {
        return a == b;
    }
};

/**
 * @brief Compacts the input array based on a specified range and returns unique values.
 *
 * This function compacts the input array `Col` by removing elements that fall within
 * the range defined by `f` and `l`. It uses a custom predicate to determine which
 * elements to keep. The resulting unique values are returned in the output array.
 *
 * @param Col Pointer to the input array of integers.
 * @param nnz The number of non-zero elements in the input array.
 * @param f The lower bound of the range.
 * @param l The upper bound of the range.
 * @param uvs Pointer to an integer array where the size of unique values will be stored.
 * @param bitcol Pointer to a pointer that will hold the compacted output array.
 * @param bitcolsize Pointer to an integer array that holds the size of the compacted output.
 * @param num_thr The number of threads to use for the operation.
 * @return int* Pointer to the host array containing the unique values.
 */
long* getmct(gsstype* Col, int nnz, int f, int l, int* uvs, long** bitcol, int* bitcolsize, int* nonuniquesize, int num_thr)
{
    long *d_out = NULL;
    size_t d_out_size = 0;

    if ((*bitcol) != NULL) {
        d_out      = *bitcol;
        d_out_size = bitcolsize[0];
    } else {
        d_out = CUDA_MALLOC(long, nnz, true);

        devicePartition(Col, nnz, [l]__device__(const long &x){
            return x < 0 || x > l;
        }, d_out, &d_out_size);

        *nonuniquesize = d_out_size;

        thrust::device_ptr<long> dev_out(d_out);
        thrust::sort(dev_out, dev_out + d_out_size);
        d_out_size = (thrust::unique(dev_out, dev_out + d_out_size) - dev_out);

        *bitcol = d_out;
        *bitcolsize = d_out_size;
    }

    uvs[0] = d_out_size;

    long* h_ptr = copyArrayToHost(d_out, uvs[0]);
    return h_ptr;
}

/**
 * @brief Kernel to process the first part of the compacted data.
 *
 * This kernel processes the unique values and distributes them into the output array
 * based on whether they are below or above a specified threshold.
 *
 * @param local Pointer to the local array for storing results.
 * @param uv Pointer to the unique values array.
 * @param all Pointer to the output array where results will be stored.
 * @param uvs The number of unique values.
 * @param locals The number of local elements.
 * @param f The threshold value.
 * @param idx Pointer to an index array for atomic operations.
 */
__global__ void primo_kernel(long* local, long* uv, long* all, int uvs, int locals, int f, unsigned int* idx)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lowerhalo = 0;
    if (tid < uvs) {
        lowerhalo = uv[tid] < f;
        if (lowerhalo) {
            all[tid] = uv[tid];
        } else {
            all[tid + locals] = uv[tid];
        }
    }

    lowerhalo = lowerhalo ? tid + 1 : 0;
    if (lowerhalo > 0) {
        unsigned int compared, result;
        while (true) {
            compared = idx[0];
            if (lowerhalo > compared) {
                result = atomicCAS(idx, compared, lowerhalo);
                if (result == lowerhalo) {
                    break;
                }
            } else {
                break;
            }
        }
    }
}

/**
 * @brief Kernel to copy local results to the global output array.
 *
 * This kernel copies elements from the local array to the global output array
 * based on the index provided. It is used to finalize the results after processing.
 *
 * @param local Pointer to the local array containing results.
 * @param locals The number of local elements to copy.
 * @param all Pointer to the global output array where results will be stored.
 * @param idx Pointer to an index array that indicates where to start copying in the global array.
 */
__global__ void secondo_kernel(long* local, int locals, long* all, unsigned int* idx)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < locals) {
        all[idx[0] + tid] = local[tid];
    }
}

extern unsigned int* idx_4shrink; ///< Pointer to an index array used for shrinking.
extern bool alloced_idx; ///< Flag indicating whether the index array has been allocated.

/**
 * @brief Compacts the input array and processes it based on specified conditions.
 *
 * This function compacts the input array `Col` by removing elements that fall within
 * the range defined by `f` and `l`. It also processes the unique values based on the
 * `first_or_last` parameter and returns the results in a new array.
 *
 * @param Col Pointer to the input array of integers.
 * @param nnz The number of non-zero elements in the input array.
 * @param f The lower bound of the range.
 * @param l The upper bound of the range.
 * @param first_or_last An integer indicating whether to process the first or last elements.
 * @param uvs Pointer to an integer array where the size of unique values will be stored.
 * @param bitcol Pointer to a pointer that will hold the compacted output array.
 * @param bitcolsize Pointer to an integer array that holds the size of the compacted output.
 * @param post_local Pointer to an integer where the index of the first local element will be stored.
 * @param num_thr The number of threads to use for the operation.
 * @return int* Pointer to the host array containing the processed results.
 */
long* getmct_4shrink(long* Col, int nnz, int f, int l, int first_or_last, int* uvs, long** bitcol, int* bitcolsize, int* nonuniquesize, int* post_local, int num_thr)
{
    long *d_out = NULL;
    long *d_in = NULL;
    size_t d_out_size = 0;
    size_t d_in_size = 0;

    if ((*bitcol) != NULL) {
        d_out      = *bitcol;
        d_out_size = bitcolsize[0];
    } else {
        d_out = CUDA_MALLOC(long, nnz, true);;

        devicePartition(Col, nnz, [l]__device__(const long &x){
            return x < 0 || x > l;
        }, d_out, &d_out_size);

        *nonuniquesize = d_out_size;

        thrust::device_ptr<long> dev_out(d_out);
        thrust::sort(dev_out, dev_out + d_out_size);
        d_out_size = (thrust::unique(dev_out, dev_out + d_out_size) - dev_out);    
        
        *bitcol = d_out;
        *bitcolsize = d_out_size;
    }

    // -----------------------------------------------------------------

    d_in = d_out + *nonuniquesize;
    d_in_size = nnz - *nonuniquesize;

    thrust::device_ptr<long> dev_in(d_in);
    thrust::sort(dev_in, dev_in + d_in_size);
    d_in_size = (thrust::unique(dev_in, dev_in + d_in_size) - dev_in);

    long d_result_size = d_out_size + d_in_size;
    long *d_result = CUDA_MALLOC(long, d_result_size);

    unsigned int first_post_local;
    if (first_or_last < 0) {
        if (d_in_size) {
            CHECK_DEVICE(cudaMemcpy(d_result            , d_in , d_in_size  * sizeof(long), cudaMemcpyDeviceToDevice));
        }
        if (d_out_size) {
            CHECK_DEVICE(cudaMemcpy(d_result + d_in_size, d_out, d_out_size * sizeof(long), cudaMemcpyDeviceToDevice));
        }
        first_post_local = 0;
    } else if (first_or_last > 0) {
        if (d_out_size) {
            CHECK_DEVICE(cudaMemcpy(d_result             , d_out, d_out_size * sizeof(long), cudaMemcpyDeviceToDevice));
        }
        if (d_in_size) {
            CHECK_DEVICE(cudaMemcpy(d_result + d_out_size, d_in , d_in_size  * sizeof(long), cudaMemcpyDeviceToDevice));
        }
        first_post_local = d_out_size;
    } else {
        if (alloced_idx == false) {
            idx_4shrink = CUDA_MALLOC(unsigned int, 1, true);
            alloced_idx = true;
        }
        unsigned int* dev_idx = idx_4shrink;

        CHECK_DEVICE(cudaMemset(dev_idx, 0, sizeof(unsigned int)));

        GridBlock gb = gb1d(d_out_size, NUM_THR);
        if (gb.g.x > 0) {
            primo_kernel<<<gb.g, gb.b>>>(d_in, d_out, d_result, d_out_size, d_in_size, 0, dev_idx);
        }
        CHECK_DEVICE(cudaMemcpy(&first_post_local, dev_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        gb = gb1d(d_in_size, NUM_THR);
        if (gb.g.x > 0) {
            secondo_kernel<<<gb.g, gb.b>>>(d_in, d_in_size, d_result, dev_idx);
        }
    }

    // -----------------------------------------------------------------

    uvs[0] = d_result_size;
    *post_local = (int)first_post_local;
    return d_result;
}
