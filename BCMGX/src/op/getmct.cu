/** @file */

#include "utility/cuCompactor.cuh"
#include "utility/cudamacro.h"
#include "utility/globals.h"
#include "utility/memory.h"
#include "utility/setting.h"
#include "utility/utils.h"

#include <stdlib.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

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
int* getmct(int* Col, int nnz, int f, int l, int* uvs, int** bitcol, int* bitcolsize, int num_thr)
{
    int *d_output, d_size;
    if ((*bitcol) != NULL) {
        d_output = *bitcol;
        d_size = bitcolsize[0];
    } else {
        if (nnz > sizeof_buffer_4_getmct) {
            if (sizeof_buffer_4_getmct > 0) {
                CUDA_FREE(buffer_4_getmct);
            }
            sizeof_buffer_4_getmct = nnz;
            buffer_4_getmct = CUDA_MALLOC(int, sizeof_buffer_4_getmct, true);
        }
        d_output = buffer_4_getmct;
        d_size = cuCompactor::compact<int>(Col, d_output, nnz, int_predicate(), num_thr, 0, l - f);
        cudaDeviceSynchronize();
    }
    thrust::device_ptr<int> dev_ptr(d_output);
    if ((*bitcol) == NULL) {
        thrust::sort(dev_ptr, dev_ptr + d_size);
    }
    thrust::device_vector<int> uv(dev_ptr, dev_ptr + d_size);
    if ((*bitcol) == NULL) {
        uv.erase(thrust::unique(uv.begin(), uv.end(), int_equal_to()), uv.end());
    }
    // -------------------------------------------------------

    int* dv_ptr = thrust::raw_pointer_cast(uv.data());
    uvs[0] = uv.size();
    int* h_ptr = NULL;
    if (uvs[0] > 0) {
        h_ptr = MALLOC(int, uvs[0]);
        CHECK_DEVICE(cudaMemcpy(h_ptr, dv_ptr, uvs[0] * sizeof(int), cudaMemcpyDeviceToHost));
    }

    if ((*bitcol) == NULL && uvs[0] != 0) {
        *bitcol = CUDA_MALLOC(int, uvs[0]);
        CHECK_DEVICE(cudaMemcpy(*bitcol, dv_ptr, uv.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        *bitcolsize = *uvs;
    }
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
__global__ void primo_kernel(int* local, int* uv, int* all, int uvs, int locals, int f, unsigned int* idx)
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
__global__ void secondo_kernel(int* local, int locals, int* all, unsigned int* idx)
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
int* getmct_4shrink(int* Col, int nnz, int f, int l, int first_or_last, int* uvs, int** bitcol, int* bitcolsize, int* post_local, int num_thr)
{
    int *d_output, d_size;
    if ((*bitcol) != NULL) {
        d_output = *bitcol;
        d_size = bitcolsize[0];
    } else {
        if (nnz > sizeof_buffer_4_getmct) {
            if (sizeof_buffer_4_getmct > 0) {
                CUDA_FREE(buffer_4_getmct);
            }
            sizeof_buffer_4_getmct = nnz;
            buffer_4_getmct = CUDA_MALLOC(int, sizeof_buffer_4_getmct, true);
        }
        d_output = buffer_4_getmct;
        d_size = cuCompactor::compact<int>(Col, d_output, nnz, int_predicate(), num_thr, 0, l);
        cudaDeviceSynchronize();
    }
    thrust::device_ptr<int> dev_ptr(d_output);
    if ((*bitcol) == NULL) {
        thrust::sort(dev_ptr, dev_ptr + d_size);
    }
    thrust::device_vector<int> uv(dev_ptr, dev_ptr + d_size);
    if ((*bitcol) == NULL) {
        uv.erase(thrust::unique(uv.begin(), uv.end(), int_equal_to()), uv.end());
    }

    thrust::device_vector<int> result((l - f + 1) + uv.size(), -1);

    unsigned int first_post_local;
    if (first_or_last) {
        if (first_or_last < 0) {
            thrust::sequence(result.begin(), result.begin() + l + 1, 0);
            thrust::copy(uv.begin(), uv.end(), result.begin() + l + 1);
            first_post_local = 0;
        } else {
            thrust::copy(uv.begin(), uv.end(), result.begin());
            thrust::sequence(result.begin() + uv.size(), result.end(), 0);
            first_post_local = (unsigned int)uv.size();
        }
    } else {
        thrust::device_vector<int> locals(l - f + 1, 1);
        thrust::sequence(locals.begin(), locals.end(), 0);

        if (alloced_idx == false) {
            idx_4shrink = CUDA_MALLOC(unsigned int, 1, true);
            alloced_idx = true;
        }
        unsigned int* dev_idx = idx_4shrink;

        CHECK_DEVICE(cudaMemset(dev_idx, 0, sizeof(unsigned int)));

        int* dv_uv = thrust::raw_pointer_cast(uv.data());
        int* dv_local = thrust::raw_pointer_cast(locals.data());
        int* dv_result = thrust::raw_pointer_cast(result.data());
        GridBlock gb = gb1d(uv.size(), NUM_THR);
        if (gb.g.x > 0) {
            primo_kernel<<<gb.g, gb.b>>>(dv_local, dv_uv, dv_result, uv.size(), locals.size(), 0, dev_idx);
        }
        CHECK_DEVICE(cudaMemcpy(&first_post_local, dev_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        gb = gb1d(locals.size(), NUM_THR);
        if (gb.g.x > 0) {
            secondo_kernel<<<gb.g, gb.b>>>(dv_local, locals.size(), dv_result, dev_idx);
        }
    }

    uvs[0] = result.size();
    int* dv_ptr = thrust::raw_pointer_cast(result.data());

    assert(uvs[0] == ((l - f + 1) + uv.size()));
    assert(uvs[0] >= (l - f));

    if ((*bitcol) == NULL) {
        int* dv_ptr2 = thrust::raw_pointer_cast(uv.data());
        *bitcol = CUDA_MALLOC(int, uvs[0]);
        CHECK_DEVICE(cudaMemcpy(*bitcol, dv_ptr2, uv.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        *bitcolsize = d_size;
    }
    *post_local = (int)first_post_local;

    d_output = CUDA_MALLOC(int, uvs[0]);
    CHECK_DEVICE(cudaMemcpy(d_output, dv_ptr, uvs[0] * sizeof(int), cudaMemcpyDeviceToDevice));

    return d_output;
}
