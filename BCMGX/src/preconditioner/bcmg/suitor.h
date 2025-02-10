/** @file */
#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "utility/handles.h"

#define SUITOR_EPS 3.885780586188048e-10

/**
 * @brief Performs approximate matching using the GPU-based Suitor algorithm.
 * 
 * This function computes an approximate maximum weight matching of the graph 
 * represented by the sparse matrix `W` using the Suitor algorithm on the GPU.
 * The results are stored in the `M` vector, which represents the matching.
 * 
 * @param h Handles for GPU execution (such as CUDA streams and memory management).
 * @param A Pointer to the original sparse matrix in CSR format.
 * @param W Pointer to the weighted adjacency matrix in CSR format (assumed to be on the device).
 * @param M Output vector storing the matched indices.
 * @param ws Output vector storing the weight scores of the matches.
 * @param mutex Auxiliary vector used for atomic operations during matching.
 * @return Pointer to the matching vector `M`.
 * 
 * @note This function launches a GPU kernel `kernel_for_matching` to perform
 *       the matching in parallel. It also synchronizes the GPU before returning.
 * 
 * @warning The input matrix `W` must be on the device (`W->on_the_device` must be true).
 */
vector<int>* approx_match_gpu_suitor(handles* h, CSR* A, CSR* W, vector<itype>* M, vector<double>* ws, vector<int>* mutex);
