/** @file */
#pragma once

#include "datastruct/vector.h"
#include "utility/setting.h"

/**
 * @brief Computes the triple inner product of three vectors in parallel using CUDA.
 *
 * This kernel performs the computation of the triple inner product:
 * \f[
 * \text{result} = \sum_{i=0}^{n-1} (r[i] \cdot v[i]) + (w[i] \cdot v[i]) + (q[i] \cdot v[i])
 * \f]
 * The results are accumulated in shared memory and then combined using atomic operations.
 *
 * @param n The number of elements in the vectors.
 * @param r Pointer to the first input vector.
 * @param w Pointer to the second input vector.
 * @param q Pointer to the third input vector.
 * @param v Pointer to the vector used for multiplication.
 * @param alpha_beta_gamma Pointer to an array where the results (alpha, beta, gamma) will be stored.
 * @param shift The shift index for accessing the vector `v`.
 */
 void triple_innerproduct(vector<vtype>* r, vector<vtype>* w, vector<vtype>* q, vector<vtype>* v, vector<vtype>* alpha_beta_gamma, vtype* alpha, vtype* beta, vtype* gamma, itype shift);
