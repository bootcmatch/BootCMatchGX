/** @file */
#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"

/**
 * @brief Performs an adaptive matrix-vector product with halo communication.
 *
 * This function computes the product of a sparse matrix `A` and a local vector `local_x`,
 * while handling communication of halo data between processes. It updates the result in
 * the vector `w` based on the specified parameters.
 *
 * @param A Pointer to the sparse matrix in CSR format.
 * @param local_x Pointer to the local input vector.
 * @param w Pointer to the output vector where the result will be stored.
 * @param degree The current degree of communication.
 * @param maxdegree The maximum allowed degree of communication.
 * @param alpha The scalar multiplier for the matrix-vector product.
 * @param beta The scalar multiplier for the output vector.
 * @return vector<vtype>* Pointer to the output vector containing the result.
 */
vector<vtype>* CSRVector_product_adaptive_miniwarp_splitted(CSR* A, vector<vtype>* local_x, vector<vtype>* w, int degree, int maxdegree, vtype alpha, vtype beta);
