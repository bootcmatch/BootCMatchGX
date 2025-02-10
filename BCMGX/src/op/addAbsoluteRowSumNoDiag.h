/**
 * @file
 */
#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"

/**
 * @brief Wrapper function to compute the absolute row sum of a sparse matrix.
 *
 * This function prepares the data and launches the CUDA kernel `add_absrowsum_nodiag`
 * to compute the absolute row sums of the sparse matrix `A`, storing the results in
 * the `sum` vector.
 *
 * @param A Pointer to the sparse matrix in CSR format.
 * @param sum Pointer to the vector where the absolute row sums will be stored.
 */
void addAbsoluteRowSumNoDiag(CSR* A, vector<vtype>* sum);
