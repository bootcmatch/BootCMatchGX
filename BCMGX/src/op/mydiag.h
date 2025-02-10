/** @file */
#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"

/**
 * @brief Extracts the diagonal of a sparse matrix.
 *
 * This function launches the CUDA kernel `mygetDiagonal` to extract the diagonal
 * elements from a sparse matrix represented in CSR format. The results are stored
 * in the vector `D`.
 *
 * @param A Pointer to the sparse matrix in CSR format.
 * @param D Pointer to the vector where the diagonal elements will be stored.
 */
void mydiag(CSR* A, vector<vtype>* D);
