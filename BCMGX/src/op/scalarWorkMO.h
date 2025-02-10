/** @file */
#pragma once

#include "datastruct/vector.h"

/**
 * @brief Performs scalar work for a matrix operation.
 *
 * This function computes a matrix operation based on the input vectors and parameters.
 * It initializes matrices, performs computations, and solves linear systems using the
 * provided vectors. The behavior of the function changes based on the value of the `iter` parameter.
 *
 * @param vm Pointer to a vector containing values for matrix operations.
 * @param W Pointer to a vector that will be modified during the operation.
 * @param alpha Pointer to a vector where the result of the operation will be stored.
 * @param beta Pointer to a vector where the result of the operation will be stored.
 * @param s The size of the matrix (s x s).
 * @param iter An integer indicating the iteration (0 for the first iteration, non-zero for subsequent iterations).
 * @return int Returns an integer status code (0 for success, non-zero for failure).
 */
int scalarWorkMO(vectordh<vtype>* vm, vector<vtype>* W, vectordh<vtype>* alpha, vectordh<vtype>* beta, int s, int iter);
