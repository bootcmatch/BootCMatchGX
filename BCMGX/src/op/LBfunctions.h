#pragma once

#ifdef SW_USE_LIB

/**
 * @brief Solves a linear system using Cholesky decomposition for multiple right-hand sides.
 *
 * This function performs Cholesky factorization of a symmetric positive definite matrix
 * and then solves the linear system \( W \cdot X = \beta \) for \( X \), where \( X \)
 * contains multiple right-hand sides.
 *
 * @param W Pointer to the matrix in column-major order.
 * @param beta Pointer to the right-hand side matrix (multiple vectors).
 * @param s The size of the matrix (s x s).
 * @return int Returns an integer status code (0 for success, non-zero for failure).
 */
extern "C" int LBsolvem(double* W, double* beta, int s);

/**
 * @brief Performs matrix-matrix multiplication and updates the result.
 *
 * This function computes the operation \( W = W - b1 \cdot \beta \) using the
 * CBLAS library for matrix-matrix multiplication.
 *
 * @param W Pointer to the matrix that will be updated.
 * @param beta Pointer to the second matrix involved in the multiplication.
 * @param b1 Pointer to the first matrix involved in the multiplication.
 * @param s The size of the matrices (s x s).
 */
extern "C" void LBdgemm(double* W, double* beta, double* b1, int s);

int LBsolve(double* W, double* alpha, int s, int id);
extern "C" int LBsolve(double* W, double* beta, int s);

#endif
