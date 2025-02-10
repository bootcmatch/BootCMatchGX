/** @file */

#pragma once

#include "utility/setting.h"

#define USEONLYATOM 1

// y = a*x + b*y
void my_axpby(double* x, stype nrx, double* y, double a, double b);

/**
 * @brief Wrapper function for the cdmv2v kernel.
 *
 * This function launches the cdmv2v kernel to perform the operation D = C + f * A * B.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the vector C.
 * @param f The scalar multiplier.
 * @param D Pointer to the output vector D.
 */
void mydgemv2v(double* A, int nrA, int ncA, double* B, double* C, double f, double* D);

// A=B+f*C
/**
 * @brief Wrapper function for the abfc kernel.
 *
 * This function launches the abfc kernel to perform the operation A = B + f * C.
 *
 * @param A Pointer to the output vector A.
 * @param nrA The number of elements in vector A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the vector C.
 * @param f The scalar multiplier.
 */
void myabfc(double* A, stype nrA, double* B, double* C, double f);

/**
 * @brief Performs the matrix multiplication C = C + A * B in parallel.
 *
 * This kernel computes the matrix product of A and B and adds it to C.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the matrix B.
 * @param nrB The number of rows in matrix B.
 * @param ncB The number of columns in matrix B.
 * @param C Pointer to the output matrix C.
 */
__global__ void cdmm(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C);

/**
 * @brief Wrapper function for the cdmm kernel.
 *
 * This function launches the cdmm kernel to perform the matrix multiplication C = C + A * B.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the matrix B.
 * @param nrB The number of rows in matrix B.
 * @param ncB The number of columns in matrix B.
 * @param C Pointer to the output matrix C.
 */
void mydgemm(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C);

__device__ __inline__ double warpReduce(double val);

__device__ double blockReduceSum(double val);

/**
 * @brief Computes the dot product of two vectors A and B.
 *
 * This kernel computes the dot product and stores the result in C.
 *
 * @param n The number of elements in the vectors.
 * @param A Pointer to the first input vector A.
 * @param B Pointer to the second input vector B.
 * @param C Pointer to the output variable where the result is stored.
 */
__global__ void cdp(stype n, double* A, double* B, double* C);

/**
 * @brief Wrapper function for the cdp kernel.
 *
 * This function launches the cdp kernel to compute the dot product of A and B.
 *
 * @param n The number of elements in the vectors.
 * @param A Pointer to the first input vector A.
 * @param B Pointer to the second input vector B.
 * @param C Pointer to the output variable where the result is stored.
 */
void myddot(stype n, double* A, double* B, double* C);

// C=C+f*A*B, nrA>>ncA dgemv2
/**
 * @brief Performs the operation C = C + f * A * B in parallel.
 *
 * This kernel computes the matrix-vector product of A and B, scales it by f,
 * and adds it to C.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the output vector C.
 * @param f The scalar multiplier.
 */
__global__ void cdmv(double* A, stype nrA, stype ncA, double* B, double* C, double f);

/**
 * @brief Wrapper function for the cdmv kernel.
 *
 * This function launches the cdmv kernel to perform the operation C = C + f * A * B.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the output vector C.
 * @param f The scalar multiplier.
 */
void mydgemv(double* A, stype nrA, stype ncA, double* B, double* C, double f);

/**
 * @brief Computes the dot product of multiple vectors A and B.
 *
 * This kernel computes the dot product for multiple segments and stores the result in C.
 *
 * @param nsp The number of segments.
 * @param n The number of elements in each vector.
 * @param A Pointer to the input matrix A.
 * @param B Pointer to the input vector B.
 * @param C Pointer to the output vector where results are stored ```cpp
 * @param C Pointer to the output vector where results are stored.
 */
__global__ void ncdp(stype nsp, stype n, double* A, double* B, double* C);

/**
 * @brief Wrapper function for the ncdp kernel.
 *
 * This function launches the ncdp kernel to compute the dot product of multiple segments.
 *
 * @param nsp The number of segments.
 * @param n The number of elements in each vector.
 * @param A Pointer to the input matrix A.
 * @param B Pointer to the input vector B.
 * @param C Pointer to the output vector where results are stored.
 */
void mynddot(stype nsp, stype n, double* A, double* B, double* C);

// C=C+A*B; E=E+f*C*D
/**
 * @brief Performs the operation C = C + A * B and E = E + f * C * D in parallel.
 *
 * This kernel computes the matrix product of A and B, adds it to C, and scales the result
 * by f before adding it to E.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the matrix B.
 * @param nrB The number of rows in matrix B.
 * @param ncB The number of columns in matrix B.
 * @param C Pointer to the output matrix C.
 * @param D Pointer to the vector D.
 * @param E Pointer to the output vector E.
 * @param f The scalar multiplier.
 */
__global__ void cdmmv(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C, double* D, double* E, double f);

/**
 * @brief Wrapper function for the cdmmv kernel.
 *
 * This function launches the cdmmv kernel to perform the operations C = C + A * B and E = E + f * C * D.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the matrix B.
 * @param nrB The number of rows in matrix B.
 * @param ncB The number of columns in matrix B.
 * @param C Pointer to the output matrix C.
 * @param D Pointer to the vector D.
 * @param E Pointer to the output vector E.
 * @param f The scalar multiplier.
 */
void mydmmv(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C, double* D, double* E, double f);
