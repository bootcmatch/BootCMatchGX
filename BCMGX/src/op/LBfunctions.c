#ifdef SW_USE_LIB

#include <stdio.h>
#ifdef USEMKL
#include <mkl.h>
#else
#include <lapacke.h>
#include <cblas.h>
#endif

/**
 * @brief Solves a linear system using Cholesky decomposition.
 *
 * This function performs Cholesky factorization of a symmetric positive definite matrix
 * and then solves the linear system \( W \cdot x = \alpha \) for \( x \).
 *
 * @param W Pointer to the matrix in column-major order.
 * @param alpha Pointer to the right-hand side vector.
 * @param s The size of the matrix (s x s).
 * @param id An identifier for the operation (not used in this implementation).
 * @return int Returns an integer status code (0 for success, non-zero for failure).
 */
int LBsolve(double *W, double *alpha, int s, int id) {

	int info;

	info=LAPACKE_dpotrf(LAPACK_COL_MAJOR,'U',s,W,s);//info=0: ok, info<0: illegal param, info>0: non positive definite matrix.
	if (info != 0) {printf("1 dpotrf: %d\n",info);	return info;}
	info=LAPACKE_dpotrs(LAPACK_COL_MAJOR,'U',s,1,W,s,alpha,s);//info=0: ok, info<0: illegal param.
	if (info != 0) {printf("1 dpotrs: %d\n",info); return info;}
}

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
int LBsolvem(double *W, double *beta, int s) {

	int info;

	info=LAPACKE_dpotrf(LAPACK_COL_MAJOR,'U',s,W,s);
	if (info != 0) {printf("m dpotrf: %d\n",info); return info;}
	info=LAPACKE_dpotrs(LAPACK_COL_MAJOR,'U',s,s,W,s,beta,s);
	if (info != 0) {printf("m dpotrs: %d\n",info); return info;}
}

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
void LBdgemm(double *W, double *beta, double *b1, int s) {
	//W=W-b1*beta
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, s, s, s, -1.0, b1, s, beta, s, 1.0, W, s);
}

#endif
