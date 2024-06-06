#include <stdio.h>
#ifdef USEMKL
#include <mkl.h>
#else
#include <lapacke.h>
#include <cblas.h>
#endif

int LBsolve(double *W, double *alpha, int s, int id) {

	int info;

	info=LAPACKE_dpotrf(LAPACK_COL_MAJOR,'U',s,W,s);//info=0: ok, info<0: illegal param, info>0: non positive definite matrix.
	if (info != 0) {printf("1 dpotrf: %d\n",info);	return info;}
	info=LAPACKE_dpotrs(LAPACK_COL_MAJOR,'U',s,1,W,s,alpha,s);//info=0: ok, info<0: illegal param.
	if (info != 0) {printf("1 dpotrs: %d\n",info); return info;}
}

int LBsolvem(double *W, double *beta, int s) {

	int info;

	info=LAPACKE_dpotrf(LAPACK_COL_MAJOR,'U',s,W,s);
	if (info != 0) {printf("m dpotrf: %d\n",info); return info;}
	info=LAPACKE_dpotrs(LAPACK_COL_MAJOR,'U',s,s,W,s,beta,s);
	if (info != 0) {printf("m dpotrs: %d\n",info); return info;}
}

void LBdgemm(double *W, double *beta, double *b1, int s) {
	//W=W-b1*beta
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, s, s, s, -1.0, b1, s, beta, s, 1.0, W, s);
}
