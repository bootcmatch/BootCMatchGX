#pragma once

#ifdef SW_USE_LIB

extern "C" int LBsolve(double* W, double* alpha, int s);
extern "C" int LBsolvem(double* W, double* beta, int s);
extern "C" void LBdgemm(double* W, double* beta, double* b1, int s);

extern "C" int my_dgetrf(double* W, int s, int* ipiv);
extern "C" int my_dgetrs(double* W, double* rhs, int s, int nrhs, int* ipiv);

int LBsolve(double* W, double* alpha, int s, int id);

#endif
