#pragma once

#include "utility/setting.h"

#define USEONLYATOM 1

// y = a*x + b*y
void my_axpby(double* x, stype nrx, double* y, double a, double b);

void mydgemv2v(double* A, int nrA, int ncA, double* B, double* C, double f, double* D);

// A=B+f*C
void myabfc(double* A, stype nrA, double* B, double* C, double f);

__global__ void cdmm(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C);

void mydgemm(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C);

__device__ __inline__ double warpReduce(double val);

__device__ double blockReduceSum(double val);

__global__ void cdp(stype n, double* A, double* B, double* C);

void myddot(stype n, double* A, double* B, double* C);

__global__ void cdmv(double* A, stype nrA, stype ncA, double* B, double* C, double f);

void mydgemv(double* A, stype nrA, stype ncA, double* B, double* C, double f);

__global__ void ncdp(stype nsp, stype n, double* A, double* B, double* C);

void mynddot(stype nsp, stype n, double* A, double* B, double* C);

__global__ void cdmmv(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C, double* D, double* E, double f);

void mydmmv(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C, double* D, double* E, double f);
