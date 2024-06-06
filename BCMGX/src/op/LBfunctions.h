#pragma once

extern "C" int LBsolve(double* W, double* alpha, int s);
extern "C" int LBsolvem(double* W, double* beta, int s);
extern "C" void LBdgemm(double* W, double* beta, double* b1, int s);

int LBsolve(double* W, double* alpha, int s, int id);
