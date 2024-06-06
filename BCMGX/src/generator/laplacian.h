#pragma once
#include "utility/setting.h"

struct CSR;

CSR* generateLocalLaplacian3D(itype n);
CSR* generateLocalLaplacian3D_7p(itype nx, itype ny, itype nz, itype P, itype Q, itype R);
CSR* generateLocalLaplacian3D_27p(itype nx, itype ny, itype nz, itype P, itype Q, itype R);
int* read_laplacian_file(const char* file_name);
