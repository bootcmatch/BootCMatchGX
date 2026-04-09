#pragma once

#include "utility/setting.h"

void col2col8(itype* col, gsstype* col8, int nnz);

// #define COPYCOL(M, N)                                                       \
//     {                                                                       \
//         int nblocksf2d;                                                     \
//         (M)->col8 = CUDA_MALLOC(gsstype, (M)->nnz, false);                  \
//         nblocksf2d = ((M)->nnz + NTHREADSF2D - 1) / NTHREADSF2D;            \
//         coli2l<<<nblocksf2d, NTHREADSF2D>>>((M)->col8, (M)->col, (M)->nnz); \
//     }

#define COPYCOL(M, N)                                      \
    if (!(M)->col8) {                                      \
        (M)->col8 = CUDA_MALLOC(gsstype, (M)->nnz, false); \
        col2col8((M)->col, (M)->col8, (M)->nnz);           \
    }
