#include "col8.h"
#include "utility/utils.h"

__global__ void _col2col8(itype* col, gsstype* col8, int nnz)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nnz) {
        return;
    }
    col8[index] = col[index];
}

void col2col8(itype* col, gsstype* col8, int nnz)
{
    GridBlock gb = gb1d(nnz, NUM_THR);
    _col2col8<<<gb.g, gb.b>>>(col, col8, nnz);
    CHECK_DEVICE(cudaDeviceSynchronize());
}

// #define NTHREADSF2D 1024
// __global__ void coli2l(long* t, int* s, int n)
// {
//     int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
//     if (tid < n) {
//         t[tid] = s[tid];
//     }
// }

// #define COPYCOL(M, N)                                                       \
//     {                                                                       \
//         int nblocksf2d;                                                     \
//         (M)->col8 = CUDA_MALLOC(gsstype, (M)->nnz, false);                  \
//         nblocksf2d = ((M)->nnz + NTHREADSF2D - 1) / NTHREADSF2D;            \
//         coli2l<<<nblocksf2d, NTHREADSF2D>>>((M)->col8, (M)->col, (M)->nnz); \
//     }
