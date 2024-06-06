#include "op/mydiag.h"

__global__ void mygetDiagonal(stype n, vtype* val, itype* col, itype* row, vtype* D, itype row_shift)
{
    stype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    stype j_start = row[i];
    stype j_stop = row[i + 1];

    int j;
    for (j = j_start; j < j_stop; j++) {
        if (col[j] == (i + row_shift)) {
            D[i] = val[j];
        }
    }
}

void mydiag(CSR* A, vector<vtype>* D)
{
    GridBlock gb = gb1d(D->n, BLOCKSIZE);
    mygetDiagonal<<<gb.g, gb.b>>>(D->n, A->val, A->col, A->row, D->val, 0);
}
