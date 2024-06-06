#include "op/addAbsoluteRowSumNoDiag.h"

__global__ void add_absrowsum_nodiag(stype n, vtype* A_val, itype* A_row, itype* A_col, stype row_shift, vtype* sum)
{
    stype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    vtype local_sum = 0.;

    int j;
    for (j = A_row[i]; j < A_row[i + 1]; j++) {
        if (A_col[j] != (i + row_shift)) {
            local_sum += fabs(A_val[j]);
        }
    }

    sum[i] += local_sum;
}

void addAbsoluteRowSumNoDiag(CSR* A, vector<vtype>* sum)
{
    assert(A->on_the_device);
    GridBlock gb = gb1d(A->n, BLOCKSIZE, false);
    add_absrowsum_nodiag<<<gb.g, gb.b>>>(A->n, A->val, A->row, A->col, 0, sum->val);
}
