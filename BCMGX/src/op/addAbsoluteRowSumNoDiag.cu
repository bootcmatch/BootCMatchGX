/** @file */

#include "op/addAbsoluteRowSumNoDiag.h"

/**
 * @brief Computes the absolute row sum of a sparse matrix, excluding the diagonal elements.
 *
 * This kernel iterates over each row of the sparse matrix represented in CSR format,
 * calculates the sum of the absolute values of the non-diagonal elements, and adds
 * the result to the corresponding entry in the `sum` array.
 *
 * @param n The number of rows in the matrix.
 * @param A_val Pointer to the non-zero values of the matrix.
 * @param A_row Pointer to the row pointers, indicating the start of each row in `A_val` and `A_col`.
 * @param A_col Pointer to the column indices of the non-zero values.
 * @param row_shift An integer to adjust the row index for diagonal exclusion.
 * @param sum Pointer to the output array where the absolute row sums will be stored.
 */
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

/**
 * @brief Wrapper function to compute the absolute row sum of a sparse matrix.
 *
 * This function prepares the data and launches the CUDA kernel `add_absrowsum_nodiag`
 * to compute the absolute row sums of the sparse matrix `A`, storing the results in
 * the `sum` vector.
 *
 * @param A Pointer to the sparse matrix in CSR format.
 * @param sum Pointer to the vector where the absolute row sums will be stored.
 */
void addAbsoluteRowSumNoDiag(CSR* A, vector<vtype>* sum)
{
    assert(A->on_the_device);
    GridBlock gb = gb1d(A->n, BLOCKSIZE, false);
    add_absrowsum_nodiag<<<gb.g, gb.b>>>(A->n, A->val, A->row, A->col, 0, sum->val);
}
