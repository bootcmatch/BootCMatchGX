/** @file */
#include "op/mydiag.h"

/**
 * @brief Extracts the diagonal elements from a sparse matrix in CSR format.
 *
 * This kernel retrieves the diagonal elements of a sparse matrix represented in
 * Compressed Sparse Row (CSR) format. The diagonal elements are stored in the
 * output array `D`.
 *
 * @param n The number of rows in the matrix.
 * @param val Pointer to the non-zero values of the matrix.
 * @param col Pointer to the column indices of the non-zero values.
 * @param row Pointer to the row pointers, indicating the start of each row in `val` and `col`.
 * @param D Pointer to the output array where the diagonal elements will be stored.
 * @param row_shift An integer to adjust the row index for diagonal extraction.
 */
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

/**
 * @brief Extracts the diagonal of a sparse matrix.
 *
 * This function launches the CUDA kernel `mygetDiagonal` to extract the diagonal
 * elements from a sparse matrix represented in CSR format. The results are stored
 * in the vector `D`.
 *
 * @param A Pointer to the sparse matrix in CSR format.
 * @param D Pointer to the vector where the diagonal elements will be stored.
 */
void mydiag(CSR* A, vector<vtype>* D)
{
    GridBlock gb = gb1d(D->n, BLOCKSIZE);
    mygetDiagonal<<<gb.g, gb.b>>>(D->n, A->val, A->col, A->row, D->val, 0);
}
