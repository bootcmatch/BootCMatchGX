#include "datastruct/matrixItem.h"
#include "utility/devicePrefixSum.h"

MPI_Datatype MPI_MATRIX_ITEM_T;

void registerMatrixItemMpiDatatypes()
{
    static int first = 1;
    if (first) {
        MPI_Type_contiguous(sizeof(matrixItem_t), MPI_BYTE, &MPI_MATRIX_ITEM_T);
        MPI_Type_commit(&MPI_MATRIX_ITEM_T);

        first = 0;
    }
}

void debugMatrixItems(const char* title, matrixItem_t* arr, size_t len, bool isOnDevice, FILE* f)
{
    matrixItem_t* hArr = arr;
    if (isOnDevice) {
        hArr = copyArrayToHost(arr, len);
    }

    fprintf(f, "%s:\n", title);
    for (size_t i = 0; i < len; i++) {
        fprintf(f, "\t[%zu]: r %d, c %d = %lf\n", i, hArr[i].row, hArr[i].col, hArr[i].val);
    }

    if (isOnDevice) {
        free(hArr);
    }
}

/**
 * CUDA kernel.
 * Scans a vector of matrix items and fills the CSR.
 * BE CAREFULL: row must be prefix-summed, after this.
 * Invoke using one thread per item.
 */
__global__ void _fillCsrFromMatrixItems(
    matrixItem_t* items,
    size_t nnz,
    int rowShift,
    itype* row,
    itype* col,
    vtype* val,
    bool transposed)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    itype irow;
    itype icol;
    while (tid < nnz) {
        matrixItem_t item = items[tid];
        if (transposed) {
            irow = item.col - rowShift + 1;
            icol = item.row;
        } else {
            irow = item.row - rowShift + 1;
            icol = item.col;
        }
        atomicAdd(&row[irow], 1);
        col[tid] = icol;
        val[tid] = item.val;
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * Scans a vector of matrix items and fills the CSR.
 */
void fillCsrFromMatrixItems(
    matrixItem_t* items,
    size_t nnz,
    size_t n,
    int rowShift,
    itype** rowRet,
    itype** colRet,
    vtype** valRet,
    bool transposed,
    bool allocateMemory)
{
    itype* row = NULL;
    itype* col = NULL;
    vtype* val = NULL;

    if (allocateMemory) {
        CHECK_DEVICE(cudaMalloc(&row, (n + 1) * sizeof(itype)));
        CHECK_DEVICE(cudaMemset(row, 0, (n + 1) * sizeof(itype)));

        CHECK_DEVICE(cudaMalloc(&col, nnz * sizeof(itype)));
        CHECK_DEVICE(cudaMemset(col, 0, nnz * sizeof(itype)));

        CHECK_DEVICE(cudaMalloc(&val, nnz * sizeof(vtype)));
        CHECK_DEVICE(cudaMemset(val, 0, nnz * sizeof(vtype)));
    } else {
        row = *rowRet;
        col = *colRet;
        val = *valRet;
    }

    GridBlock gb = getKernelParams(nnz);
    _fillCsrFromMatrixItems<<<gb.g, gb.b>>>(
        items,
        nnz,
        rowShift,
        row,
        col,
        val,
        transposed);
    CHECK_DEVICE(cudaDeviceSynchronize());

    // Adjust row indexes
    devicePrefixSum(row, n + 1);

    *rowRet = row;
    *colRet = col;
    *valRet = val;
}
