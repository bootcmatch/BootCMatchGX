#include "datastruct/matrixItem.h"
#include "utility/devicePrefixSum.h"
#include "utility/hostPrefixSum.h"
#include "utility/memory.h"

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
        fprintf(f, "\t[%zu]: r %ld, c %ld = %lf\n", i, hArr[i].row, hArr[i].col, hArr[i].val);
    }

    if (isOnDevice) {
        FREE(hArr);
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
        row = CUDA_MALLOC(itype, n + 1, true);
        col = CUDA_MALLOC(itype, nnz, true);
        val = CUDA_MALLOC(vtype, nnz, true);
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

/**
 * Scans a vector of matrix items and fills the CSR.
 */
void fillCsrFromMatrixItems_nogpu(
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
        row = MALLOC(itype, n + 1, true);
        col = MALLOC(itype, nnz, true);
        val = MALLOC(vtype, nnz, true);
    } else {
        row = *rowRet;
        col = *colRet;
        val = *valRet;
    }

    itype irow;
    itype icol;
    for (int i = 0; i < nnz; i++) {
        matrixItem_t item = items[i];
        if (transposed) {
            irow = item.col - rowShift + 1;
            icol = item.row;
        } else {
            irow = item.row - rowShift + 1;
            icol = item.col;
        }
        row[irow]++;
        col[i] = icol;
        val[i] = item.val;
    }

    // Adjust row indexes
    hostPrefixSum(row, n + 1);

    *rowRet = row;
    *colRet = col;
    *valRet = val;
}
