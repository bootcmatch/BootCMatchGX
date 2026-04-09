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
    gsstype* col8,
    vtype* val,
    bool transposed)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    itype irow;
    itype icol;
    gsstype icol8;
    while (tid < nnz) {
        matrixItem_t item = items[tid];
        if (transposed) {
            irow = item.col - rowShift + 1;
            icol = item.row;
            icol8 = item.row;
        } else {
            irow = item.row - rowShift + 1;
            icol = item.col;
            icol8 = item.col;
        }
        atomicAdd(&row[irow], 1);
        col[tid] = icol;
        col8[tid] = icol8;
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
    gsstype** colRet8,
    vtype** valRet,
    bool transposed,
    bool allocateMemory)
{
    itype* row = NULL;
    itype* col = NULL;
    gsstype* col8 = NULL;
    vtype* val = NULL;

    if (allocateMemory) {
        row = CUDA_MALLOC(itype, n + 1, true);
        col = CUDA_MALLOC(itype, nnz, true);
        col8 = CUDA_MALLOC(gsstype, nnz, true);
        val = CUDA_MALLOC(vtype, nnz, true);
    } else {
        row = *rowRet;
        col = *colRet;
        col8 = *colRet8;
        val = *valRet;
    }

    GridBlock gb = getKernelParams(nnz);
    _fillCsrFromMatrixItems<<<gb.g, gb.b>>>(
        items,
        nnz,
        rowShift,
        row,
        col,
        col8,
        val,
        transposed);
    CHECK_DEVICE(cudaDeviceSynchronize());

    // Adjust row indexes
    devicePrefixSum(row, n + 1);

    *rowRet = row;
    *colRet = col;
    *colRet8 = col8;
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
    gsstype** colRet8,
    vtype** valRet,
    bool transposed,
    bool allocateMemory)
{
    itype* row = NULL;
    itype* col = NULL;
    gsstype* col8 = NULL;
    vtype* val = NULL;

    if (allocateMemory) {
        row = MALLOC(itype, n + 1, true);
        col = MALLOC(itype, nnz, true);
        col8 = MALLOC(gsstype, nnz, true);
        val = MALLOC(vtype, nnz, true);
    } else {
        row = *rowRet;
        col = *colRet;
        col8 = *colRet8;
        val = *valRet;
    }

    itype irow;
    itype icol;
    gsstype icol8;
    for (int i = 0; i < nnz; i++) {
        matrixItem_t item = items[i];
        if (transposed) {
            irow = item.col - rowShift + 1;
            icol = item.row;
            icol8 = item.row;
        } else {
            irow = item.row - rowShift + 1;
            icol = item.col;
            icol8 = item.col;
        }
        row[irow]++;
        col[i] = icol;
        col8[i] = icol8;
        val[i] = item.val;
    }

    // Adjust row indexes
    hostPrefixSum(row, n + 1);

    *rowRet = row;
    *colRet = col;
    *colRet8 = col8;
    *valRet = val;
}
