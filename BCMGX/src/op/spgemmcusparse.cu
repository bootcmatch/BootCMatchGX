#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

double CUSPARSE_CHUNK_FRACTION = 0.025;

#define FREE_EVERYTIME 0

cusparseHandle_t handle       = NULL;
void*            dBuffer1     = NULL;
void*            dBuffer2     = NULL;
void*            dBuffer3     = NULL;
size_t           gBufferSize1 = 0;
size_t           gBufferSize2 = 0;
size_t           gBufferSize3 = 0;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

void spgemmcusparseAlloc1(size_t bufferSize) {
    if (bufferSize > gBufferSize1) {
        if (dBuffer1 != NULL) {
            CHECK_CUDA( cudaFree(dBuffer1) )
        }
        gBufferSize1 = bufferSize;
        CHECK_CUDA( cudaMalloc((void**) &dBuffer1, gBufferSize1) )
    }
}

void spgemmcusparseAlloc2(size_t bufferSize) {
    if (bufferSize > gBufferSize2) {
        if (dBuffer2 != NULL) {
            CHECK_CUDA( cudaFree(dBuffer2) )
        }
        gBufferSize2 = bufferSize;
        CHECK_CUDA( cudaMalloc((void**) &dBuffer2, gBufferSize2) )
    }
}

void spgemmcusparseAlloc3(size_t bufferSize) {
    if (bufferSize > gBufferSize3) {
        if (dBuffer3 != NULL) {
            CHECK_CUDA( cudaFree(dBuffer3) )
        }
        gBufferSize3 = bufferSize;
        CHECK_CUDA( cudaMalloc((void**) &dBuffer3, gBufferSize3) )
    }
}

int spgemmcusparse(int A_num_rows, int A_num_cols, int A_nnz,
               int *dA_csrOffsets, int *dA_columns, double *dA_values,
               int B_num_rows, int B_num_cols, int B_nnz, 
               int *dB_csrOffsets, int *dB_columns, double *dB_values,
               int *p2C_nnz, int **p2dC_csrOffsets, int **p2dC_columns, double **p2dC_values)   { 
    
    
    cusparseSpMatDescr_t matA, matB, matC;
    cusparseSpGEMMAlg_t  alg    = CUSPARSE_SPGEMM_ALG3;
    int64_t              num_prods;

    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;
    size_t bufferSize3 = 0;

    if (handle == NULL) {
        CHECK_CUSPARSE( cusparseCreate(&handle) )
    }

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                      dB_csrOffsets, dB_columns, dB_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    //--------------------------------------------------------------------------
    double              alpha       = 1.0;
    double              beta        = 0.0;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_64F;

    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // SpGEMM Computation

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                        &alpha, matA, matB, &beta, matC,
                                        computeType, alg,
                                        spgemmDesc, &bufferSize1, NULL) )
    // printf("bufferSize1=%ld\n",bufferSize1);
    spgemmcusparseAlloc1(bufferSize1);

    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                        &alpha, matA, matB, &beta, matC,
                                        computeType, alg,
                                        spgemmDesc, &bufferSize1, dBuffer1) )

    CHECK_CUSPARSE(cusparseSpGEMM_getNumProducts(spgemmDesc, &num_prods) )

    // printf("num_prods=%ld\n",num_prods);

    // ask bufferSize3 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_estimateMemory(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, alg,
                                      spgemmDesc, CUSPARSE_CHUNK_FRACTION,
                                      &bufferSize3, NULL, NULL) )
    // printf("bufferSize3=%ld\n",bufferSize3);
    spgemmcusparseAlloc3(bufferSize3);

    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_estimateMemory(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, alg,
                                      spgemmDesc, CUSPARSE_CHUNK_FRACTION,
                                      &bufferSize3, dBuffer3,
                                      &bufferSize2) )

    // CHECK_CUDA( cudaFree(dBuffer3) ) // dBuffer3 can be safely freed to
    //                                  // save more memory

    // printf("bufferSize2=%ld\n",bufferSize2);
    spgemmcusparseAlloc2(bufferSize2);

    // compute the intermediate product of A * B
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                                 &alpha, matA, matB, &beta, matC,
                                 computeType, alg,
                                 spgemmDesc, &bufferSize2, dBuffer2) )
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )
    // printf("C_nnz1=%ld\n",C_nnz1);
    p2C_nnz[0]=C_nnz1;
    // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) p2dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) ) 
    CHECK_CUDA( cudaMalloc((void**) p2dC_columns, C_nnz1 * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) p2dC_values,  C_nnz1 * sizeof(double)) )

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, p2dC_csrOffsets[0], p2dC_columns[0], p2dC_values[0]) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, alg, spgemmDesc) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    // CHECK_CUSPARSE( cusparseDestroy(handle) )

    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // device memory deallocation
    #if FREE_EVERYTIME
    void spgemmcusparseFree();
    spgemmcusparseFree();
    #endif

    return EXIT_SUCCESS;
}

void spgemmcusparseFree() {
    if (dBuffer1 != NULL) {
        CHECK_CUDA( cudaFree(dBuffer1) )
        dBuffer1 = NULL;
        gBufferSize1 = 0;
    }
    if (dBuffer2 != NULL) {
        CHECK_CUDA( cudaFree(dBuffer2) )
        dBuffer2 = NULL;
        gBufferSize2 = 0;
    }
    if (dBuffer3 != NULL) {
        CHECK_CUDA( cudaFree(dBuffer3) )
        dBuffer3 = NULL;
        gBufferSize3 = 0;
    }
    if (handle != NULL) {
        CHECK_CUSPARSE( cusparseDestroy(handle) )
        handle = NULL;
    }
}