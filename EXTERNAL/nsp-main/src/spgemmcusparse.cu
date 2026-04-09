#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int spgemmcusparse(int A_num_rows, int A_num_cols, int A_nnz,
               int *dA_csrOffsets, int *dA_columns, double *dA_values,
               int B_num_rows, int B_num_cols, int B_nnz, 
               int *dB_csrOffsets, int *dB_columns, double *dB_values,
               int *p2C_nnz, int **p2dC_csrOffsets, int **p2dC_columns, double **p2dC_values)   { 

    // size_t free, total;
    // int id;
    // printf("\n\n");
    // cudaDeviceSynchronize();

    
    // cudaGetDevice( &id );
    // cudaMemGetInfo( &free, &total );
    // printf("GPU %d memory: free=%zd, total=%zd\n",id,free,total);
    // cudaDeviceSynchronize();

  
    
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
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
    double               alpha       = 1.0;
    double               beta        = 0.0;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_64F;

    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )

    // // printf("\n\n");
    // cudaDeviceSynchronize();
    // // cudaMemGetInfo( &free, &total );
    // // printf("GPU %d memory: free=%zd, total=%zd\n",id,free,total);
    // printf("\tbuffer size needed = %zd   (%d)\n",bufferSize1,(int)bufferSize1);
    // cudaDeviceSynchronize();

    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )

    // printf("\n\n");
    // cudaDeviceSynchronize();
    // cudaMemGetInfo( &free, &total );
    // printf("GPU %d memory: free=%zd, total=%zd\n",id,free,total);
    // printf("\tbuffer size needed = %zd   (%d)\n",bufferSize1,(int)bufferSize1);  
    // cudaDeviceSynchronize();
    
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) )

    // printf("\n\n");
    // cudaDeviceSynchronize();
    // for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
    //     cudaSetDevice( gpu_id );
    //     int id;
    //     cudaGetDevice( &id );
    //     cudaMemGetInfo( &free, &total );
    //     printf("GPU %d memory: free=%zd, total=%zd\n",id,free,total);        
    // }
    // cudaDeviceSynchronize();

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )

    // cudaDeviceSynchronize();
    // // cudaMemGetInfo( &free, &total );
    // // printf("GPU %d memory: free=%zd, total=%zd\n",id,free,total);
    // printf("\tbuffer size 2 needed = %zd   (%d)\n",bufferSize2,(int)bufferSize2);
    // bufferSize2 = 14737418240;
    // printf("\tbuffer size 2 needed = %zd   (%d)\n",bufferSize2,(int)bufferSize2);
    // cudaDeviceSynchronize();

    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )
    p2C_nnz[0]=C_nnz1;
    // allocate matrix C
   CHECK_CUDA( cudaMalloc((void**) p2dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )

            
    CHECK_CUDA( cudaMalloc((void**) p2dC_columns, C_nnz1 * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) p2dC_values,  C_nnz1 * sizeof(double)) )
    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, p2dC_csrOffsets[0], p2dC_columns[0], p2dC_values[0]) )
        // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )


    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    return EXIT_SUCCESS;
}
