/** @file */
#ifndef CHOLDC_H
#define CHOLDC_H

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "header.h"
#include "utility/cudamacro.h"

/**
 * @brief Host function that launches the CUDA kernel for Cholesky decomposition and solving systems.
 * 
 * This function calls the `cudacholdcMix` kernel to process multiple systems in parallel. It divides the number of systems (`number`) into blocks, where each block handles a subset of systems. The number of threads per block is defined by `BLOCKDIMENSION`.
 * 
 * @param number The total number of systems to process.
 * @param n Array containing the size of each system.
 * @param l The lower triangular matrix, which will hold the Cholesky decomposition.
 * @param x The right-hand side vector, which will hold the solution after the system is solved.
 * @param done An array indicating whether a system has been processed.
 * @param orow The row offset for the systems.
 */
void choldc(int number, int n[], REALafsai* l, REALafsai* x, int* done, int orow);

/**
 * @brief Host function that launches the CUDA kernel for Cholesky decomposition for systems of size 1.
 * 
 * This function calls the `cudacholdc1` kernel to process the systems where the size is 1. It launches the kernel with a single thread per system, performing the filtering and computation required for the Cholesky decomposition.
 * 
 * @param number The total number of systems to process (must be of size 1).
 * @param position Array containing the position indices of the systems.
 * @param iat The row pointer array of the matrix.
 * @param ja The column index array of the matrix.
 * @param coef The value array of the matrix.
 * @param iat_Filter Output array for filtered row pointers.
 * @param ja_Filter Output array for filtered column indices.
 * @param coef_Filter Output array for filtered values.
 * @param initVal The initial value for the index calculation in `iat_Filter`.
 */
void choldc1(int number, int* position, int* iat, int* ja, REALafsai* coef, int* iat_Filter, int* ja_Filter, REALafsai* coef_Filter, int initVal);

#endif
