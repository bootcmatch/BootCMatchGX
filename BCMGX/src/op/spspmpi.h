/** @file */
#pragma once

#include "datastruct/CSR.h"
#include "datastruct/csrlocinfo.h"
#include "datastruct/vector.h"
#include "utility/handles.h"
#include "utility/utils.h"
#include <cuda_profiler_api.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Merges two sorted arrays into a third array.
 *
 * This function merges two sorted arrays into a single sorted array, removing duplicates.
 *
 * @param a Pointer to the first sorted array.
 * @param b Pointer to the second sorted array.
 * @param c Pointer to the output array where the merged result will be stored.
 * @param n1 * The size of the first array.
 * @param n2 The size of the second array.
 * @return itype The size of the merged array.
 */
itype merge(itype a[], itype b[], itype c[], itype n1, itype n2);

/**
 * @brief Main function for sparse matrix multiplication using GPU.
 *
 * This function performs sparse matrix multiplication on the GPU, handling
 * communication between processes and managing memory allocation.
 *
 * @param Alocal Pointer to the local sparse matrix.
 * @param Pfull Pointer to the full sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param used_by_solver A boolean indicating if the result is used by a solver.
 * @return CSR* Pointer to the resulting sparse matrix after multiplication.
 */
CSR* nsparseMGPU(CSR* Alocal, CSR* Pfull, csrlocinfo* Plocal, bool used_by_solver = true);

// void sym_mtx(CSR* Alocal);

/**
 * @brief Retrieves missing columns from a sparse matrix.
 *
 * This function identifies and returns the columns that are missing from the local
 * sparse matrix.
 *
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @return vector<int>* Pointer to a vector containing the missing column indices.
 */
vector<int>* get_missing_col(CSR* Alocal, CSR* Plocal);

/**
 * @brief Computes the rows to receive from other processes.
 *
 * This function calculates which rows need to be received from other processes
 * based on the local sparse matrix and the information about missing columns.
 *
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param _bitcol Pointer to the vector containing missing column indices.
 */
void compute_rows_to_rcv_CPU(CSR* Alocal, CSR* Plocal, vector<int>* _bitcol);

// CSR* nsparseMGPU_noCommu(handles* h, CSR* Alocal, CSR* Plocal);

// CSR* nsparseMGPU_commu(handles* h, CSR* Alocal, CSR* Plocal);

// ----------------------- PICO -------------------------------

/**
 * @brief Main function for sparse matrix multiplication without communication.
 *
 * This function performs sparse matrix multiplication on the GPU without
 * inter-process communication, handling local data only.
 *
 * @param h Pointer to the handles for managing GPU resources.
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param used_by_solver A boolean indicating if the result is used by a solver.
 * @return CSR* Pointer to the resulting sparse matrix after multiplication.
 */
CSR* nsparseMGPU_noCommu_new(handles* h, CSR* Alocal, CSR* Plocal, bool used_by_solver = true);

/**
 * @brief Main function for sparse matrix multiplication with communication.
 *
 * This function performs sparse matrix multiplication on the GPU, handling
 * communication between processes and managing memory allocation.
 *
 * @param h Pointer to the handles for managing GPU resources.
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param used_by_solver A boolean indicating if the result is used by a solver.
 * @return CSR* Pointer to the resulting sparse matrix after multiplication.
 */
CSR* nsparseMGPU_commu_new(handles* h, CSR* Alocal, CSR* Plocal, bool used_by_solver = true);

vector<int>* get_dev_missing_col(CSR* Alocal, CSR* Plocal);

/**
 * @brief Performs sparse matrix multiplication.
 *
 * This function orchestrates the sparse matrix multiplication process, handling
 * memory allocation and communication between processes.
 *
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param mem_alloc_size The size of memory to allocate for temporary storage.
 * @return CSR* Pointer to the resulting sparse matrix after multiplication.
 */
CSR* SpMM(CSR* Alocal, CSR* Plocal, int mem_alloc_size);
