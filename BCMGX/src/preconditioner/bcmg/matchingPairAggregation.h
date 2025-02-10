/** @file */
#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

/**
 * @brief Performs matching-based pairwise aggregation for AMG coarsening.
 * 
 * This function constructs the prolongation (`P`) and restriction (`R`) matrices 
 * using a matching-based aggregation approach. It supports both single-process 
 * and multi-process parallel execution with MPI communication.
 * 
 * @param h Handles for GPU execution (such as CUDA streams and memory management).
 * @param amg_data AMG build data containing relevant information for aggregation.
 * @param A Pointer to the input sparse matrix in CSR format.
 * @param w Vector containing weights for aggregation.
 * @param _P Output pointer to the computed prolongation matrix.
 * @param _R Output pointer to the computed restriction matrix.
 * @param used_by_solver Boolean flag indicating whether the aggregation is used by the solver.
 * 
 * @note If running in a distributed (MPI) environment, this function gathers and 
 *       communicates necessary data between processes and ensures proper shifting 
 *       of indices.
 * 
 * @warning The function assumes that the input matrix `A` and weights `w` are 
 *          properly initialized. Incorrect initialization may lead to undefined behavior.
 */
void matchingPairAggregation(handles* h, buildData* amg_data, CSR* A, vector<vtype>* w, CSR** _P, CSR** _R, bool used_by_solver = true);
