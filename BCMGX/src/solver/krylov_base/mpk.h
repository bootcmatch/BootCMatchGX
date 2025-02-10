/**
 * @file
 */
 
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/prec_setup.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"

/**
 * @brief Performs matrix-vector products with and without preconditioning.
 * 
 * This function computes a series of matrix-vector products, alternating between using the
 * preconditioner and applying the matrix multiplication. It handles both single and multiple
 * processes and ensures synchronization where necessary. The computation is done in stages, with
 * matrix-vector products applied to slices of the vector `x_loc` and the results stored in `y_loc`.
 * 
 * @param h Handle to CUDA and other resources.
 * @param Alocal The CSR matrix representing the local matrix to be used in the matrix-vector products.
 * @param x_loc The local input vector on which the matrix-vector product is performed.
 * @param s The number of stages or slices for the matrix-vector product.
 * @param sP The input vector that will be used in computing the matrix-vector products in slices.
 * @param pr Preconditioner storage.
 * @param y_loc The local output vector where the results of the matrix-vector products are stored.
 * @param p The parameters that control the behavior of the solver, including solver settings and limits.
 * @param out The output structure that will hold additional results, such as preconditioner output.
 * 
 * @note This function is designed to handle the execution of matrix-vector products in multiple stages,
 *       applying the preconditioner in specific stages, and using different strategies for matrix-vector
 *       multiplication depending on the value of `pr->ptype`.
 * 
 * @details The function works in two main cases:
 * 1. If no preconditioner is used (`pr->ptype == PreconditionerType::NONE`), matrix-vector products are
 *    computed directly, updating the vector `x_loc` iteratively.
 * 2. If a preconditioner is applied (`pr->ptype != PreconditionerType::NONE`), the computation alternates between
 *    applying the preconditioner and performing matrix-vector multiplication, synchronizing with the network as needed.
 *    The matrix-vector multiplication can either use a full or split approach depending on the configuration.
 * 
 * The function uses MPI for communication and synchronization across multiple processes, ensuring that all the
 * processes perform the same calculations and share the results at each stage.
 */
void mpk(handles* h, CSR* Alocal, vector<vtype>* x_loc, int s, vector<vtype>* sP, cgsprec* pr, vector<vtype>* y_loc, const params& p, SolverOut* out);
