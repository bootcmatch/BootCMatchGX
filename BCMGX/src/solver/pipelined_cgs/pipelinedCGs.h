/**
 * @file pipelinedCGs.cu
 * @brief Implementation of the Pipelined Conjugate Gradient Solver using CUDA.
 * 
 * This file contains the implementation of the pipelined conjugate gradient (CG) solver that uses CUDA for parallel execution.
 * The solver handles large sparse systems of linear equations by performing computations over multiple steps and optimizing memory and GPU usage.
 * 
 * This implementation is based on the following references:
 * - Tiwari, M., Vadhiyar, S., "Pipelined Preconditioned s-step Conjugate Gradient Methods for Distributed Memory Systems"
 *   Proceedings of IEEE International Conference on Cluster Computing (CLUSTER), 2021, pages 215-225.
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/prec_setup.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"

/**
 * @brief Solves the linear system \( A x = b \) using the Pipelined Conjugate Gradient Solver.
 * 
 * @param h Handle to CUDA and other resources.
 * @param Alocal The local CSR matrix used in the matrix-vector products.
 * @param rhs_loc The local right-hand side vector.
 * @param x0_loc The local initial guess vector.
 * @param p The parameters that control the behavior of the solver, including solver settings and limits.
 * @param pr Preconditioner storage.
 * @param out The output structure that will store the results, including residual history and final solution.
 * 
 * @return The solution vector after performing the CG s-step method, stored in `out->sol_local`.
 * 
 * @note The final solution is returned in the output structure `out->sol_local`, and the result of the CG s-step
 *       method is captured in `out->retv`, which indicates the status of the solver.
 * 
 * @details The function acts as a wrapper to invoke the `pipelinedcgsstep` function, providing an interface for the user
 *          to solve the system. The initial guess `x0_loc` is updated in-place with the final solution, and the
 *          output structure contains additional information, such as the residual history and exit conditions.
 */
vector<vtype>* solve_pipelined_cgs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, cgsprec* pr, const params& p, SolverOut* out);
