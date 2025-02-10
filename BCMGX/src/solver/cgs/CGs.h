/**
 * @file CGs.h
 * @brief Implementation of a variant of the Conjugate Gradient (CG) s-step method
 * that also allows for variable preconditioners across iterations.
 *
 * This implementation is based on the following references:
 * - Chronopoulos, A., Gear, C., "s-step Iterative Methods for Symmetric Linear Systems", 
 *   J. Comput. Appl. Math., Vol. 25, N. 2, 1989. pages 153--168.
 * - Chronopoulos, A., Gear, C., "On the Efficient Implementation of Preconditioned
 *   s-step Conjugate Gradient Methods on Multiprocessors with Memory Hierarchy", 
 *   Parallel Computing, Vol. 11, N. 1, 1989, pages 37--53.
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/prec_apply.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"

/**
 * @brief Solves the linear system \( A x = b \) using the Conjugate Gradient (CG) s-step method.
 * 
 * This function provides an interface to solve a linear system using the CG s-step method. It calls the `cgsstep`
 * function to perform the iterative solver steps and returns the final solution vector.
 * 
 * @param h Handle to CUDA and other resources.
 * @param Alocal The local CSR matrix used in the matrix-vector products.
 * @param rhs_loc The local right-hand side vector.
 * @param x_loc The local initial guess vector.
 * @param p The parameters that control the behavior of the solver, including solver settings and limits.
 * @param pr Preconditioner storage.
 * @param out The output structure that will store the results, including residual history and final solution.
 * 
 * @return The solution vector after performing the CG s-step method, stored in `out->sol_local`.
 * 
 * @note The final solution is returned in the output structure `out->sol_local`, and the result of the CG s-step
 *       method is captured in `out->retv`, which indicates the status of the solver.
 * 
 * @details The function acts as a wrapper to invoke the `cgsstep` function, providing an interface for the user
 *          to solve the system. The initial guess `x0_loc` is updated in-place with the final solution, and the
 *          output structure contains additional information, such as the residual history and exit conditions.
 */
vector<vtype>* solve_cgs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, cgsprec* pr, const params& p, SolverOut* out);
