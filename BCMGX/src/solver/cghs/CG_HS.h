/**
 * @file CG_HS.h
 * @brief Implementation of the Conjugate Gradient (CG) method using the Hestenes-Stiefel approach.
 *
 * This implementation is based on the following references:
 * - Hestenes, M., Stiefel, E., "Methods of Conjugate Gradients for Solving Linear Systems",
 *   Journal of Research of the National Bureau of Standards, vol. 49, n. 6, 1952, pp. 409-436.
 * - Y. Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed., SIAM, 2003.
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/prec_apply.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"

/**
 * @brief Solves a linear system using CG-HS and returns the solution.
 *
 * @param h Handles for CUDA streams and various resources.
 * @param Alocal Pointer to the sparse matrix in CSR format.
 * @param rhs_loc Right-hand side vector.
 * @param x0_loc Initial solution guess.
 * @param p User-defined solver and preconditioner settings (like iteration limits, stopping criteria, etc.).
 * @param pr Actual preconditioner data (such as preconditioner type and internal structures).
 * @param out Solver output structure.
 * @return The computed solution vector.
 */
vector<vtype>* solve_cg_hs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, cgsprec* pr, const params& p, SolverOut* out);
