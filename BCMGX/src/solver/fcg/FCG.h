/**
 * @file FCG.h
 * @brief Implementation of a variant of the flexible Conjugate Gradient (CG) method
 * that also allows for variable preconditioners across iterations.
 *
 * The variant is achieved by reorganizing the main operations to aggregate scalar
 * products and reduce the required global synchronizations to one.
 *
 * This implementation is based on the following references:
 * - Yvan Notay, Artem Napov, "A massively parallel solver for discrete Poisson-like problems",
 *   Journal of Computational Physics, Volume 281, 2015, Pages 237-250. 
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/prec_apply.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"

/**
 * @brief Solves a linear system using the Flexible Conjugate Gradient method (v3).
 * 
 * This function wraps the flexible conjugate gradient method, calling it and managing
 * the solution process. It stores the solution in `x0` and provides a copy in the return value.
 * 
 * @param h Handles for CUDA streams and various resources.
 * @param Alocal The matrix representing the linear system (CSR format).
 * @param rhs The right-hand side vector.
 * @param x0 The initial guess for the solution (input and output).
 * @param pr Actual preconditioner data (such as preconditioner type and internal structures).
 * @param p User-defined solver and preconditioner settings (like iteration limits, stopping criteria, etc.).
 * @param out Output structure to store results, including convergence history and other statistics.
 * @return vector<vtype>* The solution vector.
 */
vector<vtype>* solve_fcg(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x0, cgsprec* pr, const params& p, SolverOut* out);
