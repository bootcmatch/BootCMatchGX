/**
 * @file
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"

struct cgsprec;

/**
 * @brief Solves a linear system using the specified solver type.
 * 
 * This function selects the appropriate solver (e.g., Conjugate Gradient, Flexible Conjugate Gradient,
 * Pipelined Conjugate Gradient, etc.) based on the solver type specified in the solver parameters.
 * It invokes the corresponding solver function and returns the solution.
 * 
 * @param h Handle to CUDA and other resources.
 * @param Alocal The matrix representing the linear system (CSR format).
 * @param rhs The right-hand side vector of the system.
 * @param x0 The initial guess for the solution (input and output).
 * @param p Solver parameters, including the solver type and other settings.
 * @param pr Preconditioner configuration (if any).
 * @param out Output structure to store results, including convergence history and other statistics.
 * @return vector<vtype>* The solution vector to the system \(A x = b\).
 * 
 * @note This function handles different solver types, including CGHS, FCG, CGS, PIPELINED_CGS, and CGS_CUBLAS.
 * If an unrecognized solver type is provided, the program will exit with an error message.
 */
vector<vtype>* solve(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x0, const params& p, cgsprec& pr, SolverOut* out);
