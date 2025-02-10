/**
 * @file
 */
#pragma once

#include "config/Params.h"
#include "datastruct/vector.h"
#include "preconditioner/PrecOut.h"

/**
 * @brief Structure to store the results of the solver execution.
 * 
 * This structure contains the output information from the iterative solver process, including
 * the solution vector, residual history, number of iterations, exit status, and preconditioner-related data.
 */
struct SolverOut {
    /**
     * @brief Return value indicating the status of the solver.
     * 
     * A value of 0 indicates successful completion, while other values indicate errors or exceptional conditions.
     */
    int retv = 0;

    /**
     * @brief Local solution vector.
     * 
     * This vector holds the computed solution for the local part of the problem. It may be copied
     * from a global solution vector after MPI reductions.
     */
    vector<vtype>* sol_local;

    /**
     * @brief Residual history.
     * 
     * This array stores the history of the residual norms at each iteration, useful for monitoring convergence.
     */
    vtype* resHist;

    /**
     * @brief Final residual at the end of the solver execution.
     * 
     * This value represents the residual after the final iteration or when convergence is reached.
     */
    vtype exitRes;

    /**
     * @brief Initial residual norm.
     * 
     * The initial residual norm (delta0) used as a reference for convergence criteria.
     */
    vtype del0;

    /**
     * @brief Number of iterations performed by the solver.
     * 
     * This field stores the total number of iterations executed during the solver's run.
     */
    int niter;

    /**
     * @brief Total size of the problem (global size).
     * 
     * The global number of unknowns in the system, corresponding to the total number of rows (of the initial CSR).
     */
    gstype full_n;

    /**
     * @brief Local size of the problem.
     * 
     * The local number of unknowns, corresponding to the number of rows (of the initial CSR) handled by the current process.
     */
    stype local_n;

    /**
     * @brief Preconditioner output data.
     * 
     * This structure contains the results of applying the preconditioner during the solver process.
     */
    PrecOut precOut {};
};

struct cgsprec;

void dump(const char* filename, const params& p, const cgsprec& pr, SolverOut* out);
