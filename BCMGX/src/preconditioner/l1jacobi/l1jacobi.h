/**
 * @file
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/PrecOut.h"
#include "utility/handles.h"

struct cgsprec;

/**
 * @struct L1JacobiData
 * @brief Struct for holding data required for the L1 Jacobi preconditioner.
 * 
 * This struct contains pointers to the vectors that hold the intermediate data for the L1 Jacobi preconditioning process,
 * including the preconditioner vector `pl1j`, the local working vector `w_loc`, and the locally copied residual vector `rcopy_loc`.
 */
struct L1JacobiData {
    vector<vtype>* pl1j = NULL; /**< Pointer to the preconditioner vector used in the L1 Jacobi method. */
    vector<vtype>* w_loc = NULL; /**< Pointer to the local working vector for intermediate computations. */
    vector<vtype>* rcopy_loc = NULL; /**< Pointer to the locally copied residual vector for Jacobi updates. */
};

/**
 * @brief Performs `k` iteration of the L1 Jacobi preconditioning method.
 * 
 * This function performs `k` iterations of the L1 Jacobi preconditioner. The method is used in iterative solvers to approximate 
 * the solution of a linear system by updating the solution vector using the diagonal scaling approach.
 * 
 * @param h A pointer to the handle containing solver context (CUDA, MPI, etc.).
 * @param k The number of iterations in the L1 Jacobi method.
 * @param A A pointer to the sparse matrix \( A \) in CSR format.
 * @param D A pointer to the diagonal vector \( D \), used in the Jacobi iteration.
 * @param f A pointer to the right-hand side vector \( f \) of the linear system.
 * @param u A pointer to the current approximation of the solution vector.
 * @param u_ A pointer to the updated solution vector after applying the Jacobi iteration.
 * 
 * @note This function modifies the solution vector \( u \) using the L1 Jacobi method.
 */
void l1jacobi_iter(handles* h,
    int k,
    CSR* A,
    vector<vtype>* D,
    vector<vtype>* f, // rhs
    vector<vtype>* u, // Xtent
    vector<vtype>** u_ // Xtent2
);

/**
 * @brief Sets up the L1 Jacobi preconditioner.
 * 
 * This function initializes the L1 Jacobi preconditioner by preparing necessary data structures 
 * and performing any required setup tasks such as computing diagonal preconditioner vectors.
 * 
 * @param h A pointer to the handle containing solver context (CUDA, MPI, etc.).
 * @param Alocal A pointer to the local matrix \( A \) in CSR format.
 * @param pr A pointer to the preconditioner data structure.
 * @param p The solver parameters used to configure the setup.
 * 
 * @note This function should be called before applying the preconditioner.
 */
void l1jacobi_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p);

/**
 * @brief Applies the L1 Jacobi preconditioner.
 * 
 * This function applies the L1 Jacobi preconditioner to a given residual vector \( r_{\text{loc}} \) and updates the solution 
 * vector \( u_{\text{loc}} \) accordingly. The preconditioner approximates the solution by diagonal scaling.
 * 
 * @param h A pointer to the handle containing solver context (CUDA, MPI, etc.).
 * @param Alocal A pointer to the local matrix \( A \) in CSR format.
 * @param r_loc A pointer to the local residual vector \( r_{\text{loc}} \).
 * @param u_loc A pointer to the local solution vector \( u_{\text{loc}} \).
 * @param pr A pointer to the preconditioner data structure.
 * @param p The solver parameters used to configure the preconditioning step.
 * @param out A pointer to the output structure where the results of the preconditioning step are stored.
 * 
 * @note This function is part of the iterative solver cycle and should be called during each iteration.
 */
void l1jacobi_apply(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, cgsprec* pr, const params& p, PrecOut* out);

/**
 * @brief Finalizes the L1 Jacobi preconditioner setup.
 * 
 * This function finalizes the L1 Jacobi preconditioning process, cleaning up resources and performing any required 
 * finalization tasks such as freeing memory or resetting data structures.
 * 
 * @param Alocal A pointer to the local matrix \( A \) in CSR format.
 * @param pr A pointer to the preconditioner data structure.
 * @param p The solver parameters used to configure the finalization step.
 * 
 * @note This function should be called after the preconditioner has been applied to release any allocated resources.
 */
void l1jacobi_finalize(CSR* Alocal, cgsprec* pr, const params& p);
