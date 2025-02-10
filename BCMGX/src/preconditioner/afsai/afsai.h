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
 * @struct AfsaiData
 * @brief Data structure holding the matrices and vectors used for the Afsai method.
 * 
 * This structure is used to store the CSR (Compressed Sparse Row) matrices `Aprec` and `AprecT`, 
 * as well as a dummy vector `v_d_dummy` used in the Afsai method computations.
 * It helps to manage and organize the data that is involved in the Afsai method process.
 */
struct AfsaiData {
    CSR* Aprec = NULL; ///< Pointer to the CSR matrix representing the preconditioner for A.
    CSR* AprecT = NULL; ///< Pointer to the transpose of the CSR matrix Aprec.
    vector<double>* v_d_dummy = NULL; ///< Pointer to a dummy vector used in computations.
};

/**
 * @brief Set up the Afsai preconditioner.
 * 
 * This function initializes the Afsai preconditioner by setting up necessary matrices and vectors,
 * which are later used in the Afsai method for solving linear systems. It prepares the data 
 * structures required for efficient computation.
 *
 * @param h A pointer to the handle containing solver context (CUDA, MPI, etc.).
 * @param Alocal A pointer to the local matrix \( A \) in CSR format.
 * @param pr A pointer to the preconditioner data structure.
 * @param p The solver parameters used to configure the setup.
 */
void afsai_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p);

/**
 * @brief Apply the Afsai preconditioner.
 * 
 * This function applies the Afsai preconditioner to the input vector `v_d_r` and modifies it 
 * according to the preconditioned value in `v_d_pr`. It uses the matrix and vectors 
 * set up by `afsai_setup` and performs the preconditioning step of the method.
 * The result is written into the output vector `v_d_pr`.
 *
 * @param h A pointer to the handle containing solver context (CUDA, MPI, etc.).
 * @param Alocal A pointer to the local CSR matrix representing the system.
 * @param v_d_r A pointer to the right-hand side vector of the system, before preconditioning.
 * @param v_d_pr A pointer to the vector that will hold the preconditioned result.
 * @param pr A pointer to a `cgsprec` structure which holds information about the preconditioner.
 * @param p The parameter struct containing additional parameters used during the application of the preconditioner.
 * @param out A pointer to a `PrecOut` structure that will contain output data related to the preconditioning process.
 */
void afsai_apply(handles* h, CSR* Alocal, vector<double>* v_d_r, vector<double>* v_d_pr, cgsprec* pr, const params& p, PrecOut* out);

/**
 * @brief Finalize the Afsai preconditioner.
 * 
 * This function performs any necessary cleanup or final steps needed after the Afsai preconditioner
 * has been applied. It releases or finalizes any memory or configurations that were used during the 
 * setup or application phases.
 *
 * @param Alocal A pointer to the local CSR matrix representing the system.
 * @param pr A pointer to a `cgsprec` structure which holds information about the preconditioner.
 * @param p The parameter struct containing additional parameters needed for the finalization.
 */
void afsai_finalize(CSR* Alocal, cgsprec* pr, const params& p);
