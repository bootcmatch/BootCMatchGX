/**
 * @file
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/afsai/afsai.h"
#include "preconditioner/bcmg/bcmg.h"
#include "preconditioner/l1jacobi/l1jacobi.h"
#include "utility/handles.h"

/**
 * @brief Structure to store preconditioner data.
 * 
 * This structure holds the necessary data for different types of preconditioners.
 * The specific preconditioner used is determined by the `ptype` field.
 * 
 * @note When a particular preconditioner is selected, only its corresponding data 
 *       structure is used, while others remain unused.
 */
struct cgsprec {
    PreconditionerType ptype;

    // l1-Jacobi
    L1JacobiData l1jacobi;

    // BCMG
    BcmgData bcmg;

    // add here vars required by different precs
    AfsaiData afsai;
};

/**
 * @brief Sets up the selected preconditioner for use in iterative solvers.
 * 
 * This function initializes and configures the preconditioner specified in the 
 * `cgsprec` structure. It calls the appropriate setup function based on the 
 * selected preconditioner type.
 * 
 * The function also profiles execution time to track performance.
 * 
 * @param h A pointer to the solver's handle structure.
 * @param Alocal A pointer to the local CSR matrix representing the system.
 * @param pr A pointer to a `cgsprec` structure containing the preconditioner type and related data.
 * @param p The parameter struct containing additional solver parameters.
 * @param out A pointer to a `PrecOut` structure (not used in this function but included for consistency).
 * 
 * @note If `PreconditionerType::NONE` is selected, no setup operations are performed.
 * 
 * @warning If an unsupported `PreconditionerType` is encountered, an error message is printed, and the program exits.
 */
void prec_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p, PrecOut* out);
