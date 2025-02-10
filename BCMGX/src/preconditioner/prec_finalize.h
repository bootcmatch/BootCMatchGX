/**
 * @file
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "preconditioner/prec_setup.h"
#include "utility/handles.h"

/**
 * @brief Finalizes and deallocates resources used by the selected preconditioner.
 * 
 * This function ensures proper cleanup of any allocated resources associated with 
 * the preconditioning method specified in the `cgsprec` structure. It calls the 
 * corresponding finalization function for the selected preconditioner type.
 * 
 * The function also profiles execution time to track performance.
 * 
 * @param Alocal A pointer to the local CSR matrix representing the system.
 * @param pr A pointer to a `cgsprec` structure containing the preconditioner type and related data.
 * @param p The parameter struct containing additional solver parameters.
 * @param out A pointer to a `PrecOut` structure (not used in this function but included for consistency).
 * 
 * @note The function does not perform any operations if `PreconditionerType::NONE` is selected.
 * 
 * @warning If an unsupported `PreconditionerType` is encountered, an error message is printed, and the program exits.
 */
void prec_finalize(CSR* Alocal, cgsprec* pr, const params& p, PrecOut* out);
