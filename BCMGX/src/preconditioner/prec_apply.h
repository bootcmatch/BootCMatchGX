/**
 * @file
 */
#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/PrecOut.h"
#include "preconditioner/prec_setup.h"

/**
 * @brief Applies the selected preconditioner to the input vector.
 * 
 * This function applies a preconditioning method to the residual vector `r_loc` and 
 * stores the preconditioned result in `u_loc`. The preconditioner type is determined
 * by the `ptype` member of the `cgsprec` structure.
 * 
 * The function profiles execution time and ensures synchronization of CUDA operations.
 * 
 * @param h The handle for the computational environment, including memory and configurations.
 * @param Alocal A pointer to the local CSR matrix representing the system.
 * @param r_loc A pointer to the vector representing the residual before preconditioning.
 * @param u_loc A pointer to the vector that will store the preconditioned result.
 * @param pr A pointer to a `cgsprec` structure, which holds information about the selected preconditioner.
 * @param p The parameter struct containing additional solver parameters.
 * @param out A pointer to a `PrecOut` structure that will store output data related to the preconditioning process.
 * 
 * @note The function performs a CUDA device synchronization (`cudaDeviceSynchronize()`) before returning.
 * 
 * @warning If an unsupported `PreconditionerType` is encountered, the function prints an error message and exits the program.
 */
void prec_apply(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, cgsprec* pr, const params& p, PrecOut* out);
