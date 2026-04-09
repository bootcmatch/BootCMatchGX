/**
 * @file
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

struct Preconditioner;

/**
 * @struct BcmgData
 * @brief Structure containing data required for the BCMG preconditioner.
 */
struct BcmgData {
    bootBuildData* bootamg_data = NULL; /**< Pointer to bootstrap AMG setup data */
    boot* boot_amg = NULL; /**< Pointer to the bootstrap AMG structure */
    applyData* amg_cycle = NULL; /**< Pointer to AMG cycle application data */
    hierarchy* H = NULL; /**< Pointer to the hierarchy structure */

    int max_num_levels = 0;
    vectorCollection<vtype>* RHS = NULL;
    vectorCollection<vtype>* Xtent_local = NULL;
    vectorCollection<vtype>* Xtent_2_local = NULL;
};

/**
 * @brief Sets up the BCMG preconditioner.
 * @param h Pointer to the handles structure.
 * @param Alocal Pointer to the local CSR matrix.
 * @param pr Pointer to the preconditioner structure.
 * @param ip Reference to the input parameters.
 */
void bcmg_setup(handles* h, CSR* Alocal, Preconditioner* pr, const InputParameters& ip);

/**
 * @brief Applies the BCMG preconditioner.
 * @param h Pointer to the handles structure.
 * @param Alocal Pointer to the local CSR matrix.
 * @param rhs Pointer to the right-hand side vector.
 * @param x Pointer to the solution vector.
 * @param pr Pointer to the preconditioner structure.
 * @param ip Reference to the input parameters.
 */
void bcmg_apply(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x, Preconditioner* pr, const InputParameters& ip);

/**
 * @brief Finalizes and frees resources associated with the BCMG preconditioner.
 * @param Alocal Pointer to the local CSR matrix.
 * @param pr Pointer to the preconditioner structure.
 * @param p Reference to AMG parameters.
 */
void bcmg_finalize(CSR* Alocal, Preconditioner* pr, const InputParameters& p);
