/** @file */
#pragma once

#define RTOL 0.25

#include "datastruct/vector.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"
#include "utility/setting.h"

namespace GAMGcycle {

/** @brief Buffer for storing residuals in the multigrid cycle. */
extern vector<vtype>* Res_buffer;

/**
 * @brief Initializes the context by allocating the residual buffer.
 * @param n Size of the buffer.
 */
void initContext(int n);

/**
 * @brief Sets the buffer size for residual storage.
 * @param n New size of the buffer.
 */

__inline__ void setBufferSize(itype n);

/**
 * @brief Frees the allocated residual buffer.
 */
void freeContext();
}

/**
 * @brief Performs a recursive GAMG cycle for solving a linear system.
 * 
 * @param h CUDA stream handles.
 * @param k Current hierarchy index.
 * @param bootamg_data Bootstrap AMG data.
 * @param boot_amg Bootstrap AMG structure.
 * @param amg_cycle AMG cycle parameters.
 * @param Rhs Right-hand side vector collection.
 * @param Xtent Solution vector collection.
 * @param Xtent_2 Auxiliary solution vector collection.
 * @param l Current multigrid level.
 */
void GAMG_cycle(handles* h, int k, bootBuildData* bootamg_data, boot* boot_amg, applyData* amg_cycle, vectorCollection<vtype>* Rhs, vectorCollection<vtype>* Xtent, vectorCollection<vtype>* Xtent_2, int l /*, int coarsesolver_type*/);
