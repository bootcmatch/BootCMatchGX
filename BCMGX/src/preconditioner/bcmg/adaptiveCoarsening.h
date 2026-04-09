#pragma once

#include "config/Params.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

/**
 * @brief Function for adaptive coarsening in a multilevel solver hierarchy.
 *
 * This function builds the multilevel hierarchy by performing adaptive coarsening based on the AMG (Algebraic Multigrid) method.
 * It allocates memory for various buffers, sets up the communication patterns for the solver, and computes the prolongation
 * and restriction operators for the AMG hierarchy.
 *
 * @param h A pointer to the `handles` structure which contains solver-related data.
 * @param amg_data A pointer to the `buildData` structure which contains the matrix and related data.
 * @param p A reference to the `InputParameters` structure which holds solver parameters such as memory allocation size and preconditioner type.
 *
 * @return A pointer to the `hierarchy` structure which holds the multilevel hierarchy and its components.
 *
 * @note This function also involves device memory management for CUDA-based operations, and it manages communication patterns
 *       when multiple processes are involved.
 */
hierarchy* adaptiveCoarsening(handles* h, buildData* amg_data, const InputParameters& p);