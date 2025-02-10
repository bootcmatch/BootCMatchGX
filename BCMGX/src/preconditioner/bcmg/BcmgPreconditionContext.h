/** @file */
#pragma once

#include "datastruct/vector.h"
#include "preconditioner/bcmg/AMG.h"

/**
 * @struct BcmgPreconditionContext
 * @brief Stores the context for the BCMG preconditioner.
 */
struct BcmgPreconditionContext {
    vectorCollection<vtype>* RHS_buffer; /**< Buffer for the right-hand side vectors */

    vectorCollection<vtype>* Xtent_buffer_local; /**< Buffer for local extended vectors */
    vectorCollection<vtype>* Xtent_buffer_2_local; /**< Second buffer for local extended vectors */

    hierarchy* hrrch; /**< Pointer to the AMG hierarchy */
    int max_level_nums; /**< Maximum number of levels in the hierarchy */
    itype* max_coarse_size; /**< Array storing the maximum coarse size per level */
};

/**
 * @namespace Bcmg
 * @brief Namespace containing functions for BCMG preconditioner operations.
 */
namespace Bcmg {

/** @brief Preconditioner context instance */
extern BcmgPreconditionContext context;

/**
 * @brief Initializes the preconditioner context.
 * @param hrrch Pointer to the hierarchy structure.
 */
void initPreconditionContext(hierarchy* hrrch);

/**
 * @brief Adjusts the hierarchy buffer size based on the current hierarchy state.
 * @param hrrch Pointer to the hierarchy structure.
 */
void setHrrchBufferSize(hierarchy* hrrch);

/**
 * @brief Frees memory allocated for the preconditioner context.
 */
void freePreconditionContext();
}
