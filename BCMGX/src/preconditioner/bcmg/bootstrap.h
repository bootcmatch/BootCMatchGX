/** @file */
#pragma once

#include "config/Params.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

/**
 * @namespace Bootstrap
 * @brief Contains functions related to the bootstrap AMG method.
 */
namespace Bootstrap {

/**
 * @brief Performs inner iterations in the bootstrap AMG method.
 * @param h Pointer to the handles structure.
 * @param bootamg_data Pointer to bootstrap AMG setup data.
 * @param boot_amg Pointer to the bootstrap AMG structure.
 * @param amg_cycle Pointer to the AMG cycle application data.
 */
void innerIterations(handles* h, bootBuildData* bootamg_data, boot* boot_amg, applyData* amg_cycle);

/**
 * @brief Constructs and applies the bootstrap AMG method.
 * @param h Pointer to the handles structure.
 * @param bootamg_data Pointer to bootstrap AMG setup data.
 * @param apply_data Pointer to AMG cycle application data.
 * @param p Reference to AMG parameters.
 * @return Pointer to the constructed bootstrap AMG object.
 */
boot* bootstrap(handles* h, bootBuildData* bootamg_data, applyData* apply_data, const params& p /*, bool precondition_flag*/);

}
