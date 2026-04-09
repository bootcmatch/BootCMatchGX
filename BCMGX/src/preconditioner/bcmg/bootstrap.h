/** @file */
#pragma once

#include "config/Params.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

struct Preconditioner;

/**
 * @namespace Bootstrap
 * @brief Contains functions related to the bootstrap AMG method.
 */
namespace Bootstrap {

// /**
//  * @brief Performs inner iterations in the bootstrap AMG method.
//  * @param h Pointer to the handles structure.
//  * @param bootamg_data Pointer to bootstrap AMG setup data.
//  * @param boot_amg Pointer to the bootstrap AMG structure.
//  * @param amg_cycle Pointer to the AMG cycle application data.
//  */
// void innerIterations(handles* h, bootBuildData* bootamg_data, boot* boot_amg, applyData* amg_cycle);

/**
 * @brief Constructs and applies the bootstrap AMG method.
 * @param h Pointer to the handles structure.
 * @return Pointer to the constructed bootstrap AMG object.
 */
boot* bootstrap(handles* h, Preconditioner* pr, const InputParameters& ip);

}
