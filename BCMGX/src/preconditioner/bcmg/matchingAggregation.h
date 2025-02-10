/** @file */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

hierarchy* adaptiveCoarsening(handles* h, buildData* amg_data, const params& p);

/**
 * @brief Prepares relaxation data structures for a given level in the hierarchy.
 * 
 * @param h CUDA Handles.
 * @param level The level in the AMG hierarchy.
 * @param A The system matrix in CSR format.
 * @param hrrch The AMG hierarchy structure.
 * @param amg_data The AMG build data.
 */
void relaxPrepare(handles* h, int level, CSR* A, hierarchy* hrrch, buildData* amg_data);
