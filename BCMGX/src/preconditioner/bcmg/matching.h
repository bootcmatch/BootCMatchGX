/** @file */
#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

/**
 * @brief Applies the suitor algorithm to a given matrix in the AMG setup.
 * 
 * @param h CUDA Handles.
 * @param amg_data AMG build data structure.
 * @param A Input matrix in CSR format.
 * @param w Weight vector.
 * @return A vector of indices representing the suitor set.
 */
vector<itype>* suitor(handles* h, buildData* amg_data, CSR* A, vector<vtype>* w);
