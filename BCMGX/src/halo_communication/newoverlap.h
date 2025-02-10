/**
 * @file newoverlap.h 
 * @brief Implements the detection and handling of overlapping rows in a sparse matrix (CSR format).
 * 
 * This module identifies rows in a sparse matrix that reference elements outside 
 * a given local domain, distinguishing between "local" and "needy" rows. CUDA 
 * parallelism is utilized for efficient row classification.
 */

#pragma once

#include "datastruct/CSR.h"
   
/**
 * @brief Sets up overlapped regions in a CSR matrix.
 * 
 * This function categorizes matrix rows into "local" and "needy" categories
 * based on whether they reference external indices.
 * 
 * @param A Pointer to the CSR matrix.
 */
void setupOverlapped(CSR* A);
