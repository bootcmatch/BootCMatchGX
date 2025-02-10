/**
 * @file
 */
#pragma once

#include "datastruct/vector.h"

/**
 * @brief Computes the vector-matrix product for the given vector and preconditioner.
 * 
 * This function computes a portion of the vector-matrix product by calculating dot products between
 * the input vectors and stores the result in the `vm` vector. It optionally applies a preconditioner
 * depending on the `use_prec` flag. The resulting values are then reduced across all MPI processes.
 *
 * @param sP The vector `sP` that contains the data used for computing the vector-matrix product.
 * @param s The size parameter controlling the portion of `sP` to be used in the computation.
 * @param r_loc The local vector `r_loc` used in the dot product calculations.
 * @param vm The vector `vm` where the result of the vector-matrix product is stored.
 * @param use_prec A flag indicating whether or not to apply a preconditioner in the computation.
 * 
 * @note The preconditioner is only applied if `use_prec` is true.
 * 
 * @details This function computes the following:
 * - If `use_prec == 0`: The dot products `vm(1:s+1)` and `vm(s+1:2s)` are computed without preconditioning.
 * - If `use_prec == 1`: The dot products `vm(1:s)` and `vm(s+1:2s)` are computed with preconditioning.
 * The results are then reduced over all MPI processes using `MPI_Allreduce` to gather the final values.
 */
void computevm(vector<vtype>* sP, int s, vector<vtype>* r_loc, vectordh<vtype>* vm, bool use_prec);
