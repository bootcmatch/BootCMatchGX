/** @file */
#pragma once

#include "datastruct/vector.h"

/**
 * @brief Wrapper function to launch the double merged AXPY kernel.
 *
 * This function prepares the data and launches the CUDA kernel `_double_merged_axpy`
 * to perform the double merged AXPY operation on the input vectors.
 *
 * @param x0 Pointer to the first input vector.
 * @param x1 Pointer to the second input vector, which will be updated.
 * @param y Pointer to the output vector, which will be updated.
 * @param alpha_0 The scalar multiplier for the first input vector.
 * @param alpha_1 The scalar multiplier for the intermediate result.
 * @param n The number of elements to process.
 * @param shift The shift index for accessing the vectors.
 */
void double_merged_axpy(vector<vtype>* x0, vector<vtype>* x1, vector<vtype>* y, vtype alpha_0, vtype alpha_1, itype n, itype shift);
