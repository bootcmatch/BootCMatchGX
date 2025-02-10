/** @file */
#pragma once

#include "utility/setting.h"

/**
 * @brief Performs element-wise division of two vectors.
 *
 * This kernel computes the element-wise division of two vectors `a` and `b`
 * and stores the result in vector `c`. Specifically, it computes:
 * \[
 * c[i] = \frac{a[i]}{b[i]}
 * \]
 *
 * @tparam T The data type of the vectors (e.g., float, double).
 * @param n The number of elements in the vectors.
 * @param a Pointer to the first input vector.
 * @param b Pointer to the second input vector.
 * @param c Pointer to the output vector where the results will be stored.
 */
template <typename T>
__global__ void _diagScal(itype n, T* a, T* b, T* c)
{
    stype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    c[i] = a[i] / b[i];
}

/**
 * @brief Wrapper function to launch the diagonal scaling kernel.
 *
 * This function prepares the data and launches the CUDA kernel `_diagScal`
 * to perform element-wise division of the input vectors `a` and `b`, storing
 * the result in vector `c`.
 *
 * @tparam T The data type of the vectors (e.g., float, double).
 * @param a Pointer to the first input vector.
 * @param b Pointer to the second input vector.
 * @param c Pointer to the output vector where the results will be stored.
 */
template <typename T>
void diagScal(vector<T>* a, vector<T>* b, vector<T>* c)
{
    assert(a->n == b->n);
    GridBlock gb = gb1d(a->n, BLOCKSIZE);
    _diagScal<<<gb.g, gb.b>>>(a->n, a->val, b->val, c->val);
}
