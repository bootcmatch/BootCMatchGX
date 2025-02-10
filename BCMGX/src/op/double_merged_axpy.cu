/** @file */

#include "double_merged_axpy.h"

#include "utility/cudamacro.h"
#include "utility/profiling.h"

/**
 * @brief Performs a double merged AXPY operation in parallel.
 *
 * This kernel computes the following operation for each element:
 * \[
 * x2[i + \text{shift}] = \alpha_1 \cdot (\alpha_0 \cdot x0[i + \text{shift}] + x1[i + \text{shift}]) + x2[i + \text{shift}]
 * \]
 * and updates `x1` with the intermediate result:
 * \[
 * x1[i + \text{shift}] = \alpha_0 \cdot x0[i + \text{shift}] + x1[i + \text{shift}]
 * \]
 *
 * @param n The number of elements to process.
 * @param x0 Pointer to the first input vector.
 * @param x1 Pointer to the second input vector, which will be updated.
 * @param x2 Pointer to the output vector, which will be updated.
 * @param alpha_0 The scalar multiplier for the first input vector.
 * @param alpha_1 The scalar multiplier for the intermediate result.
 * @param shift The shift index for accessing the vectors.
 */
__global__ void _double_merged_axpy(itype n, vtype* x0, vtype* x1, vtype* x2, vtype alpha_0, vtype alpha_1, itype shift)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    vtype xi1_local = alpha_0 * x0[i + shift] + x1[i + shift];
    x2[i + shift] = alpha_1 * xi1_local + x2[i + shift];
    x1[i + shift] = xi1_local;
}

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
void double_merged_axpy(vector<vtype>* x0, vector<vtype>* x1, vector<vtype>* y, vtype alpha_0, vtype alpha_1, itype n, itype shift)
{
    BEGIN_PROF(__FUNCTION__);

    GridBlock gb = gb1d(n, BLOCKSIZE);
    _double_merged_axpy<<<gb.g, gb.b>>>(n, x0->val, x1->val, y->val, alpha_0, alpha_1, shift);
    cudaDeviceSynchronize();

    END_PROF(__FUNCTION__);
}
