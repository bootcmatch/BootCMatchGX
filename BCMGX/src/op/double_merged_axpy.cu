#include "double_merged_axpy.h"

#include "utility/cudamacro.h"

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

void double_merged_axpy(vector<vtype>* x0, vector<vtype>* x1, vector<vtype>* y, vtype alpha_0, vtype alpha_1, itype n, itype shift)
{
    PUSH_RANGE(__func__, 4)

    GridBlock gb = gb1d(n, BLOCKSIZE);
    _double_merged_axpy<<<gb.g, gb.b>>>(n, x0->val, x1->val, y->val, alpha_0, alpha_1, shift);

    POP_RANGE
}
