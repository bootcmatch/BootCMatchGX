#pragma once

#include "utility/setting.h"

template <typename T>
__global__ void _diagScal(itype n, T* a, T* b, T* c)
{
    stype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    c[i] = a[i] / b[i];
}

template <typename T>
void diagScal(vector<T>* a, vector<T>* b, vector<T>* c)
{
    assert(a->n == b->n);
    GridBlock gb = gb1d(a->n, BLOCKSIZE);
    _diagScal<<<gb.g, gb.b>>>(a->n, a->val, b->val, c->val);
}
