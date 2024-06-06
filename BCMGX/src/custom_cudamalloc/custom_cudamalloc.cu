#include "custom_cudamalloc.h"

#include "utility/cudamacro.h"
#include "utility/function_cnt.h"
#include "utility/mpi.h"
#include "utility/setting.h"
#include "utility/utils.h"

#include <assert.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

customCudaMalloc g_buff_solver, g_buff_scratch, g_buff_nsparse;

namespace CustomCudaMalloc {
void init(int n_itypes, int n_vtypes, int buff_num)
{
    customCudaMalloc* g_buff;
    switch (buff_num) {
    case 1:
        g_buff = &g_buff_scratch;
        break;
    case 2:
        g_buff = &g_buff_nsparse;
        break;
    default:
        g_buff = &g_buff_solver;
    }
    cudaMalloc_CNT
        CHECK_DEVICE(cudaMalloc((void**)&(g_buff->itype_ptr), sizeof(itype) * n_itypes));
    cudaMalloc_CNT
        CHECK_DEVICE(cudaMalloc((void**)&(g_buff->vtype_ptr), sizeof(vtype) * n_vtypes));

    g_buff->itype_allocated = n_itypes;
    g_buff->vtype_allocated = n_vtypes;

    g_buff->itype_occupied = 0;
    g_buff->vtype_occupied = 0;
}

itype* alloc_itype(int n_elements, int buff_num)
{
    customCudaMalloc* g_buff;
    switch (buff_num) {
    case 1:
        g_buff = &g_buff_scratch;
        break;
    case 2:
        g_buff = &g_buff_nsparse;
        break;
    default:
        g_buff = &g_buff_solver;
    }
    assert(g_buff->itype_allocated >= g_buff->itype_occupied + n_elements);

    itype* ptr = g_buff->itype_ptr + g_buff->itype_occupied;
    g_buff->itype_occupied += n_elements;
    return ptr;
}

vtype* alloc_vtype(int n_elements, int buff_num)
{
    customCudaMalloc* g_buff;
    switch (buff_num) {
    case 1:
        g_buff = &g_buff_scratch;
        break;
    case 2:
        g_buff = &g_buff_nsparse;
        break;
    default:
        g_buff = &g_buff_solver;
    }
    assert(g_buff->vtype_allocated >= g_buff->vtype_occupied + n_elements);

    vtype* ptr = g_buff->vtype_ptr + g_buff->vtype_occupied;
    g_buff->vtype_occupied += n_elements;
    return ptr;
}

void free(int buff_num)
{
    _MPI_ENV;

    const char* s;
    customCudaMalloc* g_buff;
    switch (buff_num) {
    case 1:
        s = "scratch (id 1)";
        g_buff = &g_buff_scratch;
        break;
    case 2:
        s = "nsparse (id 2)";
        g_buff = &g_buff_nsparse;
        break;
    default:
        s = "solver (id 0)";
        g_buff = &g_buff_solver;
    }
    if (ISMASTER) {
        printf("Custom Cuda Malloc %s\n\tn_itypes alloc: %u, n_itypes occup: %u\n\tn_vtypes alloc: %u, n_vtypes occup: %u\n", s,
            g_buff->itype_allocated, g_buff->itype_occupied, g_buff->vtype_allocated, g_buff->vtype_occupied);
    }

    MY_CUDA_CHECK(cudaFree(g_buff->itype_ptr));
    MY_CUDA_CHECK(cudaFree(g_buff->vtype_ptr));

    g_buff->itype_ptr = NULL;
    g_buff->vtype_ptr = NULL;
    g_buff->itype_allocated = 0;
    g_buff->vtype_allocated = 0;
    g_buff->itype_occupied = 0;
    g_buff->vtype_occupied = 0;
}
}
