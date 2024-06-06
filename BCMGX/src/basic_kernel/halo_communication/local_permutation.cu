#include "basic_kernel/halo_communication/local_permutation.h"
#include "custom_cudamalloc/custom_cudamalloc.h"
#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "utility/distribuite.h"
#include "utility/function_cnt.h"
#include "utility/handles.h"
#include "utility/mpi.h"
#include "utility/setting.h"
#include "utility/utils.h"

#include <assert.h>
#include <getopt.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#define DIE                                \
    CHECK_DEVICE(cudaDeviceSynchronize()); \
    MPI_Finalize();                        \
    exit(0);

// --------------- only for KDevelop ----------------------
#include <curand_mtgp32_kernel.h>
// --------------------------------------------------------

#define MAX_NNZ_PER_ROW_LAP 5
#define MPI 1
#ifndef __GNUC__
typedef int (*__compar_fn_t)(const void*, const void*);
#endif

// ----------------------------------------- New Version ------------------------------------------

__global__ void apply_mask_permut_GPU_noSideEffects_glob(itype nnz, const itype* col, int shrinking_permut_len, const itype* shrinking_permut, itype* comp_col)
{

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // int number_of_permutations = shrinking_permut_len, start, med, end, flag;
    int number_of_permutations = shrinking_permut_len, start, med, end;

    if (id < nnz) {
        start = 0;
        end = number_of_permutations;

        while (start < end) {
            med = (end + start) / 2;
            if (col[id] > shrinking_permut[med]) {
                start = med + 1;
            } else {
                end = med;
            }
        }
        comp_col[id] = start;
    }

    return;
}

vector<itype>* apply_mask_permut_GPU_noSideEffects(const CSR* Alocal, const vector<itype>* shrinking_permut)
{
    assert(Alocal->on_the_device);
    assert(shrinking_permut->on_the_device);

    // ------------- custom cudaMalloc -------------
    vector<itype>* comp_col;
    if (Alocal->custom_alloced) {
        comp_col = Vector::init<itype>(Alocal->nnz, false, true);
        comp_col->val = CustomCudaMalloc::alloc_itype(Alocal->nnz);
    } else {
        Vectorinit_CNT
            comp_col
            = Vector::init<itype>(Alocal->nnz, true, true);
    }
    // ---------------------------------------------

    GridBlock gb;
    gb = gb1d(Alocal->nnz, NUM_THR);
    apply_mask_permut_GPU_noSideEffects_glob<<<gb.g, gb.b>>>(Alocal->nnz, Alocal->col, shrinking_permut->n, shrinking_permut->val, comp_col->val);

    return (comp_col);
}

extern int srmfb;

bool shrink_col(CSR* A, CSR* P)
{
    vector<int>* get_shrinked_col(CSR*, CSR*);
    if (!(A->shrinked_flag)) {
        if (P != NULL) { // product compatibility check
            if (A->m != P->full_n) {
                fprintf(stderr, "A->m=%lu, P->full_n=%lu\n", A->m, P->full_n);
            }
            assert(A->m == P->full_n);
        } else {
            assert(A->m == A->full_n);
        }

        vector<itype>* shrinking_permut = get_shrinked_col(A, P);
        if (0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "shrinking_permut_%x_%d", A, srmfb);
            FILE* fp = fopen(filename, "w");
            if (fp == NULL) {
                fprintf(stderr, "Could not open X\n");
            }
            Vector::print(shrinking_permut, -1, fp);
            fclose(fp);
        }
        assert(shrinking_permut->n >= (P != NULL ? P->n : A->n));
        vector<itype>* shrinkedA_col = apply_mask_permut_GPU_noSideEffects(A, shrinking_permut);

        A->shrinked_flag = true;
        A->shrinked_m = shrinking_permut->n;
        A->shrinked_col = shrinkedA_col->val;

        Vector::free(shrinking_permut);
        std::free(shrinkedA_col);
        return (true);
    } else {
        return (false);
    }
}

bool shrink_col(CSR* A, stype firstlocal, stype lastlocal, itype global_len)
{
    vector<int>* get_shrinked_col(CSR*, stype, stype);
    if (!(A->shrinked_flag)) {

        assert(A->m == global_len);

        vector<itype>* shrinking_permut = get_shrinked_col(A, firstlocal, lastlocal);
        assert(shrinking_permut->n >= (lastlocal - firstlocal + 1));
        vector<itype>* shrinkedA_col = apply_mask_permut_GPU_noSideEffects(A, shrinking_permut);

        A->shrinked_flag = true;
        A->shrinked_m = shrinking_permut->n;
        A->shrinked_col = shrinkedA_col->val;

        Vector::free(shrinking_permut);
        std::free(shrinkedA_col);
        return (true);
    } else {
        return (false);
    }
}

CSR* get_shrinked_matrix(CSR* A, CSR* P)
{

    if (!(A->shrinked_flag)) {
        assert(shrink_col(A, P));
    } else {
        bool test = (P != NULL) ? (P->row_shift == A->shrinked_firstrow) : (A->row_shift == A->shrinked_firstrow);
        test = test && ((P != NULL) ? (P->row_shift + P->n == A->shrinked_lastrow) : (A->row_shift + A->n == A->shrinked_lastrow));
        assert(test); // NOTE: check Pfirstrow, Plastrow
    }

    CSR* A_ = CSRm::init(A->n, A->shrinked_m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);
    A_->row = A->row;
    A_->val = A->val;
    A_->col = A->shrinked_col;

    return (A_);
}
