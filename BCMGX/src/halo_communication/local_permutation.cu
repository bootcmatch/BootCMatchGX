#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "halo_communication/local_permutation.h"
#include "utility/distribuite.h"
#include "utility/handles.h"
#include "utility/memory.h"
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

/**
 * @brief Kernel function to apply mask permutations on the columns of a sparse matrix.
 * 
 * This CUDA kernel is used to apply a mask permutation on the columns of a sparse matrix
 * without side effects. It updates the `comp_col` array by applying binary search to find 
 * the position of each column index in the shrinking permutation array.
 * 
 * @param nnz The number of non-zero elements in the sparse matrix.
 * @param col A pointer to the array of column indices of the sparse matrix.
 * @param shrinking_permut_len The length of the shrinking permutation array.
 * @param shrinking_permut A pointer to the shrinking permutation array.
 * @param comp_col A pointer to the array where the result of the permutation will be stored.
 */
__global__ void apply_mask_permut_GPU_noSideEffects_glob(itype nnz, const itype* col, int shrinking_permut_len, const itype* shrinking_permut, itype* comp_col)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

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

/**
 * @brief Applies mask permutations to the columns of a local sparse matrix.
 * 
 * This function applies a mask permutation to the columns of the local sparse matrix 
 * `Alocal` based on the `shrinking_permut` vector. It uses a CUDA kernel to perform 
 * the computation on the GPU and returns a vector `comp_col` representing the permuted 
 * column indices.
 * 
 * @param Alocal A pointer to the CSR (Compressed Sparse Row) matrix representing the local matrix.
 * @param shrinking_permut A pointer to the vector of shrinking permutations.
 * 
 * @return A pointer to a vector containing the permuted column indices of the sparse matrix.
 * 
 * @note Both the `Alocal` matrix and `shrinking_permut` vector should be on the device memory.
 */
vector<itype>* apply_mask_permut_GPU_noSideEffects(const CSR* Alocal, const vector<itype>* shrinking_permut)
{
    assert(Alocal->on_the_device);
    assert(shrinking_permut->on_the_device);

    vector<itype>* comp_col = Vector::init<itype>(Alocal->nnz, true, true);

    GridBlock gb = gb1d(Alocal->nnz, NUM_THR);
    apply_mask_permut_GPU_noSideEffects_glob<<<gb.g, gb.b>>>(Alocal->nnz, Alocal->col, shrinking_permut->n, shrinking_permut->val, comp_col->val);
    return (comp_col);
}

extern int srmfb;

/**
 * @brief Shrinks the columns of a sparse matrix `A` based on a given product matrix `P`.
 * 
 * This function shrinks the columns of matrix `A` based on a shrinking permutation vector
 * computed by the function `get_shrinked_col`. If the matrix `A` has not been shrunk before,
 * it performs the column shrinking and sets the `shrinked_flag` to true.
 * 
 * @param A A pointer to the CSR matrix to be shrunk.
 * @param P A pointer to the CSR matrix `P`, which is used for product compatibility check. Can be `NULL`.
 * 
 * @return A boolean value indicating whether the columns were successfully shrunk.
 */
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
        assert(shrinking_permut->n >= (P != NULL ? P->n : A->n));
        vector<itype>* shrinkedA_col = apply_mask_permut_GPU_noSideEffects(A, shrinking_permut);

        A->shrinked_flag = true;
        A->shrinked_m = shrinking_permut->n;
        A->shrinked_col = shrinkedA_col->val;

        Vector::free(shrinking_permut);
        FREE(shrinkedA_col);
        return (true);
    } else {
        return (false);
    }
}

/**
 * @brief Shrinks the columns of a sparse matrix `A` for a specified local row range.
 * 
 * This function shrinks the columns of matrix `A` based on a shrinking permutation vector
 * computed by the function `get_shrinked_col`. It considers only the rows from `firstlocal` 
 * to `lastlocal`. If the matrix `A` has not been shrunk before, it performs the column shrinking
 * and sets the `shrinked_flag` to true.
 * 
 * @param A A pointer to the CSR matrix to be shrunk.
 * @param firstlocal The index of the first local row to be considered for shrinking.
 * @param lastlocal The index of the last local row to be considered for shrinking.
 * @param global_len The global length of the matrix rows.
 * 
 * @return A boolean value indicating whether the columns were successfully shrunk.
 */
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
        FREE(shrinkedA_col);
        return (true);
    } else {
        return (false);
    }
}

/**
 * @brief Returns a shrunk version of the given CSR matrix `A` using matrix `P` for product compatibility.
 * 
 * This function returns a new CSR matrix representing a shrunk version of matrix `A` based 
 * on the shrinking permutation stored in `A->shrinked_col`. If matrix `A` has not been shrunk 
 * previously, it calls the `shrink_col` function to perform the shrinking. The result is 
 * returned as a new CSR matrix with the shrunk columns.
 * 
 * @param A A pointer to the CSR matrix to be shrunk.
 * @param P A pointer to the CSR matrix `P`, which is used for product compatibility check. Can be `NULL`.
 * 
 * @return A pointer to the shrunk CSR matrix `A_`.
 */
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
