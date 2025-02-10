/** @file */

#ifndef LOCAL_PERMUTATION

#define LOCAL_PERMUTATION

#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include <stdio.h>
#include <unistd.h>

// ---------------------------------------------------------------------------------

// vector<itype>* compute_mask_permut(CSR* Alocal, CSR* Plocal, vector<int>* bitcol, FILE* fp = stdout);

// vector<itype>* compute_mask_permut_GPU(const CSR* Alocal, const CSR* Plocal, vector<int>* bitcol);

void apply_mask_permut(CSR** Alocal_dev, vector<itype>* mask_permut, FILE* fp = stdout);

void apply_mask_permut_GPU(CSR* Alocal, vector<itype>* shrinking_permut, FILE* fp = stdout);

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
vector<itype>* apply_mask_permut_GPU_noSideEffects(const CSR* Alocal, const vector<itype>* shrinking_permut);

void reverse_mask_permut(CSR** Alocal_dev, vector<itype>* mask_permut, FILE* fp = stdout);

void reverse_mask_permut_GPU(CSR* Alocal, vector<itype>* shrinking_permut, FILE* fp = stdout);

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
bool shrink_col(CSR* A, CSR* P = NULL);

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
CSR* get_shrinked_matrix(CSR* A, CSR* P);

#endif
