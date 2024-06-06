#ifndef LOCAL_PERMUTATION

#define LOCAL_PERMUTATION

#include <unistd.h>
#include <stdio.h>
#include "prec_setup/AMG.h"
// #include "basic_kernel/halo_communication/localization_debug.h"


// ---------------------------------------------------------------------------------

vector<itype>* compute_mask_permut (CSR *Alocal, CSR *Plocal, vector<int> *bitcol, FILE* fp = stdout);

vector<itype>* compute_mask_permut_GPU (const CSR *Alocal, const CSR *Plocal, vector<int> *bitcol);

void apply_mask_permut (CSR **Alocal_dev, vector<itype>* mask_permut, FILE* fp = stdout);

void apply_mask_permut_GPU (CSR *Alocal, vector<itype>* shrinking_permut, FILE* fp = stdout);

vector<itype>* apply_mask_permut_GPU_noSideEffects (const CSR *Alocal, const vector<itype>* shrinking_permut);

void reverse_mask_permut (CSR **Alocal_dev, vector<itype>* mask_permut, FILE* fp = stdout);

void reverse_mask_permut_GPU (CSR *Alocal, vector<itype>* shrinking_permut, FILE* fp = stdout);

bool shrink_col(CSR* A, CSR* P = NULL);

CSR* get_shrinked_matrix(CSR* A, CSR* P);

#endif
