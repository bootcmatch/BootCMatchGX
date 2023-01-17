#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <unistd.h>
#include <getopt.h>

#define DIE CHECK_DEVICE(cudaDeviceSynchronize());MPI_Finalize();exit(0);

#include "basic_kernel/matrix/scalar.h"
#include "basic_kernel/matrix/vector.h"
#include "basic_kernel/matrix/matrixIO.h"
#include "utility/myMPI.h"
#include "utility/handles.h"
#include "prec_setup/AMG.h"
#include <string>
#include "utility/distribuite.h"
#include "basic_kernel/halo_communication/localization_debug.h"
#include "utility/utils.h"

#define NUM_THR 1024

// --------------- only for KDevelop ----------------------
#include <curand_mtgp32_kernel.h>
// --------------------------------------------------------

#define MAX_NNZ_PER_ROW_LAP 5
#define MPI 1
#ifndef __GNUC__
typedef int (*__compar_fn_t)(const void *, const void *);
#endif

void print_bit (int b, int mask, int index, FILE* fp) {
    // ATTENZIONE!! Una delle due righe di "print_bit" deve sempre essere commentata
    if (mask != 0) {
//         print_bit (b, mask << 1, index+1, fp); // per usare l'ordine dell'espansione binaria (most significant bit a sx)
        if ((index% 8) == 0)
            fprintf(fp, "| ");
        if (mask & b)
            fprintf(fp, "1 ");
        else
            fprintf(fp, "0 ");
        
//         if ((index% 8) == 0)
//             fprintf(fp, "| ");
        print_bit (b, mask << 1, index+1, fp); // per usare l'ordine di vettore booleano (most significant bit a dx)
    }
    return;
}
  
void print_bitabit(vector<int> *b, FILE* fp) {
    int k, j;
    for (j=0; j<b->n; j++) {
        k = 1;
        print_bit (b->val[j], k, 0, fp);
        fprintf(fp, "| ");
    }
    fprintf(fp, "\n");
    return;
}

void print_row_to_get_info (CSR* Alocal, int nprocs, FILE* fp) {
    if (Alocal->rows_to_get == NULL) {
        fprintf(fp, "row_to_get_info is void\n");
        return;
    }
    int i;
    rows_to_get_info *p = Alocal->rows_to_get;
    
    fprintf(fp, "itype *P_n_per_process: ");
    for (i=0; i<nprocs; i++)
        fprintf(fp, "%3d ", p->P_n_per_process[i]);
    fprintf(fp, "\n");
    
    fprintf(fp, "rows to send to other process:\n");
    fprintf(fp, "\tunsigned int *rcvcntp: ");
    for (i=0; i<nprocs; i++)
        fprintf(fp, "%3d ", p->rcvcntp[i]);
    fprintf(fp, "\n");
    fprintf(fp, "\tint *displr: ");
    for (i=0; i<nprocs; i++)
        fprintf(fp, "%3lu ", (p->displr[i])/sizeof(itype));
    fprintf(fp, "\n");
    
    fprintf(fp, "rows to be received from other process:\n");
    fprintf(fp, "\titype rows2bereceived = %d\n", p->rows2bereceived);
    fprintf(fp, "\tunsigned int *rcounts2(/sizeof(itype)): ");
    for (i=0; i<nprocs; i++)
        fprintf(fp, "%3lu ", (p->rcounts2[i])/sizeof(itype));
    fprintf(fp, "\n");
    fprintf(fp, "\tint *displr2: ");
    for (i=0; i<nprocs; i++)
        fprintf(fp, "%3lu ", (p->displr2[i])/sizeof(itype));
    fprintf(fp, "\n");
    
    fprintf(fp, "global index of the rows to send to other process:\n");
    fprintf(fp, "\titype *rcvprow: ");
//     for (i=0; i<p->total_row_to_rec; i++)
//         fprintf(fp, "%3d ", p->rcvprow[i]);
//     fprintf(fp, "\n");
    
    fprintf(fp, "global index of the rows to be recived from other process::\n");
    fprintf(fp, "\titype *whichprow: ");
    for (i=0; i<p->rows2bereceived; i++)
        fprintf(fp, "%3d ", p->whichprow[i]);
    fprintf(fp, "\n");
    
//     fprintf(fp, " ...  ... :\n");
//     fprintf(fp, "\titype *rcvpcolxrow: ");
//     for (i=0; i<p->rows2bereceived; i++)
//         fprintf(fp, "%3d ", p->rcvpcolxrow[i]);
//     fprintf(fp, "\n");
    
}

void print_mask_permut_result(vector<itype> *v, int n_, FILE* fp){
    vector<itype> *v_;

    int n;

    if(n_ == -1)
      n = v->n;
    else
      n = n_;

    if(v->on_the_device){
      v_ = Vector::copyToHost<itype>(v);
    }else{
      v_ = v;
    }

    int i;
    fprintf(fp, "Mask_permut result:\n\t");
    for(i=0; i<(n/2); i++)
        fprintf(fp, "%3d ", v_->val[i]);
    fprintf(fp, "\n\t");
    for(i=0; i<(n/2); i++)
        fprintf(fp, " || ");
    fprintf(fp, "\n\t");
    for(i=0; i<(n/2); i++)
        fprintf(fp, " \\/ ");
    fprintf(fp, "\n\t");
    for(i=(n/2); i<n; i++)
        fprintf(fp, "%3d ", v_->val[i]);
    fprintf(fp, "\n\n");

    if(v->on_the_device){
      Vector::free<itype>(v_);
    }
}

// bool test_mask_permut(CSR *A, FILE *fp) {
//     vector<int> *_bitcol = NULL;
//     _bitcol = get_missing_col( A, NULL );
//     vector<itype> *y = compute_mask_permut(A, _bitcol, fp);
//     
//     CSR *B = CSRm::clone(A);
//     apply_mask_permut(&A,y,fp);
//     reverse_mask_permut(&A,y,fp);
//     bool r = CSRm::equals(A, B);
//     return(r);
// }
// 
// bool test_mask_permut_and_shrink(CSR *A, FILE *fp) {
//     vector<int> *_bitcol = NULL;
//     _bitcol = get_missing_col( A, NULL );
//     vector<itype> *y = compute_mask_permut(A, _bitcol, fp);
//     
//     CSR *B = CSRm::clone(A);
//     apply_mask_permut_and_shrink(&A,y,fp);
//     reverse_mask_permut_and_shrink(&A,y,fp);
//     bool r = CSRm::equals(A, B);
//     return(r);
// } 
