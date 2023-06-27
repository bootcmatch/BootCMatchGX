#include <string.h>
#include "basic_kernel/matrix/CSR.h"
#include "utility/cudamacro.h"
#include <basic_kernel/custom_cudamalloc/custom_cudamalloc.h>
// #include "basic_kernel/halo_communication/extern.h"
#include "basic_kernel/halo_communication/extern2.h"

#include "utility/function_cnt.h"

int CSRm::choose_mini_warp_size(CSR *A){

  int density = A->nnz / A->n;

  if(density < MINI_WARP_THRESHOLD_2)
    return 2;
  else if(density < MINI_WARP_THRESHOLD_4)
    return 4;
  else if(density < MINI_WARP_THRESHOLD_8)
    return 8;
  else if(density < MINI_WARP_THRESHOLD_16)
    return 16;
  else{
    return 32;
  }
}

// extern functionCall_cnt function_cnt;

CSR* CSRm::init(stype n, gstype m, stype nnz, bool allocate_mem, bool on_the_device, bool is_symmetric, gstype full_n, gstype row_shift){

// ---------- Pico ----------
  if ( n<=0 || m <=0 || nnz <=0){
      fprintf(stderr, "error in CSRm::init:\n\tint  n: %d  m: %d  nnz: %d\n\tunsigned  n: %u  m: %u  nnz: %u\n", n, m, nnz, n, m, nnz);
      //printf("error in CSRm::init:\n\tint  n: %d  m: %d  nnz: %d\n\tunsigned  n: %u  m: %u  nnz: %u\n", n, m, nnz, n, m, nnz);
      //fflush(stdout);
  }
  assert(n > 0);
  assert(m > 0);
  assert(nnz > 0);
// --------------------------
  
  CSR *A = NULL;

  // on the host
  A = (CSR*) malloc(sizeof(CSR));
  CHECK_HOST(A);

  A->nnz = nnz;
  A->n = n;
  A->m = m;
  A->full_m = m; // NOTE de sostituire cambiando CSRm::Init inserendo full_m

  A->on_the_device = on_the_device;
  A->is_symmetric = false;
  A->custom_alloced = false;

  A->full_n = full_n;
  A->row_shift = row_shift;

  A->rows_to_get = NULL;
  
  //itype shrinked_firstrow = 0;
  //itype shrinked_lastrow  = 0;
  A->shrinked_flag = false;
  A->shrinked_col = NULL;
  A->shrinked_m = m;
  A->halo.init = false;
  A->col_shifted=0;
  
  A->post_local = 0;
  A->bitcolsize = 0;
  A->bitcol = NULL;

  if(allocate_mem){
    if(on_the_device){
      // on the device
      cudaError_t err;
      err = cudaMalloc( (void**) &A->val, nnz * sizeof(vtype) );
      CHECK_DEVICE(err);
      err = cudaMalloc( (void**) &A->col, nnz * sizeof(itype) );
      CHECK_DEVICE(err);
      err = cudaMalloc( (void**) &A->row, (n + 1) * sizeof(itype) );
      CHECK_DEVICE(err);
    }else{
      // on the host
      A->val = (vtype*) malloc( nnz * sizeof(vtype) );
      CHECK_HOST(A->val);
      A->col = (itype*) malloc( nnz * sizeof(itype) );
      CHECK_HOST(A->col);
      A->row = (itype*) malloc( (n + 1) * sizeof(itype) );
      CHECK_HOST(A->row);
    }
  }

  cusparseMatDescr_t *descr = NULL;
  descr = (cusparseMatDescr_t*) malloc( sizeof(cusparseMatDescr_t) );
  CHECK_HOST(descr);

  cusparseStatus_t  err = cusparseCreateMatDescr(descr);
  CHECK_CUSPARSE(err);

  cusparseSetMatIndexBase(*descr, CUSPARSE_INDEX_BASE_ZERO);

  if(is_symmetric)
    cusparseSetMatType(*descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
  else
    cusparseSetMatType(*descr, CUSPARSE_MATRIX_TYPE_GENERAL);

  A->descr = descr;
  return A;
}

void CSRm::print(CSR *A, int type, int limit, FILE* fp){
  CSR *A_ = NULL;
  
  if(A->on_the_device)
    A_ = CSRm::copyToHost(A);
  else
    A_ = A;
  
  switch(type) {
    case 0:
      fprintf(fp, "ROW: %d (%d)\n\t", A_->n, A_->full_n);
      if(limit == 0)
        limit = A_->full_n + 1;
      for(int i=0; i<limit; i++){
        fprintf(fp, "%3d ", A_->row[i]);
      }
      break;
    case 1:
      fprintf(fp, "COL:\n");
      if(limit == 0)
        limit = A_->nnz;
      for(int i=0; i<limit; i++){
        fprintf(fp, "%d\n", A_->col[i]);
      }
      break;
    case 2:
      fprintf(fp, "VAL:\n");
      if(limit == 0)
        limit = A_->nnz;
      for(int i=0; i<limit; i++){
        fprintf(fp, "%14.12g\n", A_->val[i]);
      }
      break;
    case 3:
      fprintf(fp, "MATRIX_Form:\n");
      for(int i=0; i<A_->n; i++){
          fprintf(fp, "\t");
          for (int j=0; j<A_->m; j++) {
              int flag = 0, temp = A_->row[i];
              for ( temp = A_->row[i]; flag==0 && (i!=(A_->n)-1 ? temp < (A_->row[i+1]) : temp < A_->nnz) ; temp++ ) {
                if (A_->col[temp] == j) {
                    fprintf(fp, "%g ", A_->val[temp]);
                    flag = 1;
                }
              }
              if (flag == 0)
                fprintf(fp, "%g ", 0.0);
          }
          fprintf(fp, "\n");
      }
      break;
    case 4:
      fprintf(fp, "boolMATRIX_Form:\n");
      for(int i=0; i<A_->n; i++){
          fprintf(fp, "\t");
          for (int j=0; j<A_->m; j++) {
              if (j % 32 == 0)
                fprintf(fp, "| ");
              int flag = 0, temp = A_->row[i];
              for ( temp = A_->row[i]; flag==0 && (i!=(A_->n)-1 ? temp < (A_->row[i+1]) : temp < A_->nnz) ; temp++ ) {
                if (A_->col[temp] == j) {
                    fprintf(fp, "\033[0;31mX\033[0m ");
                    flag = 1;
                }
              }
              if (flag == 0)
                fprintf(fp, "O ");
          }
          fprintf(fp, "\n");
      }
      break;
    case 5:
      fprintf(fp, "SHRINKED COL:\n");
      if(limit == 0)
        limit = A_->shrinked_m;
      for(int i=0; i<limit; i++){
        fprintf(fp, "%d\n", A_->shrinked_col[i]);
      }
      break;
      
  }
  fprintf(fp, "\n\n");
  
  if(A->on_the_device)
    CSRm::free(A_);
}

bool CSRm::equals(CSR *A, CSR *B) {
    CSR *A_ = NULL, *B_ = NULL;
    bool A_dev_flag, B_dev_flag, r = true;
  
    if(A->on_the_device) {
        A_ = CSRm::copyToHost(A);
        A_dev_flag = true;
    } else {
        A_ = A;
        A_dev_flag = false;
    }
  
    if(B->on_the_device) {
        B_ = CSRm::copyToHost(B);
        B_dev_flag = true;
    } else {
        B_ = B;
        B_dev_flag = false;
    }

    if ( (A_->n != B_->n) ) {
        r = false;
//         printf("(A_->n != B_->n)\n");
//         printf("memcmp(A_->row, B_->row, sizeof(itype)*((A_->n)+1))\n");
    }else{
        if ( memcmp(A_->row, B_->row, sizeof(itype)*((A_->n)+1)) ) {
            r = false;
//             printf("memcmp(A_->row, B_->row, sizeof(itype)*((A_->n)+1))\n");
        }
    }
    if ( (A_->m != B_->m) ) {
        r = false;
//         printf("(A_->m != B_->m)\n");
    }
    if ( (A_->nnz != B_->nnz) ) {
        r = false;
//         printf("(A_->nnz != B_->nnz)\n");
//         printf("memcmp(A_->val, B_->val, sizeof(vtype)*A_->nnz)\n");
//         printf("memcmp(A_->col, B_->col, sizeof(itype)*A_->nnz)\n");
    }else{
        if ( memcmp(A_->val, B_->val, sizeof(vtype)*A_->nnz) ) {
            r = false;
//             printf("memcmp(A_->val, B_->val, sizeof(vtype)*A_->nnz)\n");
        }
        if ( memcmp(A_->col, B_->col, sizeof(itype)*A_->nnz) ) {
            r = false;
//             printf("memcmp(A_->col, B_->col, sizeof(itype)*A_->nnz)\n");
        }
    }
    
    
    if (A_dev_flag)
        CSRm::free(A_);
    if (B_dev_flag)
        CSRm::free(B_);
    return(r);
}

bool CSRm::halo_equals(halo_info *a, halo_info* b) {
    _MPI_ENV;
    
    if (a->to_receive_n != b->to_receive_n)
        return(false);
//     if (Vector::equals<itype>(a->to_receive, b->to_receive) != true)
//         return(false);
//     if (Vector::equals<itype>(a->to_receive_d, b->to_receive_d) != true)
//         return(false);
    if ((b->to_receive_n>0) && (memcmp((void*) a->to_receive->val, (const void*) b->to_receive->val, sizeof(itype) * a->to_receive->n) != 0))
        return(false);
    
    
    if (memcmp((void*) b->to_receive_counts, (const void*) a->to_receive_counts, sizeof(int) * nprocs) != 0)
        return(false);
    if (memcmp((void*) b->to_receive_spls, (const void*) a->to_receive_spls, sizeof(int) * nprocs) != 0)
        return(false);
    if ((b->to_receive_n>0) && (memcmp((void*) b->what_to_receive, (const void*) a->what_to_receive, sizeof(vtype) * b->to_receive_n) != 0))
        return(false);
    
    
    if (a->to_send_n != b->to_send_n)
        return(false);
//     if (Vector::equals<itype>(a->to_send, b->to_send) != true)
//         return(false);
//     if (Vector::equals<itype>(a->to_send_d, b->to_send_d) != true)
//         return(false);
    if ((b->to_receive_n>0) && (memcmp((void*) a->to_send->val, (const void*) b->to_send->val, sizeof(itype) * a->to_send->n) != 0))
        return(false);
    
    if (memcmp((void*) b->to_send_counts, (const void*) a->to_send_counts, sizeof(int) * nprocs) != 0)
        return(false);
    if (memcmp((void*) b->to_send_spls, (const void*) a->to_send_spls, sizeof(int) * nprocs) != 0)
        return(false);
    if ((b->to_receive_n>0) && (memcmp((void*) b->what_to_send, (const void*) a->what_to_send, sizeof(vtype) * b->to_send_n) != 0))
        return(false);
    
    return(true);
}

bool CSRm::chk_uniprol(CSR *A) {
    CSR *A_ = NULL;
    bool dev_flag, r = true;
  
    if(A->on_the_device) {
        A_ = CSRm::copyToHost(A);
        dev_flag = true;
    } else {
        A_ = A;
        dev_flag = false;
    }
  

    for (int i = 1; r && (i<A->n); i++) {
        if (A_->row[i] != A_->row[i-1] +1)
            r = false;
    }
    
    
    if (dev_flag)
        CSRm::free(A_);
    return(r);
}

void CSRm::compare_nnz(CSR *A, CSR *B, int type) {
    FILE *fp = stdout;
    CSR *A_ = NULL, *B_ = NULL;
    bool A_dev_flag, B_dev_flag;
    itype nnz_border = 0;
    
    if (A->n != B->n || A->m != B->m) {
        fprintf(fp, "A->n != B->n || A->m != B->m\n");
        return;
    }
    if ( (type != 4) && (A->nnz != B->nnz) ) {
        fprintf(fp, "(type != 4) && (A->nnz != B->nnz)\n");
        fprintf(fp, "A->nnz = %d,  B->nnz = %d\n", A->nnz, B->nnz);
        if (A->nnz > B->nnz)
            nnz_border = B->nnz;
        else
            nnz_border = A->nnz;
    }
    
    if(A->on_the_device) {
        A_ = CSRm::copyToHost(A);
        A_dev_flag = true;
    } else {
        A_ = A;
        A_dev_flag = false;
    }
  
    if(B->on_the_device) {
        B_ = CSRm::copyToHost(B);
        B_dev_flag = true;
    } else {
        B_ = B;
        B_dev_flag = false;
    }


    
    switch(type) {
    case 0:
      fprintf(fp, "A ROW: %d (%d)\n\t", A_->n, A_->full_n);
      for(int i=0; i<A_->n + 1; i++){
        if (A_->row[i] == B_->row[i])
            fprintf(fp, "%3d ", A_->row[i]);
        else
            fprintf(fp, "\033[0;31m%3d\033[0m ", A_->row[i]);
      }
      fprintf(fp, "\n\nB ROW: %d (%d)\n\t", B_->n, B_->full_n);
      for(int i=0; i<B_->n + 1; i++){
        if (A_->row[i] == B_->row[i])
            fprintf(fp, "%3d ", B_->row[i]);
        else
            fprintf(fp, "\033[0;31m%3d\033[0m ", B_->row[i]);
      }
      break;
    case 1:
      fprintf(fp, "A COL:\n\t");
      for(int i=0; i<nnz_border; i++){
        if (A_->col[i] == B_->col[i])
            fprintf(fp, "%3d ", A_->col[i]);
        else
            fprintf(fp, "\033[0;31m%3d\033[0m ", A_->col[i]);
      }
      fprintf(fp, "\n\nB COL:\n\t");
      for(int i=0; i<nnz_border; i++){
        if (A_->col[i] == B_->col[i])
            fprintf(fp, "%3d ", B_->col[i]);
        else
            fprintf(fp, "\033[0;31m%3d\033[0m ", B_->col[i]);
      }
      break;
    case 2:
      fprintf(fp, "A VAL:\n\t");
      for(int i=0; i<nnz_border; i++){
        if (A_->val[i] == B_->val[i])
            fprintf(fp, "%6.2lf ", A_->val[i]);
        else
            fprintf(fp, "\033[0;31m%6.2lf\033[0m ", A_->val[i]);
      }
      fprintf(fp, "\n\nB VAL:\n\t");
      for(int i=0; i<nnz_border; i++){
        if (A_->val[i] == B_->val[i])
            fprintf(fp, "%6.2lf ", B_->val[i]);
        else
            fprintf(fp, "\033[0;31m%6.2lf\033[0m ", B_->val[i]);
      }
      break;
    case 3:
      fprintf(fp, "Nnz differences in MATRIX_Form:\n\t[ ... ]\n");
      break;
    case 4:
        fprintf(fp, "Nnz differences in boolMATRIX_Form:\n");
        for(int i=0; i<A_->n; i++){
            fprintf(fp, "\t");
            for (int j=0; j<A_->m; j++) {
                if (j % 32 == 0)
                    fprintf(fp, "| ");
                int flag = 0, tempA = A_->row[i], tempB = B_->row[i];
                for ( tempA = A_->row[i]; flag==0 && (i!=(A_->n)-1 ? tempA < (A_->row[i+1]) : tempA < A_->nnz) ; tempA++ ) {
                    if (A_->col[tempA] == j) {
                        flag = 1;
                    }
                }
                for ( tempB = B_->row[i]; (i!=(B_->n)-1 ? tempB < (B_->row[i+1]) : tempB < B_->nnz) ; tempB++ ) {
                    if (B_->col[tempB] == j) {
                        if(flag == 0) {
                            flag = 1;
                        } else {
                            if (A_->val[tempA-1] != B_->val[tempB])
                                flag = 2;
                            else
                                flag = -1;
                        }
                        break;
                    }
                }
                if (flag == 0 || flag == -1) {
                    if (flag == 0)
                        fprintf(fp, "O ");
                    else
                        fprintf(fp, "\033[0;32mX\033[0m ");
                } else {
                    if (flag == 1)
                        fprintf(fp, "\033[0;33mX\033[0m ");
                    else
                        fprintf(fp, "\033[0;35mX\033[0m ");
                }
            }
            fprintf(fp, "\n");
        }
        break;
    }
    fprintf(fp, "\n\n");
    
    
  
    if (A_dev_flag)
        CSRm::free(A_);
    if (B_dev_flag)
        CSRm::free(B_);
    return;
}



CSR* CSRm::clone(CSR *A_) {
    CSR *Out = NULL, *A = NULL;
    bool dev_flag;
    if(A_->on_the_device) {
        A = CSRm::copyToHost(A_);
        dev_flag = true;
    } else {
        A = A_;
        dev_flag = false;
    }
    Out = CSRm::init(A->n, A->m, A->nnz, true, false, false, A->full_n, A->row_shift);
    Out->row = (itype*) malloc(sizeof(itype) * ((A->n) +1) );
    Out->col = (itype*) malloc(sizeof(itype) * (A->nnz) );
    Out->val = (vtype*) malloc(sizeof(vtype) * (A->nnz) );
    
    memcpy((void*) Out->row, (const void*) A->row, sizeof(itype) * ((A->n) +1) );
    memcpy((void*) Out->col, (const void*) A->col, sizeof(itype) * (A->nnz) );
    memcpy((void*) Out->val, (const void*) A->val, sizeof(vtype) * (A->nnz) );
    
    if (dev_flag){
        CSRm::free(A);
        A = Out;
        Out = CSRm::copyToDevice(Out);
        CSRm::free(A);
    }
    return(Out);
}

halo_info CSRm::clone_halo( halo_info* a) {
    _MPI_ENV;
    
    halo_info b;
    
    b.to_receive_n = a->to_receive_n;
    if (a->to_receive_n > 0) {
        b.to_receive = Vector::clone<gstype>(a->to_receive);
        b.to_receive_d = Vector::clone<gstype>(a->to_receive_d);
    }
    
    b.to_receive_counts = (int*) malloc(sizeof(int) * nprocs);
    b.to_receive_spls = (int*) malloc(sizeof(int) * nprocs);
    b.what_to_receive = (vtype*) malloc(sizeof(vtype) * nprocs);
    cudaMalloc_CNT
    CHECK_DEVICE( cudaMalloc(&(b.what_to_receive_d), sizeof(vtype) * b.to_receive_n) );
    
    if (a->to_receive_n != 0) {
        memcpy((void*) b.to_receive_counts, (const void*) a->to_receive_counts, sizeof(int) * nprocs);
        memcpy((void*) b.to_receive_spls, (const void*) a->to_receive_spls, sizeof(int) * nprocs);
        memcpy((void*) b.what_to_receive, (const void*) a->what_to_receive, sizeof(vtype) * b.to_receive_n);
//         if (a->what_to_receive_d != NULL)
//             CHECK_DEVICE( cudaMemcpy(b.what_to_receive_d, a->what_to_receive_d, sizeof(vtype) * b.to_receive_n, cudaMemcpyDeviceToDevice) );
    }
    
    b.to_send_n = a->to_send_n;
    printf("a->to_send_n = %d, a->to_send->n = %d\n", a->to_send_n, a->to_send->n);
    if (a->to_send_n > 0) {
        b.to_send = Vector::clone<itype>(a->to_send);
        b.to_send_d = Vector::clone<itype>(a->to_send_d);
    }
    
    b.to_send_counts = (int*) malloc(sizeof(int) * nprocs);
    b.to_send_spls = (int*) malloc(sizeof(int) * nprocs);
    b.what_to_send = (vtype*) malloc(sizeof(vtype) * nprocs);
    cudaMalloc_CNT
    CHECK_DEVICE( cudaMalloc(&(b.what_to_send_d), sizeof(vtype) * b.to_send_n) );
    
    if (a->to_send_n != 0) {
        memcpy((void*) b.to_send_counts, (const void*) a->to_send_counts, sizeof(int) * nprocs);
        memcpy((void*) b.to_send_spls, (const void*) a->to_send_spls, sizeof(int) * nprocs);
        memcpy((void*) b.what_to_send, (const void*) a->what_to_send, sizeof(vtype) * b.to_send_n);
//         if (a->what_to_receive_d != NULL)
//             CHECK_DEVICE( cudaMemcpy(b.what_to_send_d, a->what_to_send_d, sizeof(vtype) * b.to_send_n, cudaMemcpyDeviceToDevice) );
    }
    
    return (b);
}

void CSRm::free_rows_to_get(CSR *A){
  if (A->rows_to_get != NULL){
    std::free( A->rows_to_get->rcvprow);
    std::free( A->rows_to_get->whichprow);
    std::free( A->rows_to_get->rcvpcolxrow);
    std::free( A->rows_to_get->scounts);
    std::free( A->rows_to_get->displs);
    std::free( A->rows_to_get->displr);
    std::free( A->rows_to_get->rcounts2);
    std::free( A->rows_to_get->scounts2);
    std::free( A->rows_to_get->displs2);
    std::free( A->rows_to_get->displr2);
    std::free( A->rows_to_get->rcvcntp);
    std::free( A->rows_to_get->P_n_per_process);
    if (A->rows_to_get->nnz_per_row_shift != NULL){
        Vector::free(A->rows_to_get->nnz_per_row_shift);
    }
    std::free( A->rows_to_get );
  }
  A->rows_to_get = NULL; 
}

void CSRm::free(CSR *A){
  if(A->on_the_device){
    if (A->custom_alloced == false) {
        cudaError_t err;
        err = cudaFree(A->val);
        CHECK_DEVICE(err);
        err = cudaFree(A->col);
        CHECK_DEVICE(err);
        err = cudaFree(A->row);
        CHECK_DEVICE(err);
        if (A->shrinked_col != NULL) {
            err = cudaFree(A->shrinked_col);
            CHECK_DEVICE(err);
        }
    }
  }else{
    std::free(A->val);
    std::free(A->col);
    std::free(A->row);
  }
  CHECK_CUSPARSE( cusparseDestroyMatDescr(*A->descr) );
  if (A->rows_to_get != NULL){
    std::free( A->rows_to_get->rcvprow);
    std::free( A->rows_to_get->whichprow);
    std::free( A->rows_to_get->rcvpcolxrow);
    std::free( A->rows_to_get->scounts);
    std::free( A->rows_to_get->displs);
    std::free( A->rows_to_get->displr);
    std::free( A->rows_to_get->rcounts2);
    std::free( A->rows_to_get->scounts2);
    std::free( A->rows_to_get->displs2);
    std::free( A->rows_to_get->displr2);
    std::free( A->rows_to_get->rcvcntp);
    std::free( A->rows_to_get->P_n_per_process);
    if (A->rows_to_get->nnz_per_row_shift != NULL){
        Vector::free(A->rows_to_get->nnz_per_row_shift);
    }
    std::free( A->rows_to_get );
  }
  
  // ------------- custom cudaMalloc -------------
//   if (A->bitcol != NULL) {
//       CHECK_DEVICE( cudaFree(A->bitcol) );
//   }
  // ---------------------------------------------
  
  // Free the halo_info halo halo_info halo; 
  std::free(A->descr);
  std::free(A);
}


void CSRm::freeStruct(CSR *A){
  CHECK_CUSPARSE( cusparseDestroyMatDescr(*A->descr) );
  std::free(A->descr);
  std::free(A);
}


void CSRm::printInfo(CSR *A, FILE* fp){
  fprintf(fp, "Device?: %d\n", A->on_the_device);
  fprintf(fp, "nnz: %d\n", A->nnz);
  fprintf(fp, "n: %d\n", A->n);
  fprintf(fp, "m: %d\n", A->m);
  fprintf(fp, "full_n: %d\n", A->full_n);
  fprintf(fp, "row_shift: %d\n", A->row_shift);
  if(A->is_symmetric)
    fprintf(fp, "SYMMETRIC\n");
  else
    fprintf(fp, "GENERAL\n");
	fprintf(fp, "\n");
}

void shift_cpucol(itype *Arow, itype *Acol, unsigned int n, stype row_shift) {

  for(unsigned int i=0; i<n; i++){
      for (unsigned int j = Arow[i]; j< Arow[i+1]; j++) {
           Acol[j]+=row_shift;
      }
  }
  /*  A->col_shifted=-row_shift; */
}


CSR* CSRm::copyToDevice(CSR *A){

  assert( !A->on_the_device );

  itype n, nnz;
  gstype m;
  n = A->n;
  m = A->m;

  nnz = A->nnz;

  // allocate CSR matrix on the device memory
  CSR *A_d = CSRm::init(n, m, nnz, true, true, A->is_symmetric, A->full_n, A->row_shift);
  A_d->full_m = A->full_m;  // NOTE da eliminare cambiando CSRm::Init inserendo full_m

  cudaError_t err;
  err = cudaMemcpy(A_d->val, A->val, nnz * sizeof(vtype), cudaMemcpyHostToDevice);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A_d->row, A->row, (n + 1) * sizeof(itype), cudaMemcpyHostToDevice);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A_d->col, A->col, nnz * sizeof(itype), cudaMemcpyHostToDevice);
  CHECK_DEVICE(err);

  return A_d;
}


CSR* CSRm::copyToHost(CSR *A_d){

  assert( A_d->on_the_device );

  itype n, m, nnz;

  n = A_d->n;
  m = A_d->m;

  nnz = A_d->nnz;

  // allocate CSR matrix on the device memory
  CSR *A = CSRm::init(n, m, nnz, true, false, A_d->is_symmetric, A_d->full_n, A_d->row_shift);
  A->full_m = A_d->full_m;  // NOTE da eliminare cambiando CSRm::Init inserendo full_m

  cudaError_t err;

  err = cudaMemcpy(A->val, A_d->val, nnz * sizeof(vtype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A->row, A_d->row, (n + 1) * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A->col, A_d->col, nnz * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
  if(A_d->shrinked_m) {
    A->shrinked_col=(itype *)malloc(A_d->shrinked_m*sizeof(itype));
    CHECK_HOST(A->shrinked_col);
    err = cudaMemcpy(A->shrinked_col, A_d->shrinked_col,A_d->shrinked_m * sizeof(itype), cudaMemcpyDeviceToHost);    
  } else {
    A->shrinked_col=NULL;
  }
  

  return A;
}



__global__
void _shift_cols(itype n, itype *col, gsstype shift){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(i >= n)
    return;
  gsstype scratch=col[i];
  scratch+=shift;
  col[i] = scratch;
}

void CSRm::shift_cols(CSR* A, gsstype shift){
  PUSH_RANGE(__func__, 7)
    
  assert(A->on_the_device);
  gridblock gb = gb1d(A->nnz, BLOCKSIZE);
  _shift_cols<<<gb.g, gb.b>>>(A->nnz, A->col, shift);
  
  POP_RANGE
}

// return a copy of A->T
CSR* CSRm::T(cusparseHandle_t cusparse_h, CSR* A){

  assert( A->on_the_device );

  cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
  cusparseIndexBase_t idxbase = CUSPARSE_INDEX_BASE_ZERO;

  // --------------------------- Custom CudaMalloc ---------------------------------
  CSR *AT = CSRm::init((stype)A->m, (gstype) A->n, A->nnz, true, true, A->is_symmetric, A->m, 0);
  //CSR *AT = CSRm::init(n_rows, A->n, A->nnz, true, true, A->is_symmetric, A->m, 0);
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  //CSR *AT = CSRm::init(A->m, A->n, A->nnz, false, true, A->is_symmetric, A->m, 0);
  //AT->val = CustomCudaMalloc::alloc_vtype(AT->nnz);
  //AT->col = CustomCudaMalloc::alloc_itype(AT->nnz);
  //AT->row = CustomCudaMalloc::alloc_itype((AT->n) +1);
  //AT->custom_alloced = true;
  // -------------------------------------------------------------------------------
  
  size_t buff_T_size = 0;

  cusparseStatus_t err = cusparseCsr2cscEx2_bufferSize(
    cusparse_h,
    A->n,
    A->m,
    A->nnz,
    A->val,
    A->row,
    A->col,
    AT->val,
    AT->row,
    AT->col,
    CUDA_R_64F,
    copyValues,
    idxbase,
    CUSPARSE_CSR2CSC_ALG1,
    &buff_T_size
  );

  CHECK_CUSPARSE(err);
  assert(buff_T_size);


  void *buff_T = NULL;
  CHECK_DEVICE(  cudaMalloc(&buff_T, buff_T_size) );

  err = cusparseCsr2cscEx2(
    cusparse_h,
    A->n,
    A->m,
    A->nnz,
    A->val,
    A->row,
    A->col,
    AT->val,
    AT->row,
    AT->col,
    CUDA_R_64F,
    copyValues,
    idxbase,
    CUSPARSE_CSR2CSC_ALG1,
    buff_T
  );
  CHECK_CUSPARSE(err);

  CHECK_DEVICE( cudaFree(buff_T) );

  return AT;
}



CSR* CSRm::T_multiproc(cusparseHandle_t cusparse_h, CSR* A, stype n_rows, bool used_by_solver){

  assert( A->on_the_device );

  cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
  cusparseIndexBase_t idxbase = CUSPARSE_INDEX_BASE_ZERO;

  // --------------------------- Custom CudaMalloc ---------------------------------
  //CSR *AT = CSRm::init(A->m, A->n, A->nnz, true, true, A->is_symmetric, A->m, 0);
  //CSR *AT = CSRm::init(n_rows, A->full_n, A->nnz, true, true, A->is_symmetric, A->m, 0);
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  CSR *AT = CSRm::init(n_rows, A->full_n, A->nnz, false, true, A->is_symmetric, A->m, 0);
  AT->val = CustomCudaMalloc::alloc_vtype(AT->nnz, (used_by_solver ? 0: 1));
  AT->col = CustomCudaMalloc::alloc_itype(AT->nnz, (used_by_solver ? 0: 1));
  AT->row = CustomCudaMalloc::alloc_itype((AT->n) +1, (used_by_solver ? 0: 1));
  AT->custom_alloced = true;
  // -------------------------------------------------------------------------------
  
  size_t buff_T_size = 0;

  cusparseStatus_t err = cusparseCsr2cscEx2_bufferSize(
    cusparse_h,
    A->n,
    A->m,
    A->nnz,
    A->val,
    A->row,
    A->col,
    AT->val,
    AT->row,
    AT->col,
    CUDA_R_64F,
    copyValues,
    idxbase,
    CUSPARSE_CSR2CSC_ALG1,
    &buff_T_size
  );

  CHECK_CUSPARSE(err);
  assert(buff_T_size);


  void *buff_T = NULL;
  cudaMalloc_CNT
  CHECK_DEVICE(  cudaMalloc(&buff_T, buff_T_size) );

  err = cusparseCsr2cscEx2(
    cusparse_h,
    A->n,
    A->m,
    A->nnz,
    A->val,
    A->row,
    A->col,
    AT->val,
    AT->row,
    AT->col,
    CUDA_R_64F,
    copyValues,
    idxbase,
    CUSPARSE_CSR2CSC_ALG1,
    buff_T
  );
  CHECK_CUSPARSE(err);

  CHECK_DEVICE( cudaFree(buff_T) );

  return AT;
}




template <int OP_TYPE>
__global__
void CSRm::_CSR_vector_mul_mini_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype *A_val, itype *A_row, itype *A_col, vtype *x, vtype *y){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  vtype T_i = 0.;

  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=MINI_WARP_SIZE){
    if(OP_TYPE == 0)
        T_i += (alpha * A_val[j]) * __ldg(&x[A_col[j]]);
    else if(OP_TYPE == 1)
        T_i += A_val[j] * __ldg(&x[A_col[j]]);
    else if(OP_TYPE == 2)
        T_i += -A_val[j] * __ldg(&x[A_col[j]]);
  }

  for(int k=MINI_WARP_SIZE >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0){
    if(OP_TYPE == 0)
        y[warp] = T_i + (beta * y[warp]);
    else if(OP_TYPE == 1)
        y[warp] = T_i;
    else if(OP_TYPE == 2)
        y[warp] = T_i + y[warp];
  }
}


vector<vtype>* CSRm::CSRVector_product_adaptive_miniwarp(cusparseHandle_t cusparse_h, CSR *A, vector<vtype> *x, vector<vtype> *y, vtype alpha, vtype beta){
  itype n = A->n;

  int density = A->nnz / A->n;

  int min_w_size;

  if(density < MINI_WARP_THRESHOLD_2)
    min_w_size = 2;
  else if(density < MINI_WARP_THRESHOLD_4)
    min_w_size = 4;
  else if(density < MINI_WARP_THRESHOLD_8)
    min_w_size = 4;
  else
    min_w_size = 16;


  if(y == NULL){
    assert( beta == 0. );
    Vectorinit_CNT
    y = Vector::init<vtype>(n, true, true);     // OK perchè vettore di output
  }

  gridblock gb = gb1d(n, BLOCKSIZE, true, min_w_size);

  if(alpha == 1. && beta == 0.){
    CSRm::_CSR_vector_mul_mini_warp<1><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
  }else if(alpha == -1. && beta == 1.){
    CSRm::_CSR_vector_mul_mini_warp<2><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
  }else{
    CSRm::_CSR_vector_mul_mini_warp<0><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
  }
  return y;
}

__global__
void _vector_sync( vtype* local_x, itype local_n, vtype *what_to_receive_d, itype receive_n, itype post_local, vtype* x, itype x_n ) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (id < x_n) {
        if (id < post_local) {
            x[id] = what_to_receive_d[id];
        } else {
            if ( id < post_local + local_n ) {
                x[id] = local_x[id - post_local];
            } else {
                x[id] = what_to_receive_d[id - local_n];
            }
        }
    }
    
}

vector<vtype>* CSRm::CSRVector_product_adaptive_miniwarp_new(cusparseHandle_t cusparse_h, CSR *A, vector<vtype> *local_x, vector<vtype> *w, vtype alpha, vtype beta) {
  PUSH_RANGE(__func__,4)
    
  _MPI_ENV;
  
  if(nprocs == 1) {
    vector<vtype> *w_ = NULL;
    if (w == NULL) {
        Vectorinit_CNT
        w_ = Vector::init<vtype>(A->n, true, true);
        Vector::fillWithValue(w_, 0.);
    } else
        w_ = w;
    CSRm::CSRVector_product_adaptive_miniwarp(cusparse_h, A, local_x, w_, alpha, beta);
    return(w_);
  }
  
  assert(A->shrinked_flag == 1);
  
  CSR* A_ = CSRm::init(A->n, (gstype)A->shrinked_m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);
  A_->row = A->row;
  A_->val = A->val;
  A_->col = A->shrinked_col;

  
  // ----------------------------------------- temp check -----------------------------------------
  assert( A->halo.to_receive_n + local_x->n == A_->m );
  // ----------------------------------------------------------------------------------------------
  
  int post_local = A->post_local;

  vector<vtype> *x_ = NULL;
  if ( A->halo.to_receive_n > 0 ) {
    x_ = Vector::init<vtype>(A_->m, false, true);
    if(A_->m>xsize) {
        if(xsize>0) {
            CHECK_DEVICE( cudaFree(xvalstat) );
        }
        xsize = A_->m;
        cudaMalloc_CNT
        CHECK_DEVICE( cudaMalloc(&xvalstat,sizeof(vtype)*xsize) );
    }
    x_->val = xvalstat;
    gridblock gb = gb1d(A_->m, BLOCKSIZE);
    _vector_sync<<<gb.g, gb.b>>>(local_x->val, A->n, A->halo.what_to_receive_d, A->halo.to_receive_d->n, post_local, x_->val, x_->n);
  } else {
    x_ = local_x;
  }
  
  vector<vtype> *w_ = NULL;
  if (w == NULL) {
    Vectorinit_CNT
    w_ = Vector::init<vtype>(A->n, true, true);
    Vector::fillWithValue(w_, 1.);
  } else
    w_ = w;
  CSRm::CSRVector_product_adaptive_miniwarp(cusparse_h, A_, x_, w_, alpha, beta);
  
  // --------------------------------------- print -----------------------------------------
//   vector<vtype> *what_to_receive_d = Vector::init<vtype>(A->halo.to_receive_n, false, true);
//   what_to_receive_d->val = A->halo.what_to_receive_d;
//   
//   PICO_PRINT(  \
//     fprintf(fp, "A->halo:\n\tto_receive: "); Vector::print(A->halo.to_receive, -1, fp); \
//     fprintf(fp, "\tto_send: "); Vector::print(A->halo.to_send, -1, fp); \
//     fprintf(fp, "post_local = %d\n", post_local); \
//     fprintf(fp, "what_to_receive_d: "); Vector::print(what_to_receive_d, -1, fp); \
//     fprintf(fp, "local_x: "); Vector::print(local_x, -1, fp); \
//     fprintf(fp, "x_: "); Vector::print(x_, -1, fp); \
//   )
//   
//   std::free(what_to_receive_d);
  // ---------------------------------------------------------------------------------------
  
  if ( A->halo.to_receive_n > 0 )
      std::free(x_);
//     Vector::free(x_);
  A_->col = NULL;
  A_->row = NULL;
  A_->val = NULL;
  std::free(A_);
  
  POP_RANGE
  return(w_);
}

template <int OP_TYPE>
__global__
void _shifted_CSR_vector_mul_mini_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype *A_val, itype *A_row, itype *A_col, vtype *x, vtype *y, itype shift){

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  vtype T_i = 0.;

  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=MINI_WARP_SIZE){
    if(OP_TYPE == 0)
        T_i += (alpha * A_val[j]) * __ldg(&x[A_col[j]]);
    else if(OP_TYPE == 1)
        T_i += A_val[j] * __ldg(&x[A_col[j]]);
    else if(OP_TYPE == 2)
        T_i += -A_val[j] * __ldg(&x[A_col[j]]);
  }

  for(int k=MINI_WARP_SIZE >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0){
    if(OP_TYPE == 0)
        y[shift+warp] = T_i + (beta * y[shift+warp]);
    else if(OP_TYPE == 1)
        y[shift+warp] = T_i;
    else if(OP_TYPE == 2)
        y[shift+warp] = T_i + y[shift+warp];
  }
}


vector<vtype>* CSRm::shifted_CSRVector_product_adaptive_miniwarp(CSR *A, vector<vtype> *x, vector<vtype> *y, itype shift, vtype alpha, vtype beta){
  itype n = A->n;

  int density = A->nnz / A->n;

  int min_w_size;

  if(density < MINI_WARP_THRESHOLD_2)
    min_w_size = 2;
  else if(density < MINI_WARP_THRESHOLD_4)
    min_w_size = 4;
  else if(density < MINI_WARP_THRESHOLD_8)
    min_w_size = 8;
  else
    min_w_size = 16;

  if(y == NULL){
    assert( beta == 0. );
    Vectorinit_CNT
    y = Vector::init<vtype>(n, true, true);
  }

  gridblock gb = gb1d(n, BLOCKSIZE, true, min_w_size);

  if(alpha == 1. && beta == 0.){
    _shifted_CSR_vector_mul_mini_warp<1><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val, shift);
  }else if(alpha == -1. && beta == 1.){
    _shifted_CSR_vector_mul_mini_warp<2><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val, shift);
  }else{
    _shifted_CSR_vector_mul_mini_warp<0><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val, shift);
  }
  return y;
}


__global__
void _shifted_CSR_vector_mul_mini_warp2(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype *A_val, itype *A_row, itype *A_col, vtype *x, vtype *y, itype shift){

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  vtype T_i = 0.;

  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=MINI_WARP_SIZE){
      T_i += A_val[j] * __ldg(&x[A_col[j]-shift]);

  }

  for(int k=MINI_WARP_SIZE >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0){
      y[warp] = T_i;
  }
}


vector<vtype>* CSRm::shifted_CSRVector_product_adaptive_miniwarp2(CSR *A, vector<vtype> *x, vector<vtype> *y, itype shift, vtype alpha, vtype beta){
  PUSH_RANGE(__func__, 6)
    
  itype n = A->n;

  int density = A->nnz / A->n;

  int min_w_size;

  if(density < MINI_WARP_THRESHOLD_2)
    min_w_size = 2;
  else if(density < MINI_WARP_THRESHOLD_4)
    min_w_size = 4;
  else if(density < MINI_WARP_THRESHOLD_8)
    min_w_size = 8;
  else
    min_w_size = 16;

  if(y == NULL){
    assert( beta == 0. );
    Vectorinit_CNT
    y = Vector::init<vtype>(n, true, true);
  }

  gridblock gb = gb1d(n, BLOCKSIZE, true, min_w_size);

  if(alpha == 1. && beta == 0.)
    _shifted_CSR_vector_mul_mini_warp2<<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val, shift);

  POP_RANGE
  return y;
}


vector<vtype>* CSRVector_product_MPI(CSR *Alocal, vector<vtype> *x, int type){

  assert(Alocal->on_the_device);
  assert(x->on_the_device);


  if(type == 0){

    // everyone gets all
    Vectorinit_CNT
    vector<vtype> *out = Vector::init<vtype>(x->n, true, true);
    Vector::fillWithValue(out, 0.);

    CSRm::shifted_CSRVector_product_adaptive_miniwarp(Alocal, x, out, Alocal->row_shift);

    vector<vtype> *h_out = Vector::copyToHost(out);
    vector<vtype> *h_full_out = Vector::init<vtype>(x->n, true, false);
    //Vector::print(h_out);

    CHECK_MPI( MPI_Allreduce(
      h_out->val,
      h_full_out->val,
      h_full_out->n * sizeof(vtype),
      MPI_DOUBLE,
      MPI_SUM,
      MPI_COMM_WORLD
    ) );

    Vector::free(out);
    Vector::free(h_out);

    return h_full_out;

  }else if (type == 1){

    // local vector outputs
    Vectorinit_CNT
    vector<vtype> *out = Vector::init<vtype>(Alocal->n, true, true);
    CSRm::shifted_CSRVector_product_adaptive_miniwarp(Alocal, x, out, 0);
    return out;

  }else{
    assert(false);
    return NULL;
  }
}

vector<vtype>* CSRm::CSRVector_product_CUSPARSE(cusparseHandle_t cusparse_h, CSR *A, vector<vtype> *x, vector<vtype> *y, bool trans, vtype alpha, vtype beta){
  itype n = A->n;

  itype y_n;
  itype m = A->m;

  cusparseOperation_t op;
  if(trans){
    op = CUSPARSE_OPERATION_TRANSPOSE;
    y_n = m;
    assert( x->n == n );
  }else{
    op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    y_n = n;
    assert( x->n == m );
  }

  if(y == NULL){
    assert( beta == 0. );
    Vectorinit_CNT
    y = Vector::init<vtype>(y_n, true, true);
  }

  cusparseDnVecDescr_t x_dnVecDescr, y_dnVecDescr;


  cusparseCreateDnVec(
    &x_dnVecDescr,
    (int64_t)x->n,
    (void*) x->val,
    CUDA_R_64F
  );

  cusparseCreateDnVec(
    &y_dnVecDescr,
    (int64_t)y_n,
    (void**) y->val,
    CUDA_R_64F
  );


  cusparseSpMatDescr_t A_descr;
  cusparseStatus_t err;


  err = cusparseCreateCsr(
    &A_descr,
    (int64_t) A->n,
    (int64_t) A->m,
    (int64_t) A->nnz,
    A->row,
    A->col,
    A->val,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F
  );

  CHECK_CUSPARSE(err);

  size_t buffer_size = 0;

  err = cusparseSpMV_bufferSize(
    cusparse_h,
    op,
    &alpha,
    A_descr,
    x_dnVecDescr,
    &beta,
    y_dnVecDescr,
    CUDA_R_64F,
    CUSPARSE_MV_ALG_DEFAULT,
    &buffer_size
  );

  CHECK_CUSPARSE(err);

  void *buffer = NULL;
  cudaMalloc_CNT
  CHECK_DEVICE(  cudaMalloc(&buffer, buffer_size) );


  err = cusparseSpMV(
    cusparse_h,
    op,
    &alpha,
    A_descr,
    x_dnVecDescr,
    &beta,
    y_dnVecDescr,
    CUDA_R_64F,
    CUSPARSE_MV_ALG_DEFAULT,
    buffer
  );

  CHECK_CUSPARSE(err);


  CHECK_DEVICE( cudaFree(buffer) );
  cusparseDestroyDnVec(x_dnVecDescr);
  cusparseDestroyDnVec(y_dnVecDescr);
  cusparseDestroySpMat(A_descr);

  return y;
}


vtype CSRm::vectorANorm(cublasHandle_t cublas_h, CSR *A, vector<vtype> *x){
  _MPI_ENV;

  if(nprocs > 1)
    assert(A->n != x->n);

  vector<vtype> *temp = CSRVector_product_MPI(A, x, 1);

  vector<vtype> *x_shift = Vector::init<vtype>(A->n, false, true);

  x_shift->val = x->val + A->row_shift;
  vtype local_norm = Vector::dot(cublas_h, temp, x_shift), norm;

  if(nprocs > 1){
    CHECK_MPI( MPI_Allreduce(
      &local_norm,
      &norm,
      1,//sizeof(vtype),
      MPI_DOUBLE,//MPI_BYTE,
      MPI_SUM,
      MPI_COMM_WORLD
    ) );
    local_norm = norm;
  }

  norm = sqrt(local_norm);

  Vector::free(temp);

  return norm;
}

void CSRm::partialAlloc(CSR *A, bool init_row, bool init_col, bool init_val){

  assert(A->on_the_device);

  cudaError_t err;
  if(init_val){
    cudaMalloc_CNT
    err = cudaMalloc( (void**) &A->val, A->nnz * sizeof(vtype) );
    CHECK_DEVICE(err);
  }
  if(init_col){
    cudaMalloc_CNT
    err = cudaMalloc( (void**) &A->col, A->nnz * sizeof(itype) );
    CHECK_DEVICE(err);
  }
  if(init_row){
    cudaMalloc_CNT
    err = cudaMalloc( (void**) &A->row, (A->n + 1) * sizeof(itype) );
    CHECK_DEVICE(err);
  }
}

__global__ void _getDiagonal_warp(itype n, int MINI_WARP_SIZE, vtype *A_val, itype *A_col, itype *A_row, vtype *D){

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  itype j_start = A_row[warp];
  itype j_stop = A_row[warp+1];

  int j_d = WARP_SIZE, j;

  for(j = j_start+lane; ; j+=MINI_WARP_SIZE){
    int is_diag = __ballot_sync(warp_mask, ( (j < j_stop) && (A_col[j] == warp) ) ) ;
    j_d = __clz(is_diag);
    if(j_d != MINI_WARP_SIZE)
      break;
  }

}


//SUPER temp kernel
__global__ void _getDiagonal(itype n, vtype *val, itype *col, itype *row, vtype *D, itype row_shift){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  itype r = i;
  itype j_start = row[i];
  itype j_stop = row[i+1];

  int j;
  for(j=j_start; j<j_stop; j++){
    itype c = col[j];

    // if is a diagonal element
    if(c == (r /* + row_shift */)){
      D[i] = val[j];
      break;
    }
  }
}


// get a copy of the diagonal
vector<vtype>* CSRm::diag(CSR *A){
  Vectorinit_CNT
  vector<vtype> *D = Vector::init<vtype>(A->n, true, true);

  gridblock gb = gb1d(D->n, BLOCKSIZE);
  _getDiagonal<<<gb.g, gb.b>>>(D->n, A->val, A->col, A->row, D->val, A->row_shift);

  return D;
}


__global__ void _row_sum_2(itype n, vtype *A_val, itype *A_row, itype *A_col, vtype *sum){

  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  vtype local_sum = 0.;

  int j;
  for(j=A_row[i]; j<A_row[i+1]; j++)
      local_sum += fabs(A_val[j]);

    sum[i] = local_sum;
}

vector<vtype>* CSRm::absoluteRowSum(CSR *A, vector<vtype> *sum){
  _MPI_ENV;

  assert(A->on_the_device);

  if(sum == NULL){
    Vectorinit_CNT
    sum = Vector::init<vtype>(A->n, true, true);
  }else{
    assert(sum->on_the_device);
  }

  gridblock gb = gb1d(A->n, BLOCKSIZE, false);
  _row_sum_2<<<gb.g, gb.b>>>(A->n, A->val, A->row, A->col, sum->val);

  return sum;
}

__global__
void _CSR_vector_mul_prolongator(itype n, vtype *A_val, itype *A_row, itype *A_col, vtype *x, vtype *y){

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid >= n)
    return;

  itype j = A_row[tid];
  y[tid] += A_val[j] * __ldg(&x[A_col[j]]);

}

vector<vtype>* CSRm::CSRVector_product_prolungator(CSR *A, vector<vtype> *x, vector<vtype> *y){
  itype n = A->n;

  assert( A->on_the_device );
  assert( x->on_the_device );

  gridblock gb = gb1d(n, BLOCKSIZE);

  _CSR_vector_mul_prolongator<<<gb.g, gb.b>>>(n, A->val, A->row, A->col, x->val, y->val);

  return y;
}


// checks if the colmuns are in the correct order
void CSRm::checkColumnsOrder(CSR *A_){

  CSR *A;
  if(A_->on_the_device)
    A = CSRm::copyToHost(A_);
  else
    A = A_;

  for (int i=0; i<A->n; i++){
    itype _c = -1;
    for (int j=A->row[i]; j<A->row[i+1]; j++){
      itype c = A->col[j];

      if(c < _c){
        printf("WRONG ORDER COLUMNS: %d %d-%d\n", i, c, _c);
        exit(1);
      }
      if(c > _c){
        _c = c;
      }
      if(c > A->m-1){
        printf("WRONG COLUMN TO BIG: %d %d-%d\n", i, c, _c);
        exit(1);
      }
    }
  }
  if(A_->on_the_device)
  CSRm::free(A);
}

#define MY_EPSILON 0.0001
void CSRm::checkMatrix(CSR *A_, bool check_diagonal){
  _MPI_ENV;
  CSR *A = NULL;

  if(A_->on_the_device)
    A = CSRm::copyToHost(A_);
  else
    A = A_;

  for (int i=0; i < A->n; i++){
    for (int j=A->row[i]; j<A->row[i+1]; j++){
      int c = A->col[j];
      double v = A->val[j];
      int found = 0;
      for(int jj = A->row[c]; jj < A->row[c+1]; jj++){
        if(A->col[jj] == i){
            found = 1;
            vtype diff = abs(v - A->val[jj]);
            if(A->val[jj] != v && diff >= MY_EPSILON){
              printf("\n\nNONSYM %lf %lf %lf\n\n", v, A->val[jj], diff);
              exit(1);
            }
            break;
        }
      }
      if(!found){
        printf("BAD[%d]: %d %d\n", myid, i, c);
        exit(1);
      }
    }
  }

  checkColumnsOrder(A);

  if(check_diagonal){
    printf("CHECKING DIAGONAL\n");
    for (int i=0; i < A->n; i++){
      bool found = false;
      for(int j=A->row[i]; j<A->row[i+1]; j++){
        int c = A->col[j];
        vtype v = A->val[j];
        if(c == i && v > 0.)
          found = true;
      }
      if(!found){
        printf("MISSING ELEMENT DIAG %d\n", i);
        exit(1);
      }
    }
    if(A_->on_the_device)
    CSRm::free(A);
    }
}

void CSRm::checkMatching(vector<itype> *v_){
  _MPI_ENV;
  vector<itype> *V = NULL;
  if(v_->on_the_device)
    V = Vector::copyToHost(v_);
  else
    V = v_;

  for(int i=0; i<V->n; i++){
    int v = i;
    int u = V->val[i];

    if(u == -1)
      continue;

    if(V->val[u] != v){
      printf("\n%d]ERROR-MATCHING: %d %d %d\n", myid, i, v, V->val[u]);
      exit(1);
    }
  }

  if(v_->on_the_device)
    Vector::free(V);
}
