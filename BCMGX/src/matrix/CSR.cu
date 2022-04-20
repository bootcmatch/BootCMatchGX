#include "matrix/CSR.h"
#include "utility/cudamacro.h"

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


CSR* CSRm::init(itype n, itype m, itype nnz, bool allocate_mem, bool on_the_device, bool is_symmetric, itype full_n, itype row_shift){

  assert(n > 0 && m > 0 && nnz >= 0);

  CSR *A = NULL;

  // on the host
  A = (CSR*) malloc(sizeof(CSR));
  CHECK_HOST(A);

  A->nnz = nnz;
  A->n = n;
  A->m = m;

  A->on_the_device = on_the_device;
  A->is_symmetric = false;

  A->full_n = full_n;
  A->row_shift = row_shift;

  A->rows_to_get = NULL;

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

void CSRm::print(CSR *A, int type, int limit=0){
  CSR *A_ = NULL;

  if(A->on_the_device)
    A_ = CSRm::copyToHost(A);
  else
    A_ = A;

  switch(type) {
    case 0:
      printf("ROW: %d (%d)\n\t", A_->n, A_->full_n);
      if(limit == 0)
        limit = A_->full_n + 1;
      for(int i=0; i<limit; i++){
        printf("%d ", A_->row[i]);
      }
      break;
    case 1:
      printf("COL:\n\t");
      if(limit == 0)
        limit = A_->nnz;
      for(int i=0; i<limit; i++){
        printf("%d ", A_->col[i]);
      }
      break;
    case 2:
      printf("VAL:\n\t");
      if(limit == 0)
        limit = A_->nnz;
      for(int i=0; i<limit; i++){
        printf("%lf ", A_->val[i]);
      }
      break;
  }
  printf("\n\n");

  if(A->on_the_device)
    CSRm::free(A_);
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
    cudaError_t err;
    err = cudaFree(A->val);
    CHECK_DEVICE(err);
    err = cudaFree(A->col);
    CHECK_DEVICE(err);
    err = cudaFree(A->row);
    CHECK_DEVICE(err);
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
  // Free the halo_info halo halo_info halo; 
  std::free(A->descr);
  std::free(A);
}


void CSRm::freeStruct(CSR *A){
  CHECK_CUSPARSE( cusparseDestroyMatDescr(*A->descr) );
  std::free(A->descr);
  std::free(A);
}


void CSRm::printInfo(CSR *A){
  printf("Device?: %d\n", A->on_the_device);
  printf("nnz: %d\n", A->nnz);
  printf("n: %d\n", A->n);
  printf("m: %d\n", A->m);
  printf("full_n: %d\n", A->full_n);
  printf("row_shift: %d\n", A->row_shift);
  if(A->is_symmetric)
    printf("SYMMETRIC\n");
  else
    printf("GENERAL\n");
	printf("\n");
}

CSR* CSRm::copyToDevice(CSR *A){

  assert( !A->on_the_device );

  itype n, m, nnz;

  n = A->n;
  m = A->m;

  nnz = A->nnz;

  // alocate CSR matrix on the device memory
  CSR *A_d = CSRm::init(n, m, nnz, true, true, A->is_symmetric, A->full_n, A->row_shift);

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

  // alocate CSR matrix on the device memory
  CSR *A = CSRm::init(n, m, nnz, true, false, A_d->is_symmetric, A_d->full_n, A_d->row_shift);

  cudaError_t err;

  err = cudaMemcpy(A->val, A_d->val, nnz * sizeof(vtype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A->row, A_d->row, (n + 1) * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A->col, A_d->col, nnz * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);

  return A;
}



__global__
void _shift_cols(itype n, itype *col, itype shift){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  col[i] += shift;
}

void CSRm::shift_cols(CSR* A, itype shift){
  assert(A->on_the_device);
  gridblock gb = gb1d(A->nnz, BLOCKSIZE);
  _shift_cols<<<gb.g, gb.b>>>(A->nnz, A->col, shift);
}

// return a copy of A->T
CSR* CSRm::T(cusparseHandle_t cusparse_h, CSR* A){

  assert( A->on_the_device );

  cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
  cusparseIndexBase_t idxbase = CUSPARSE_INDEX_BASE_ZERO;

  CSR *AT = CSRm::init(A->m, A->n, A->nnz, true, true, A->is_symmetric, A->m, 0);

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

  cudaFree(buff_T);

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
    min_w_size = 8;
  else
    min_w_size = 16;


  if(y == NULL){
    assert( beta == 0. );
    y = Vector::init<vtype>(n, true, true);
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
    y = Vector::init<vtype>(n, true, true);
  }

  gridblock gb = gb1d(n, BLOCKSIZE, true, min_w_size);

  if(alpha == 1. && beta == 0.)
    _shifted_CSR_vector_mul_mini_warp2<<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val, shift);

  return y;
}


vector<vtype>* CSRVector_product_MPI(CSR *Alocal, vector<vtype> *x, int type){

  assert(Alocal->on_the_device);
  assert(x->on_the_device);


  if(type == 0){

    // everyone gets all
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


  cudaFree(buffer);
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
    err = cudaMalloc( (void**) &A->val, A->nnz * sizeof(vtype) );
    CHECK_DEVICE(err);
  }
  if(init_col){
    err = cudaMalloc( (void**) &A->col, A->nnz * sizeof(itype) );
    CHECK_DEVICE(err);
  }
  if(init_row){
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
    if(c == (r + row_shift)){
      D[i] = val[j];
    }
  }
}


// get a copy of the diagonal
vector<vtype>* CSRm::diag(CSR *A){
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
