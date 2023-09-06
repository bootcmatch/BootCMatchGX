#include "basic_kernel/smoother/relax.h"
#include "utility/function_cnt.h"

halo_info *H_halo_info;

relaxContext Relax::context;

void Relax::initContext(itype n){
    Vectorinit_CNT
    Relax::context.temp_buffer = Vector::init<vtype>(n, true, true);
    Relax::context.n = n;
}

void Relax::set_n_context(itype n){
    Relax::context.temp_buffer->n = n;
}

void Relax::freeContext(){
    Relax::set_n_context(Relax::context.n);
    Vector::free(Relax::context.temp_buffer);
}


void aggregateSolution_complete(CSR *A, vector<vtype> *u, itype local_n, itype shift){
  _MPI_ENV;
  // get your slice
  vtype *u_val = u->val + shift;
  itype full_n = u->n;

  vector<vtype> *h_u_local = Vector::init<vtype>(local_n, true, false);
  vector<vtype> *h_u = Vector::init<vtype>(full_n, true, false);

  // cpy slice to host
  CHECK_DEVICE( cudaMemcpy(h_u_local->val, u_val, local_n * sizeof(vtype), cudaMemcpyDeviceToHost) );

  int row_ns[nprocs];
  int chunks[nprocs], chunkn[nprocs];

  for(int i=0; i<nprocs-1; i++)
    row_ns[i] = full_n / nprocs;
  row_ns[nprocs-1] = full_n - ( (full_n / nprocs) * (nprocs - 1) );

  for(int i=0; i<nprocs; i++){
    chunkn[i] = row_ns[i] * sizeof(vtype);
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }

  CHECK_MPI(
    MPI_Allgatherv(
      h_u_local->val,
      local_n * sizeof(vtype),
      MPI_BYTE,
      h_u->val,
      chunkn,
      chunks,
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  CHECK_DEVICE( cudaMemcpy(u->val, h_u->val, full_n * sizeof(vtype), cudaMemcpyHostToDevice) );

  Vector::free(h_u_local);
  Vector::free(h_u);
}

void aggregateSolution(CSR *A, vector<vtype> *u, itype local_n, itype shift, int level){
  _MPI_ENV;

#if SMART_VECTOR_AGGREGATION == 1
  assert(A->full_n == u->n);
#if DEBUG_SMART_AGG == 1
  vector<vtype> *ucp = Vector::clone(u);
  aggregateSolution_complete(A, ucp, local_n, shift);
#endif
  halo_sync(H_halo_info[level], A, u);
#if DEBUG_SMART_AGG == 1
  if(!checkSync(A, u, ucp, level)){
    printf("ERROR: level: %d myid: %d\n", level, myid);
    DIE;
  }
  Vector::free(ucp);
#endif

#else
  // get your slice
  aggregateSolution_complete(A, u, local_n, shift);
#endif
}


template <int OP_TYPE, int MINI_WARP_SIZE>
__global__
void _jacobi_it(itype n, vtype relax_weight, vtype *A_val, itype *A_row, itype *A_col, vtype *D, vtype *u, vtype *f, vtype *u_, itype shift){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;
  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  itype rows, rowe;
#ifdef SHARED_MEM_START_END_ROW
 __shared__ itype p2Arow[jacobi_BLOCKSIZE/MINI_WARP_SIZE][2];
 if(lane<2) {
     p2Arow[threadIdx.x/MINI_WARP_SIZE][lane] = A_row[warp+lane];
 }
 __syncwarp(warp_mask);
 rows = p2Arow[threadIdx.x/MINI_WARP_SIZE][0];
 rowe = p2Arow[threadIdx.x/MINI_WARP_SIZE][1];
#else
 rows = A_row[warp];
 rowe = A_row[warp+1];
#endif

  vtype T_i = 0.;

  // A * u
  for(int j=rows+lane; j<rowe; j+=MINI_WARP_SIZE){
    T_i += A_val[j] * __ldg(&u[A_col[j]]);
  }

  // WARP sum reduction
  #pragma unroll MINI_WARP_SIZE
  for(int k=MINI_WARP_SIZE >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }


  if(lane == 0){
    u_[warp+shift] = ( (-T_i + f[warp]) / D[warp] ) + u[warp+shift];
  }
}


template <int OP_TYPE, int MINI_WARP_SIZE>
__global__
void _jacobi_it_full(itype n, vtype relax_weight, vtype *A_val, itype *A_row, itype *A_col, vtype *D, vtype *u, vtype *f, vtype *u_){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  vtype T_i = 0.;

  // A * u
  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=MINI_WARP_SIZE){
    T_i += A_val[j] * __ldg(&u[A_col[j]]);
  }

  // WARP sum reduction
  #pragma unroll MINI_WARP_SIZE
  for(int k=MINI_WARP_SIZE >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0){
    if(OP_TYPE == 0)
      u_[warp] = ( (-T_i + f[warp]) / D[warp] ) + u[warp];
    else if(OP_TYPE == 1)
      u_[warp] = ( -T_i / D[warp] ) + u[warp];
  }

}


vector<vtype>* jacobi_adaptive_miniwarp(cusparseHandle_t cusparse_h, cublasHandle_t cublas_h, int k, CSR *A, vector<vtype> *u, vector<vtype> **u_, vector<vtype> *f, vector<vtype> *D, vtype relax_weight, int level){
  _MPI_ENV;

  assert(f != NULL);
  vector<vtype> *swap_temp;

  itype n = A->n;
  int density = A->nnz / A->n;

  aggregateSolution(A, u, A->n, A->row_shift, level);

  if(density <  MINI_WARP_THRESHOLD_2){
    //miniwarp 2
    gridblock gb = gb1d(n, BLOCKSIZE, true, 2);

    for(int i=0; i<k; i++){

      _jacobi_it<0, 2><<<gb.g, gb.b>>>(
        n,
        relax_weight,
        A->val, A->row, A->col,
        D->val,
        u->val, 
        f->val, 
        (*u_)->val, 
        A->row_shift
      );

      swap_temp = u;
      aggregateSolution(A, *u_, A->n, A->row_shift, level);
      u = *u_;

      *u_ = swap_temp;
    }
 }else if (density <  MINI_WARP_THRESHOLD_4){
    //miniwarp 4
    gridblock gb = gb1d(n, BLOCKSIZE, true, 4);

    for(int i=0; i<k; i++){
      _jacobi_it<0, 4><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val,A->row_shift);
      swap_temp = u;
      aggregateSolution(A, *u_, A->n, A->row_shift, level);
      u = *u_;
      *u_ = swap_temp;
    }
  }else if (density <  MINI_WARP_THRESHOLD_8){
    //miniwarp 8
    gridblock gb = gb1d(n, BLOCKSIZE, true, 8);

    for(int i=0; i<k; i++){
      _jacobi_it<0, 8><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val, A->row_shift);
      swap_temp = u;
      aggregateSolution(A, *u_, A->n, A->row_shift, level);
      u = *u_;
      *u_ = swap_temp;
    }
  }else if(density < MINI_WARP_THRESHOLD_16){
    //miniwarp 16
    gridblock gb = gb1d(n, BLOCKSIZE, true, 16);

    for(int i=0; i<k; i++){
      _jacobi_it<0, 16><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val, A->row_shift);
      swap_temp = u;
      aggregateSolution(A, *u_, A->n, A->row_shift, level);
      u = *u_;
      *u_ = swap_temp;
    }
  }else{
    //miniwarp 32
    gridblock gb = gb1d(n, BLOCKSIZE, true, 32);

    for(int i=0; i<k; i++){
      _jacobi_it<0, 32><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val, A->row_shift);
      swap_temp = u;
      aggregateSolution(A, *u_, A->n, A->row_shift, level);
      u = *u_;
      *u_ = swap_temp;
    }
  }

  return u;
}

void jacobi_adaptive_miniwarp_coarsest(cusparseHandle_t cusparse_h, cublasHandle_t cublas_h, int k, CSR *A, vector<vtype> *u, vector<vtype> **u_, vector<vtype> *f, vector<vtype> *D, vtype relax_weight){

  _MPI_ENV;
  assert(f != NULL);
  vector<vtype> *swap_temp;

  itype n = A->n;
  int density = A->nnz / A->n;

  if(density <  MINI_WARP_THRESHOLD_2){
    //miniwarp 2
    gridblock gb = gb1d(n, BLOCKSIZE, true, 2);

    for(int i=0; i<k; i++){
      _jacobi_it_full<0, 2><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      if(k==1) break;
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }else if (density <  MINI_WARP_THRESHOLD_4){
    //miniwarp 4
    gridblock gb = gb1d(n, BLOCKSIZE, true, 4);

    for(int i=0; i<k; i++){
      _jacobi_it_full<0, 4><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      if(k==1) break;
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }else if (density <  MINI_WARP_THRESHOLD_8){
    //miniwarp 8
    gridblock gb = gb1d(n, BLOCKSIZE, true, 8);

    for(int i=0; i<k; i++){
      _jacobi_it_full<0, 8><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      if(k==1) break;
      cudaDeviceSynchronize();
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }else if(density < MINI_WARP_THRESHOLD_16){
    //miniwarp 16
    gridblock gb = gb1d(n, BLOCKSIZE, true, 16);

    for(int i=0; i<k; i++){
      _jacobi_it_full<0, 16><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      if(k==1) break;
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }else{
    //miniwarp 32
    gridblock gb = gb1d(n, BLOCKSIZE, true, 32);

    for(int i=0; i<k; i++){
      _jacobi_it_full<0, 32><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      if(k==1) break;
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }
}


void relaxCoarsest(handles *h, int k, CSR* A, vector<vtype>* D, vector<vtype>* M, vector<vtype> *f, int relax_type, vtype relax_weight, vector<vtype> *u, vector<vtype> **u_, itype nlocal, bool forward){
  if(relax_type == 0){
    jacobi_adaptive_miniwarp_coarsest(h->cusparse_h0, h->cublas_h, k, A, u, u_, f, D, relax_weight);
  }else if(relax_type == 4){
    jacobi_adaptive_miniwarp_coarsest(h->cusparse_h0, h->cublas_h, k, A, u, u_, f, M, relax_weight);
  }
}

#include "basic_kernel/smoother/relaxation_sm.cu"
#define SMOOTHER jacobi_adaptive_miniwarp_overlapped

void relax(handles *h, int k, int level, CSR* A, vector<vtype>* D, vector<vtype>* M, vector<vtype> *f, int relax_type, vtype relax_weight, vector<vtype> *u, vector<vtype> **u_, bool forward){
  _MPI_ENV;

  if(nprocs == 1){
    relaxCoarsest(h, k, A, D, M, f, relax_type, relax_weight, u, u_, 0);
    return;
  }

  if(relax_type == 0){
    SMOOTHER(h->cusparse_h0, h->cublas_h, k, A, u, u_, f, D, relax_weight, level);
  }else if(relax_type == 4){
    SMOOTHER(h->cusparse_h0, h->cublas_h, k, A, u, u_, f, M, relax_weight, level);
  }
}
