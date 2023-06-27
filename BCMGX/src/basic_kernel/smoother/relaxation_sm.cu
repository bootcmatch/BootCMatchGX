#include <cub/cub.cuh>

#include "basic_kernel/smoother/relax.h"
#include "basic_kernel/halo_communication/local_permutation.h"
#include "basic_kernel/halo_communication/extern.h"
#include "basic_kernel/matrix/vector.h"

#include "utility/function_cnt.h"

struct Matched{
    int compare;
    __host__ __device__ __forceinline__
    Matched(int compare) : compare(compare) {}
    __host__ __device__ __forceinline__
    bool operator()(const int &a) const {
        return (a != compare);
    }
};

overlappedSmootherList* init_overlappedSmootherList(int nm){

  overlappedSmootherList *osl = (overlappedSmootherList*) malloc(sizeof(overlappedSmootherList));
  CHECK_HOST(osl);

  osl->nm = nm;

  osl->oss = (overlappedSmoother*) malloc( sizeof(overlappedSmoother) * nm );
  CHECK_HOST(osl->oss);

  osl->local_stream = (cudaStream_t*) malloc( sizeof(cudaStream_t) );
  osl->comm_stream = (cudaStream_t*) malloc( sizeof(cudaStream_t) );

  CHECK_DEVICE( cudaStreamCreate(osl->local_stream) );
  CHECK_DEVICE( cudaStreamCreate(osl->comm_stream) );

  return osl;
}

void free_overlappedSmootherList(overlappedSmootherList *osl){

  for(int i=0; i<osl->nm; i++){
    if(osl->oss[i].loc_n)
      Vector::free(osl->oss[i].loc_rows);

    if(osl->oss[i].needy_n)
      Vector::free(osl->oss[i].needy_rows);
  }

  free(osl->oss);

  CHECK_DEVICE( cudaStreamDestroy(*osl->local_stream) );
  CHECK_DEVICE( cudaStreamDestroy(*osl->comm_stream) );

  free(osl);
}

__global__
void _findNeedyRows(itype n, int MINI_WARP_SIZE, itype *A_row, itype *A_col, itype *needy_rows, itype *loc_rows, itype *needy_n, itype *loc_n, gstype start, gstype end){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;
  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  itype rows, rowe;
  rows = A_row[warp];
  rowe = A_row[warp+1];

  int flag = 0;
  for(int j=rows+lane; j<rowe; j+=MINI_WARP_SIZE){
    itype c = A_col[j];
    if(c < start || c >= end)
      flag = 1;
  }

  unsigned needy = __any_sync(warp_mask, flag);

  if(lane == 0){
    if(needy){
      atomicAdd(needy_n, 1);
      needy_rows[warp] = warp;
    }else{
      atomicAdd(loc_n, 1);
      loc_rows[warp] = warp;
    }
  }
}

void setupOverlappedSmoother(CSR *A, overlappedSmoother *os){
  _MPI_ENV;

  Vectorinit_CNT
  vector<itype> *loc_rows = Vector::init<itype>(A->n, true, true);
  Vector::fillWithValue(loc_rows, -1);

  Vectorinit_CNT
  vector<itype> *needy_rows = Vector::init<itype>(A->n, true, true);
  Vector::fillWithValue(needy_rows, -1);

  scalar<itype> *loc_n = Scalar::init(0, true);
  scalar<itype> *needy_n = Scalar::init(0, true);

  int warpsize = CSRm::choose_mini_warp_size(A);
  gridblock gb = gb1d(A->n, BLOCKSIZE, warpsize);
  _findNeedyRows<<<gb.g, gb.b>>>(A->n, warpsize, A->row, A->col, needy_rows->val, loc_rows->val, needy_n->val, loc_n->val, A->row_shift, A->row_shift+A->n);

  itype *_loc_n = Scalar::getvalueFromDevice(loc_n);
  itype *_needy_n = Scalar::getvalueFromDevice(needy_n);

  os->loc_n = *_loc_n;
  os->needy_n = *_needy_n;


  if(os->loc_n){
    Vectorinit_CNT
    os->loc_rows = Vector::init<itype>(os->loc_n, true, true);
  }
  if(os->needy_n){
    Vectorinit_CNT
    os->needy_rows = Vector::init<itype>(os->needy_n, true, true);
  }

  Matched m(-1);
  scalar<itype> *d_num_selected_out = Scalar::init<itype>(0, true);
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  if(os->loc_n){
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, loc_rows->val, os->loc_rows->val, d_num_selected_out->val, A->n, m);
    // Allocate temporary storage
    cudaMalloc_CNT
    CHECK_DEVICE( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, loc_rows->val, os->loc_rows->val, d_num_selected_out->val, A->n, m);
  }

  if(os->needy_n){
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, needy_rows->val, os->needy_rows->val, d_num_selected_out->val, A->n, m);
    // Allocate temporary storage
    if(d_temp_storage == NULL) {
      cudaMalloc_CNT
      CHECK_DEVICE( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
    }
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, needy_rows->val, os->needy_rows->val, d_num_selected_out->val, A->n, m);
  }

  cudaFree(d_temp_storage);
  Scalar::free(d_num_selected_out);
  Scalar::free(loc_n);
  Scalar::free(needy_n);
  free(_loc_n);
  free(_needy_n);
  Vector::free(loc_rows);
  Vector::free(needy_rows);
}

// CPU version of setupOverlappedSmoothe (for debugging)
void setupOverlappedSmoother_cpu(CSR *A, overlappedSmoother *os){
  _MPI_ENV;

  if(ISMASTER)
    printf("[TEMP] copy for setupOverlappedSmoother\n");
  CSR *A_host = CSRm::copyToHost(A);

  vector<itype> *loc_rows = Vector::init<itype>(A->n, true, false);
  vector<itype> *needy_rows = Vector::init<itype>(A->n, true, false);

  itype loc_n = 0;
  itype needy_n = 0;

  for(itype i=0; i<A->n; i++){
    bool needy = false;
    for(itype j=A_host->row[i]; j<A_host->row[i+1]; j++){
      itype c = A_host->col[j];
      if(c < A->row_shift || c >= A->row_shift+A->n){
        needy_rows->val[needy_n] = i;
        needy_n++;
        needy = true;
        break;
      }
    }
    if(needy)
      continue;
    loc_rows->val[loc_n] = i;
    loc_n++;
  }

  loc_rows->n = loc_n;
  needy_rows->n = needy_n;

  (*os).loc_n = loc_n;
  (*os).needy_n = needy_n;

  if(loc_n) {
    VectorcopyToDevice_CNT
    (*os).loc_rows = Vector::copyToDevice(loc_rows);
  }
  if(needy_n) {
    VectorcopyToDevice_CNT
    (*os).needy_rows = Vector::copyToDevice(needy_rows);
  }

  Vector::free(loc_rows);
  Vector::free(needy_rows);
}

template <int OP_TYPE, int MINI_WARP_SIZE>
__global__
void _jacobi_it_partial(itype n, itype *rows, vtype relax_weight, vtype *A_val, itype *A_row, itype *A_col, vtype *D, vtype *u, vtype *f, vtype *u_, itype shift){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  // only local rows
  warp = rows[warp];

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

  if(lane == 0)
    u_[warp+shift] = ( (-T_i + f[warp]) / D[warp] ) + u[warp+shift];
}


__forceinline__
__device__
int binsearch(int array[], unsigned int size, int value) {
 unsigned int low, high, medium;
 low=0;
 high=size;
 while(low<high) {
     medium=(high+low)/2;
     if(value > array[medium]) {
       low=medium+1;
     } else {
       high=medium;
     }
 }
 return low;
}

template <int OP_TYPE, int MINI_WARP_SIZE>
__global__
void _jacobi_it_partial_new(itype n, itype *rows, vtype relax_weight, vtype *A_val, itype *A_row, itype *A_col, vtype *D, vtype *u, vtype *f, vtype *u_, gstype shift, itype An, gstype* halo_index, itype halo_index_size, vtype* halo_val, bool balc_flag, int post_local){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  // only local rows
  warp = rows[warp];

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  vtype T_i = 0.;

  // A * u
  
  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=MINI_WARP_SIZE){
        T_i += A_val[j] * __ldg(&halo_val[A_col[j]]);
  }

  // WARP sum reduction
  #pragma unroll MINI_WARP_SIZE
  for(int k=MINI_WARP_SIZE >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0)
    u_[warp] = ( (-T_i + f[warp]) / D[warp] ) + u[warp];
}

__global__
void _relax_sync( vtype* local_x, itype local_n, vtype *what_to_receive_d, itype receive_n, itype post_local, vtype* x, itype x_n ) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < x_n) {
        if (id < post_local) {
            if(id>receive_n) {
                printf("Horror 1 in thread %d\n",id);
            }
            x[id] = what_to_receive_d[id];
        } else {
            if ( id < post_local + local_n ) {
                if((id-post_local)<0 || (id-post_local)>local_n) {
                    printf("Horror 2 in thread %d %d\n",id,post_local);
                }
                x[id] = local_x[id - post_local];
            } else {
                if((id-local_n)<0 || (id-local_n)>receive_n) {
                    printf("Horror 3 in thread %d local_n=%d receive_n=%d x_n=%d post_local=%d\n",id,local_n,receive_n, x_n,post_local);
                }
                x[id] = what_to_receive_d[id - local_n];
            }
        }
    }
    
}

static int relax_xsize=0;
static vtype *relax_xvalstat=NULL;
int srmfb=-1;
#define GET_TIME_J false
#define MAXNTASKS 4096
#define JACOBI_TAG 1234

template <int MINI_WARP_SIZE>
vector<vtype>* internal_jacobi_overlapped(int level, int k, CSR *A, vector<vtype> *u, vector<vtype> **u_, vector<vtype> *f, vector<vtype> *D, vtype relax_weight){
  _MPI_ENV;

  vector<vtype> *swap_temp;
  gridblock gb;
  overlappedSmoother os = osl->oss[level];
  halo_info hi = A->halo;//H_halo_info[level];
  static MPI_Request requests[MAXNTASKS];
  static int ntr=0;
  
  if(A->shrinked_flag==false) {
#if 0
       srmfb=myid;
       printf("Task %d: shrinking col of %x\n",myid,A);
       shrink_col(A,NULL);
       itype H_ASC[A->nnz];
       cudaMemcpy(H_ASC,A->shrinked_col,sizeof(itype)*A->nnz,cudaMemcpyDeviceToHost);
    char filename[256];
    snprintf(filename,sizeof(filename),"ASC_%d_%d",level,myid);
    FILE *fp=fopen(filename,"w");
    if(fp==NULL) {
               fprintf(stderr,"Could not open X\n");
    }
    for(int i=0; i<A->nnz; i++) {
        fprintf(fp,"%d\n",H_ASC[i]);
    }
    fclose(fp);
        itype H_AR[A->n+1];
        cudaMemcpy(H_AR,A->row,sizeof(itype)*(A->n+1),cudaMemcpyDeviceToHost);
    snprintf(filename,sizeof(filename),"AR_%d_%d",level,myid);
    fp=fopen(filename,"w");
    if(fp==NULL) {
               fprintf(stderr,"Could not open X\n");
    }
    for(int i=0; i<(A->n+1); i++) {
        fprintf(fp,"%d\n",H_AR[i]);
    }
    fclose(fp);
#endif
    fprintf(stderr,"A must be shrinked before relaxation!\n");
    exit(1);
  } 
  int post_local = A->post_local;
  vector<vtype> *x_ = NULL;
#if 0
  printf("Task %d, level %d, matrix=%x,to_send_n=%d, to_recv_n=%d, needy_n=%d, u_n=%d, A_n=%d, A_m=%d, A_shrinked_m=%d,  A_full_n=%lu, post_local=%d loc_n=%d\n",myid,level,A,hi.to_send_n,hi.to_receive_n,os.needy_n,u->n,A->n,A->m, A->shrinked_m,A->full_n,post_local,os.loc_n);      
#endif

  for(int i=0; i<k; i++){

    if(i)
      cudaStreamSynchronize(*osl->comm_stream);

    // start get to send
    if(hi.to_send_n){
      assert( hi.what_to_send != NULL );
      assert( hi.what_to_send_d != NULL );
      gridblock gb = gb1d(hi.to_send_n, BLOCKSIZE);
      _getToSend_new<<<gb.g, gb.b, 0, *osl->comm_stream>>>(hi.to_send_d->n, u->val, hi.what_to_send_d, hi.to_send_d->val, A->row_shift);
      CHECK_DEVICE( cudaMemcpyAsync(hi.what_to_send, hi.what_to_send_d, hi.to_send_n*sizeof(vtype), cudaMemcpyDeviceToHost, *osl->comm_stream) );
    }

    x_ = u;
    
    if(os.loc_n){
        // start compute local
        gb = gb1d(os.loc_n, BLOCKSIZE, true, MINI_WARP_SIZE);
        _jacobi_it_partial_new<0, MINI_WARP_SIZE><<<gb.g, gb.b, 0, *osl->local_stream>>>(
            os.loc_n,
            os.loc_rows->val,
            relax_weight,
            A->val, A->row, A->shrinked_col,
            D->val, 
            u->val, 
            f->val, 
            (*u_)->val,
            A->row_shift,
            A->n,
            hi.to_receive_d->val, 
            hi.to_receive_d->n,
	    x_->val,
/*            hi.what_to_receive_d, */
            false,
	    post_local
        );
    }
    

    int j=0;
    for(int t=0; t<nprocs; t++) {
		    if(t==myid) continue;
		    if(hi.to_receive_counts[t]>0) {
			    CHECK_MPI (
			          MPI_Irecv(hi.what_to_receive+(hi.to_receive_spls[t]),hi.to_receive_counts[t],VTYPE_MPI,t,JACOBI_TAG,MPI_COMM_WORLD,requests+j));
			    j++;
			    if(j==MAXNTASKS) {
				   fprintf(stderr,"Too many tasks in jacobi, max is %d\n",MAXNTASKS);
				   exit(1);
			    }
		    }
    }
    ntr=j;
    if(hi.to_send_n){
    cudaStreamSynchronize(*osl->comm_stream);
    }
    for(int t=0; t<nprocs; t++) {
	    if(t==myid) continue;
	    if(hi.to_send_counts[t]>0) {
			    CHECK_MPI (MPI_Send(hi.what_to_send+(hi.to_send_spls[t]),hi.to_send_counts[t],VTYPE_MPI,t,JACOBI_TAG,MPI_COMM_WORLD));
	    }
    }

    // copy received data
    if(hi.to_receive_n){
      if(ntr>0) { CHECK_MPI(MPI_Waitall(ntr,requests,MPI_STATUSES_IGNORE)); }
      assert( hi.what_to_receive != NULL );
      assert( hi.what_to_receive_d != NULL );
      CHECK_DEVICE( cudaMemcpyAsync(hi.what_to_receive_d, hi.what_to_receive, hi.to_receive_n * sizeof(vtype), cudaMemcpyHostToDevice, *osl->comm_stream) );
      if (u->n == A->full_n) {        // PICO
        gb = gb1d(hi.to_receive_n, BLOCKSIZE);
        setReceivedWithMask<<<gb.g, gb.b, 0, *osl->comm_stream>>>(hi.to_receive_n , u->val, hi.what_to_receive_d, hi.to_receive_d->val, A->row_shift);
#if 0
        printf("A->full_n=%d, A->n=%d, A->m=%d, A->shrinked_m=%d\n",A->full_n,A->n,A->m,A->shrinked_m);
#endif
      }

      x_ = Vector::init<vtype>(A->shrinked_m, false, true);
          if(A->shrinked_m>relax_xsize) {
	    cudaError_t err;
      	    if(relax_xsize>0) {
     	           CHECK_DEVICE( cudaFree(relax_xvalstat) );
  	    }
  	    relax_xsize = A->shrinked_m;
            cudaMalloc_CNT
            err=cudaMalloc(&relax_xvalstat,sizeof(vtype)*relax_xsize);
            CHECK_DEVICE(err);
          }
      x_->val = relax_xvalstat;
      gridblock gb = gb1d(A->shrinked_m, BLOCKSIZE);
      //printf("_relax_sync\n");
      _relax_sync<<<gb.g, gb.b,0,*osl->comm_stream>>>(u->val, A->n, hi.what_to_receive_d, hi.to_receive_d->n, post_local, x_->val, x_->n);
#if 0
{
  char filename[256];
  snprintf(filename,sizeof(filename),"x_%d_%d",i,myid);
  FILE *fp=fopen(filename,"w");
  if(fp==NULL) {
               fprintf(stderr,"Could not open X\n");
  }
  Vector::print(x_,-1,fp);
  fclose(fp);
}     
#endif
      // complete computation for halo
      gb = gb1d(os.needy_n, BLOCKSIZE, true, MINI_WARP_SIZE);
      if(os.needy_n) {
            // start compute local
            _jacobi_it_partial_new<0, MINI_WARP_SIZE><<<gb.g, gb.b, 0, *osl->comm_stream>>>(
                os.needy_n,
                os.needy_rows->val,
                relax_weight,
                A->val, A->row, A->shrinked_col,
                D->val, 
                u->val, 
                f->val, 
                (*u_)->val,
                A->row_shift,
                A->n,
                hi.to_receive_d->val, 
                hi.to_receive_d->n,
/*                hi.what_to_receive_d, */
		x_->val,		
                true,
		post_local
            );
      }
    }

    cudaDeviceSynchronize();
    swap_temp = u;
    u = *u_;
#if 0
{
  char filename[256];
  snprintf(filename,sizeof(filename),"u_%d_%d",i,myid);
  FILE *fp=fopen(filename,"w");
  if(fp==NULL) {
               fprintf(stderr,"Could not open X\n");
  }
  Vector::print(u,-1,fp);
  fclose(fp);
}     
#endif
    *u_ = swap_temp;
  }
    

  cudaStreamSynchronize(*osl->local_stream);
  cudaStreamSynchronize(*osl->comm_stream);

  if ( hi.to_receive_n > 0 )
      std::free(x_);

  return u;
}

vector<vtype>* jacobi_adaptive_miniwarp_overlapped(cusparseHandle_t cusparse_h, cublasHandle_t cublas_h, int k, CSR *A, vector<vtype> *u, vector<vtype> **u_, vector<vtype> *f, vector<vtype> *D, vtype relax_weight, int level){
  int density = A->nnz / A->n;

  if(density <  MINI_WARP_THRESHOLD_2)
    return internal_jacobi_overlapped<2>(level, k, A, u, u_, f, D, relax_weight);
  else if (density <  MINI_WARP_THRESHOLD_4)
    return internal_jacobi_overlapped<4>(level, k, A, u, u_, f, D, relax_weight);
  else if (density <  MINI_WARP_THRESHOLD_8)
    return internal_jacobi_overlapped<4>(level, k, A, u, u_, f, D, relax_weight);
  else if(density < MINI_WARP_THRESHOLD_16)
    return internal_jacobi_overlapped<16>(level, k, A, u, u_, f, D, relax_weight);
  else
    return internal_jacobi_overlapped<32>(level, k, A, u, u_, f, D, relax_weight);
}
