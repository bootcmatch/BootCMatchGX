#include "AMG.h"

#include "basic_kernel/matrix/matrixIO.h"
#include "utility/timing.h"
// #include "basic_kernel/custom_cudamalloc/custom_cudamalloc.h"
#include "utility/cudamacro.h"

#include "utility/function_cnt.h"


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


__global__
void _write_T_warp(itype n, int MINI_WARP_SIZE, vtype *A_val, itype *A_col, itype *A_row, itype shift){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);
  vtype t;

  itype j_stop = A_row[warp+1];

  for(int j=A_row[warp]+lane; j<j_stop; j+=MINI_WARP_SIZE){
    itype c = A_col[j] /* - shift */;

    if(c < 0 || c>=n){
      continue;
    }


    if(warp < c)
       break;

    int nc = A_row[c+1] - A_row[c];

    int jj=binsearch(A_col+A_row[c], nc, warp /* +shift */);

    t = A_val[jj+A_row[c]];
    A_val[j] = t;
  }
}

__global__
void _makeAH_warp(itype n, int AH_MINI_WARP_SIZE, vtype *A_val, itype *A_col, itype *A_row, vtype *w, vtype *C, vtype *AH_val, itype *AH_col, itype *AH_row, itype row_shift){

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  itype warp = tid / AH_MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % AH_MINI_WARP_SIZE;
  itype j_stop = A_row[warp+1];

  for(int j=A_row[warp]+lane; j<j_stop; j+=AH_MINI_WARP_SIZE){
    itype c = A_col[j];

    if(c < 0 /* row_shift */ || c >= /* row_shift+ */ n){
      itype offset = c > (warp /* + row_shift */) ? warp + 1  : warp;
      AH_val[j - offset] = 99999.;
      AH_col[j - offset] = c;
    }else{

      if(c != warp /* + row_shift */){
        vtype a = A_val[j];
        itype offset = c > (warp /* + row_shift */) ? warp + 1  : warp;
        AH_col[j - offset] = c;
        vtype norm = c > (warp /* + row_shift */) ? C[warp] + C[c /* -row_shift */] : C[c /* -row_shift */] + C[warp];
        if(norm > DBL_EPSILON){
          vtype w_temp = c > (warp /* + row_shift */)? w[warp] * w[c /* -row_shift */] : w[c /* -row_shift */] * w[warp];
          AH_val[j - offset] = 1. - ( (2. * a * w_temp) / norm);
        }else {
          AH_val[j - offset] = DBL_EPSILON;
	}
      }
    }
  }

  if(lane == 0){
    AH_row[warp+1] = j_stop - (warp + 1);
  }

  if(tid == 0){
    // set the first index of the row pointer to 0
    AH_row[0] = 0;
  }
}

__global__
void _makeC(stype n, vtype *val, itype *col, itype *row, vtype *w, vtype *C, itype row_shift){

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
    if(c == r/* +row_shift */){
      C[r] = val[j]*w[r]*w[r] /* pow(w[r], 2) */;
      break;
    }
  }
}

CSR* makeAH(CSR *A, vector<vtype> *w){
  _MPI_ENV;
  static int cnt=0;
  assert(A->on_the_device);
  assert(w->on_the_device);

  itype n;
  n = A->n;

  // init a vector on the device
  vector<vtype> *C = Vector::init<vtype>(A->n, false, true);
  C->val = (vtype*) MemoryPool::local[0];

  gridblock gb = gb1d(n, BLOCKSIZE, false);
  // only local access to w, c must be local but with shift
  _makeC<<<gb.g, gb.b>>>(n, A->val, A->col, A->row, w->val, C->val, A->row_shift);
  
  //Diagonal MUST be non-empty!
  // ----------------------------------------- custom cudaMalloc -------------------------------------------
//   CSR *AH = CSRm::init(A->n, A->m, (A->nnz - A->n), true, true, A->is_symmetric, A->full_n, A->row_shift);
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//   CSR *AH = CSRm::init(A->n, A->m, (A->nnz - A->n), false, true, A->is_symmetric, A->full_n, A->row_shift);
//   AH->val = CustomCudaMalloc::alloc_vtype(AH->nnz);
//   AH->col = CustomCudaMalloc::alloc_itype(AH->nnz);
//   AH->row = CustomCudaMalloc::alloc_itype((AH->n) +1);
//   AH->custom_alloced = true;
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  CSR *AH = CSRm::init(A->n, A->m, (A->nnz - A->n), false, true, A->is_symmetric, A->full_n, A->row_shift);
  AH->val = AH_glob_val;
  AH->col = AH_glob_col;
  AH->row = AH_glob_row;
  // -------------------------------------------------------------------------------------------------------
  
  int miniwarp_size = CSRm::choose_mini_warp_size(A);
  gb = gb1d(n, BLOCKSIZE, true, miniwarp_size);
  _makeAH_warp<<<gb.g, gb.b>>>(n, miniwarp_size, A->val, A->col, A->row, w->val, C->val, AH->val, AH->col, AH->row, AH->row_shift);

  free(C);
  cnt++;

  return AH;
}



struct AbsMin{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &lhs, const T &rhs) const {
      T ab_lhs = fabs(lhs);
      T ab_rhs = fabs(rhs);
      return ab_lhs < ab_rhs  ? ab_lhs  : ab_rhs;
    }
};

void     *d_temp_storage_max_min = NULL;
vtype *min_max = NULL;
// find the max (op_type==0) or the absolute min (op_type==1) in the input device array (with CUB utility)
vtype* find_Max_Min(vtype *a, stype n, int op_type){
  size_t   temp_storage_bytes = 0;

  cudaError_t err;
  if(min_max==NULL) {
      cudaMalloc_CNT
	  err = cudaMalloc((void**)&min_max, sizeof(vtype) * 1);
	  CHECK_DEVICE(err);
  }

  if(op_type == 0){
    cub::DeviceReduce::Max(d_temp_storage_max_min, temp_storage_bytes, a, min_max, n);
    // Allocate temporary storage
    cudaMalloc_CNT
    err = cudaMalloc(&d_temp_storage_max_min, temp_storage_bytes);
    CHECK_DEVICE(err);
    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage_max_min, temp_storage_bytes, a, min_max, n);
  }else if(op_type == 1){
    AbsMin absmin;
    cub::DeviceReduce::Reduce(d_temp_storage_max_min, temp_storage_bytes, a, min_max, n, absmin, DBL_MAX);
    // Allocate temporary storage
    cudaMalloc_CNT
    err = cudaMalloc(&d_temp_storage_max_min, temp_storage_bytes);
    CHECK_DEVICE(err);
    // Run max-reduction
    cub::DeviceReduce::Reduce(d_temp_storage_max_min, temp_storage_bytes, a, min_max, n, absmin, DBL_MAX);
  }

//  err = cudaFree(d_temp_storage);
  CHECK_DEVICE(err);

  return min_max;
}


__global__ void _make_w(stype nnz, vtype *val, vtype min){

  stype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= nnz)
    return;
  vtype scratch=fabs(val[i]);
//  val[i] = log(  scratch>DBL_EPSILON?scratch:DBL_EPSILON / (0.999 * (min))  );
  val[i] = log(  scratch?scratch:EPS / (0.999 * (min))  );  
// if(!(fabs(scratch)>0)) {printf("make_w: %d %14.12g %14.12g %14.12g\n",i,EPS,DBL_EPSILON,log(DBL_EPSILON/(0.999*(min))));}
}


CSR* toMaximumProductMatrix(CSR *AH){
  _MPI_ENV;
  assert(AH->on_the_device);

  stype nnz = AH->nnz;
  // find the min value
  vtype *min = find_Max_Min(AH->val, nnz, 1);

  vtype h_local_min;
  CHECK_DEVICE( cudaMemcpy(&h_local_min, min, sizeof(vtype), cudaMemcpyDeviceToHost) );
//  CHECK_DEVICE( cudaFree(min) );
#if 0
  if(!(fabs(h_local_min)>0)) {
          fprintf(stderr,"Error for Task %d, size of AH=%d, full size of AH=%d, h_local_min=%g, count=%d\n",myid,AH->n,AH->full_n,h_local_min,cnt);
	  exit(0);
  }
#endif  
  if((fabs(h_local_min))<DBL_EPSILON) {
          h_local_min=DBL_EPSILON;
  }

  gridblock gb = gb1d(nnz, BLOCKSIZE, false);
  _make_w<<<gb.g, gb.b>>>(nnz, AH->val, h_local_min);
  
  return AH;
}


vector<itype>* suitor(handles *h, CSR *A, vector<vtype> *w){
  PUSH_RANGE(__func__, 7)
    
  _MPI_ENV;
  TIMER_DEF;
  assert(A->on_the_device && w->on_the_device);

  if(DETAILED_TIMING && ISMASTER) {
    cudaDeviceSynchronize();
//     TIME::start();
    TIMER_START;
  }
  CSR *AH = makeAH(A, w);

  CSR *W = toMaximumProductMatrix(AH);

  if(DETAILED_TIMING && ISMASTER){
    cudaDeviceSynchronize();
//     TOTAL_MAKEAHW_TIME += TIME::stop();
    TIMER_STOP;
    TOTAL_MAKEAHW_TIME += TIMER_ELAPSED;
  }

  vector<itype> *_M = NULL;

  if(DETAILED_TIMING && ISMASTER) {
    cudaDeviceSynchronize();
//     TIME::start();
    TIMER_START;
  }

  vector<vtype> *ws_buffer;
  vector<itype> *mutex_buffer;
  itype n = W->n;

  ws_buffer = Vector::init<vtype>(n , false, true);
  ws_buffer->val = (vtype*) MemoryPool::local[0];

  mutex_buffer = Vector::init<itype>(n , false, true);
  mutex_buffer->val = (itype*) MemoryPool::local[1];

  _M = Vector::init<itype>(n, false, true);
  _M->val = (itype*) MemoryPool::local[2];

  int warp_size = CSRm::choose_mini_warp_size(W);
  gridblock gb = gb1d(n, BLOCKSIZE, true, warp_size);
  _write_T_warp<<<gb.g, gb.b>>>(n, warp_size, W->val, W->col, W->row, W->row_shift);

  approx_match_gpu_suitor(h, A, W, _M, ws_buffer, mutex_buffer);

  if(DETAILED_TIMING && ISMASTER){
//     SUITOR_TIME += TIME::stop();
      TIMER_STOP;
      SUITOR_TIME += TIMER_ELAPSED;
  }
  free(ws_buffer);
  free(mutex_buffer);

  // --- custom cudaMalloc ---
//   CSRm::free(W);
  // >>>>>>>>>>>>>>>>>>>>>>>>>
  std::free(W);
  // -------------------------
  
  POP_RANGE

  return _M;
}
