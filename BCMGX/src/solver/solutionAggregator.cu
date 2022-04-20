#include "solver/solutionAggregator.h"

__global__
void _getMissingMask(itype nnz, itype *A_col, itype *missing, itype row_shift, itype n){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= nnz)
    return;

  itype col = A_col[i];

  if(col < row_shift || col >= row_shift+n)
    missing[i] = col;
  else
    missing[i] = -1;
}

__global__
void _count(itype nnz, itype *missing, itype *c){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= nnz)
    return;

  if(missing[i])
    atomicAdd(c, 1);
}

void getMissing(CSR *A, vector<itype> **missing, CSR *R){
  _MPI_ENV;

  itype row_ns[nprocs], ends[nprocs];

  CHECK_MPI(
    MPI_Allgather(
      &A->n,
      sizeof(itype),
      MPI_BYTE,
      &row_ns,
      sizeof(itype),
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  ends[0] = row_ns[0];
  for(itype i=1; i<nprocs; i++)
      ends[i] = row_ns[i] + ends[i-1];

  assert(ends[nprocs-1] == A->full_n);

  itype nnz = 0;

  if(R != NULL)
    nnz = R->nnz;
  else
    nnz = A->nnz;

  vector<itype> *mask, *mask_sorted;
  mask = Vector::init<itype>(nnz, true, true);
  scalar<itype> *d_num_selected_out = Scalar::init<itype>(0, true);

  mask_sorted = Vector::init<itype>(nnz, true, true);

  gridblock gb;

  if(R != NULL){
    gb = gb1d(nnz, BLOCKSIZE);
    _getMissingMask<<<gb.g, gb.b>>>(nnz, R->col, mask->val, A->row_shift, A->n);
  }else{
    gb = gb1d(nnz, BLOCKSIZE);
    _getMissingMask<<<gb.g, gb.b>>>(nnz, A->col, mask->val, A->row_shift, A->n);
  }

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortKeys(
    d_temp_storage,
    temp_storage_bytes,
    mask->val,
    mask_sorted->val,
    nnz
  );

  CHECK_DEVICE( cudaMalloc(&d_temp_storage, temp_storage_bytes) );

  cub::DeviceRadixSort::SortKeys(
    d_temp_storage,
    temp_storage_bytes,
    mask->val,
    mask_sorted->val,
    nnz
  );

  cudaFree(d_temp_storage);
  d_temp_storage = NULL;
  temp_storage_bytes = 0;

  cub::DeviceSelect::Unique(
    d_temp_storage,
    temp_storage_bytes,
    mask_sorted->val,
    mask->val,
    d_num_selected_out->val,
    nnz
  );

  CHECK_DEVICE( cudaMalloc(&d_temp_storage, temp_storage_bytes) );

  cub::DeviceSelect::Unique(
    d_temp_storage,
    temp_storage_bytes,
    mask_sorted->val,
    mask->val,
    d_num_selected_out->val,
    nnz
  );

  itype *cp = Scalar::getvalueFromDevice(d_num_selected_out);
  itype c = (*cp)-1;
  free(cp);

  for(int i=0; i<nprocs; i++){
    if(i != myid)
      missing[i]->n = 0;
  }

  if(c > 0){
    itype *missing_flat = NULL;
    missing_flat = (itype*)  malloc(sizeof(itype) * c);
    CHECK_HOST(missing_flat);
    cudaMemcpy(missing_flat, mask->val+1, c * sizeof(itype), cudaMemcpyDeviceToHost);

    itype J = 0, I = 0;
    for(itype i=0; i<c; i++){
      itype j = missing_flat[i];

      CHECK_AGAIN:
      if(j >= ends[J]){

        if(J != myid)
          missing[J]->n = I;

        J++;
        I = 0;
        goto CHECK_AGAIN;
      }
      missing[J]->val[I] = j;
      I++;
    }

    if(I){
      missing[J]->n = I;
    }

    free(missing_flat);
  }
  Vector::free(mask);
  Vector::free(mask_sorted);
  Scalar::free(d_num_selected_out);
}

__global__
void _getToSendMask(itype n, itype *to_send, itype *to_send_mask, itype shift){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  to_send_mask[to_send[i]-shift] = i;
}


halo_info setupAggregation(CSR *A, CSR *R=NULL){
  _MPI_ENV;

  vector<itype> **my_missing = (vector<itype>**) malloc(sizeof(vector<itype> *) * nprocs);
  CHECK_HOST(my_missing);

  for(int i=0; i<nprocs; i++){
    if(i != myid)
      my_missing[i] = Vector::init<itype>(A->nnz, true, false);
    else
      my_missing[i] = NULL;
  }

  getMissing(A, my_missing, R);

  itype total_n = 0;
  for(itype i=0; i<nprocs; i++){
      if(myid == i)
        continue;
    total_n += my_missing[i]->n;
  }

  int *sendcounts = (int*) malloc(sizeof(int)*nprocs);
  int *sdispls = (int*) malloc(sizeof(int)*nprocs);
  CHECK_HOST(sendcounts);
  CHECK_HOST(sdispls);


  vector<itype> *my_missing_flat = NULL;

  if(total_n > 0){
    my_missing_flat = Vector::init<itype>(total_n, true, false);
  }

  itype shift = 0;
  for(itype i=0; i<nprocs; i++){

      if(myid == i){
        sendcounts[i] = 0;
        sdispls[i] = shift;
        continue;
      }

      if(my_missing[i]->n > 0)
        memcpy(my_missing_flat->val+shift, my_missing[i]->val, my_missing[i]->n*sizeof(itype));

      sendcounts[i] = my_missing[i]->n;
      sdispls[i] = shift;
      shift += my_missing[i]->n;
  }

  int *recvcounts = (int*) malloc(sizeof(int)*nprocs);
  CHECK_HOST(recvcounts);
  CHECK_MPI(
    MPI_Alltoall(
      sendcounts,
      1,
      ITYPE_MPI,
      recvcounts,
      1,
      ITYPE_MPI,
      MPI_COMM_WORLD
    )
  );

  int *rdispls = (int*) malloc(sizeof(int)*nprocs);
  CHECK_HOST(rdispls);


  shift = 0;
  itype their_missing_flat_total_n = 0;
  for(itype i=0; i<nprocs; i++){
    rdispls[i] = shift;
    shift += recvcounts[i];
    their_missing_flat_total_n += recvcounts[i];
  }

  vector<itype> *their_missing_flat = NULL;
  if(their_missing_flat_total_n > 0)
    their_missing_flat = Vector::init<itype>(their_missing_flat_total_n, true, false);

  CHECK_MPI(
    MPI_Alltoallv(
      my_missing_flat != NULL ? my_missing_flat->val : NULL,
      sendcounts,
      sdispls,
      ITYPE_MPI,
      their_missing_flat != NULL ? their_missing_flat->val : NULL,
      recvcounts,
      rdispls,
      ITYPE_MPI,
      MPI_COMM_WORLD
    )
  );

  halo_info hi;

  hi.to_receive = my_missing_flat;
  hi.to_receive_n = total_n;
  hi.to_receive_counts = sendcounts;
  hi.to_receive_spls = sdispls;

  hi.what_to_receive = NULL;
  hi.to_receive_d = NULL;
  if(hi.to_receive_n > 0){
    hi.what_to_receive = (vtype*)malloc(sizeof(vtype)*hi.to_receive_n);
    CHECK_HOST(hi.what_to_receive);
    CHECK_DEVICE( cudaMalloc( (void**) &hi.what_to_receive_d , sizeof(vtype)*hi.to_receive_n ) );
    hi.to_receive_d = Vector::copyToDevice(hi.to_receive);
  }

  hi.to_send = their_missing_flat;
  hi.to_send_n = their_missing_flat_total_n;
  hi.to_send_counts = recvcounts;
  hi.to_send_spls = rdispls;

  hi.what_to_send_d = NULL;
  hi.what_to_send = NULL;

  if(hi.to_send_n > 0){
    CHECK_DEVICE(  cudaMallocHost((void**)&hi.what_to_send, sizeof(vtype)*hi.to_send_n) );

    CHECK_HOST(hi.what_to_send);
    CHECK_DEVICE( cudaMalloc( (void**) &hi.what_to_send_d , sizeof(vtype)*hi.to_send_n ) );
    hi.to_send_d = Vector::copyToDevice(hi.to_send);
  }


  for(itype i=0; i<nprocs; i++){
    if(my_missing[i] != NULL)
      Vector::free(my_missing[i]);
  }
  std::free(my_missing);

  return hi;
}


__global__
void _getToSend(itype n, vtype *x, vtype *what_to_send, itype *to_send){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  itype j = to_send[i];
  what_to_send[i] = x[j];
}

__global__
void setReceivedWithMask(itype n, vtype *x, vtype *received, itype *receive_map, itype shift){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  itype j = receive_map[i];
  vtype val = received[i];
  x[j] = val;
}
#define SYNCSOL_TAG 4321
#define MAXNTASKS 1024

void sync_solution(halo_info hi, CSR *A, vector<vtype> *x){
  _MPI_ENV;

  assert(A->on_the_device);
  assert(x->on_the_device);
  static MPI_Request requests[MAXNTASKS];
  static cudaStream_t sync_stream; 
  static int first=1;
  if(first) {
	  first=0;
	  CHECK_DEVICE( cudaStreamCreate(&sync_stream) );
  }

  gridblock gb;

  if(hi.to_send_n){
#if SMART_AGGREGATE_GETSET_GPU == 1
    gridblock gb = gb1d(hi.to_send_n, BLOCKSIZE);
    _getToSend<<<gb.g, gb.b>>>(hi.to_send_d->n, x->val, hi.what_to_send_d, hi.to_send_d->val);

    CHECK_DEVICE( cudaMemcpyAsync(hi.what_to_send, hi.what_to_send_d, hi.to_send_n*sizeof(vtype), cudaMemcpyDeviceToHost, sync_stream) );
#else
    vector<vtype> *x_host = Vector::copyToHost(x);
    int start = 0;
    for(int i=0; i<nprocs; i++){
      int end = start + hi.to_send_counts[i];
      for(int j=start; j<end; j++){
        itype v = hi.to_send->val[j];
        hi.what_to_send[j] = x_host->val[v];
      }
      start = end;
    }
#endif
  }
    int j=0, ntr;
    for(int t=0; t<nprocs; t++) {
                    if(t==myid) continue;
                    if(hi.to_receive_counts[t]>0) {
                            CHECK_MPI (
                                  MPI_Irecv(hi.what_to_receive+(hi.to_receive_spls[t]),hi.to_receive_counts[t],VTYPE_MPI,t,SYNCSOL_TAG,MPI_COMM_WORLD,requests+j));
                            j++;
                            if(j==MAXNTASKS) {
                                   fprintf(stderr,"Too many tasks in sync_solution, max is %d\n",MAXNTASKS);
                                   exit(1);
                            }
                    }
    }
    ntr=j;
    if(hi.to_send_n){
   	 cudaStreamSynchronize(sync_stream);
    }

    for(int t=0; t<nprocs; t++) {
            if(t==myid) continue;
            if(hi.to_send_counts[t]>0) {
                            CHECK_MPI (MPI_Send(hi.what_to_send+(hi.to_send_spls[t]),hi.to_send_counts[t],VTYPE_MPI,t,SYNCSOL_TAG,MPI_COMM_WORLD));
            }
    }

  if(!hi.to_receive_n)
    return;
  if(ntr>0) { CHECK_MPI(MPI_Waitall(ntr,requests,MPI_STATUSES_IGNORE)); }

#if SMART_AGGREGATE_GETSET_GPU == 1
    CHECK_DEVICE( cudaMemcpy(hi.what_to_receive_d, hi.what_to_receive, hi.to_receive_n * sizeof(vtype), cudaMemcpyHostToDevice) );

    gb = gb1d(hi.to_receive_n , BLOCKSIZE);
    setReceivedWithMask<<<gb.g, gb.b>>>(hi.to_receive_n , x->val, hi.what_to_receive_d, hi.to_receive_d->val, A->row_shift);

#else
  vector<vtype> *x_host = Vector::copyToHost(x);
  int start = 0;
  for(int i=0; i<nprocs; i++){
    int end = start + hi.to_receive_counts[i];
    for(int j=start; j<end; j++){
      itype v = hi.to_receive->val[j];
      x_host->val[v] = hi.what_to_receive[j];
    }
    start = end;
  }
  CHECK_DEVICE( cudaMemcpy(x->val, x_host->val, x_host->n * sizeof(vtype), cudaMemcpyHostToDevice) );
  Vector::free(x_host);
#endif
}

bool checkSync(CSR *_A, vector<vtype> *_x0, vector<vtype> *_x1, int level){
  _MPI_ENV;
  CSR *A = CSRm::copyToHost(_A);
  vector<vtype> *x0 = Vector::copyToHost(_x0);
  vector<vtype> *x1 = Vector::copyToHost(_x1);

  bool flag = true;
  for(int i=0; i<A->n; i++){
    for(int j=A->row[i]; j<A->row[i+1]; j++){
      itype col = A->col[j];
      if(x0->val[col] != x1->val[col]){
        printf("n %d] %d} -- col: %d | ", myid, level, col);
        std::cout << x0->val[col] << " ---- " << x1->val[col] << "\n";
        flag = false;
      }    
    }
  }

  CSRm::free(A);
  Vector::free(x0);
  Vector::free(x1);

  return flag;
}


void sync_solution_stream(halo_info hi, CSR *A, vector<vtype> *x, cudaStream_t stream=0){
  _MPI_ENV;

  assert(A->on_the_device);
  assert(x->on_the_device);

  gridblock gb;

  if(hi.to_send_n){
    gridblock gb = gb1d(hi.to_send_n, BLOCKSIZE);
    _getToSend<<<gb.g, gb.b, 0, stream>>>(hi.to_send_d->n, x->val, hi.what_to_send_d, hi.to_send_d->val);
    CHECK_DEVICE( cudaMemcpyAsync(hi.what_to_send, hi.what_to_send_d, hi.to_send_n*sizeof(vtype), cudaMemcpyDeviceToHost, stream) );
  }

  cudaStreamSynchronize(stream);

  CHECK_MPI(
    MPI_Alltoallv(
      hi.what_to_send,
      hi.to_send_counts,
      hi.to_send_spls,
      VTYPE_MPI,
      hi.what_to_receive,
      hi.to_receive_counts,
      hi.to_receive_spls,
      VTYPE_MPI,
      MPI_COMM_WORLD
    )
  );


  if(!hi.to_receive_n)
    return;

    CHECK_DEVICE( cudaMemcpyAsync(hi.what_to_receive_d, hi.what_to_receive, hi.to_receive_n * sizeof(vtype), cudaMemcpyHostToDevice, stream) );
    gb = gb1d(hi.to_receive_n , BLOCKSIZE);
    setReceivedWithMask<<<gb.g, gb.b, 0, stream>>>(hi.to_receive_n , x->val, hi.what_to_receive_d, hi.to_receive_d->val, A->row_shift);
}

