#include "utility/myMPI.h"
#include <cuda_runtime.h>
#include "utility/cudamacro.h"

template <typename T>
__inline__
void chop_array_MPI_same(int nprocs, int n, int *chunks, int *chunkn){
  int i;
  int e4chunk = n / nprocs * sizeof(T);
  for(i=0; i<nprocs-1; i++){
    chunkn[i] = e4chunk;
    chunks[i] = (i)*e4chunk;
  }
  chunkn[nprocs-1] = (n * sizeof(T))  - i * e4chunk;
  chunks[i] = e4chunk * (nprocs-1);
}

template <typename T>
__inline__
void chop_array_MPI(int nprocs, int n, int n_local, int *chunks, int *chunkn){
  itype ns[nprocs];

  CHECK_MPI(
    MPI_Allgather(
      &n_local,
      sizeof(itype),
      MPI_BYTE,
      &ns,
      sizeof(itype),
      MPI_BYTE,
      MPI_COMM_WORLD
      )
    );


  int i;
  itype tot = 0;
  for(i=0; i<nprocs-1; i++){
    chunkn[i] = ns[i] * sizeof(T);
    chunks[i] = tot;
    tot += chunkn[i];
  }
  chunkn[nprocs-1] = (n * sizeof(T))  - tot;
  chunks[i] = tot;
}

template <typename T>
vector<T>* aggregateVector(vector<T> *u_local, itype full_n, vector<T> *u=NULL){
  _MPI_ENV;

  vector<T> *h_u_local = Vector::copyToHost(u_local);

  vector<T> *h_u = Vector::init<T>(full_n, true, false);

  int chunks[nprocs], chunkn[nprocs];
  chop_array_MPI<vtype>(nprocs, full_n, u_local->n, chunks, chunkn);

  CHECK_MPI(
    MPI_Allgatherv(
      h_u_local->val,
      u_local->n * sizeof(T),
      MPI_BYTE,
      h_u->val,
      chunkn,
      chunks,
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  if(u == NULL)
    u = Vector::copyToDevice(h_u);
  else
    CHECK_DEVICE( cudaMemcpy(u->val, h_u->val, h_u->n * sizeof(T), cudaMemcpyHostToDevice) );

  Vector::free(h_u_local);
  Vector::free(h_u);

  return u;
}

void aggregateFullPartialVector(vector<vtype> *u, itype local_n, itype shift){
  _MPI_ENV;
  // get your slice
  vtype *u_val = u->val + shift;
  itype full_n = u->n;

  vector<vtype> *h_u_local = Vector::init<vtype>(local_n, true, false);
  vector<vtype> *h_u = Vector::init<vtype>(full_n, true, false);

  // cpy slice to host
  CHECK_DEVICE( cudaMemcpy(h_u_local->val, u_val, local_n * sizeof(vtype), cudaMemcpyDeviceToHost) );

  int chunks[nprocs], chunkn[nprocs];

  chop_array_MPI<vtype>(nprocs, u->n, local_n, chunks, chunkn);

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

__global__
void _split_local(itype nstart, itype nrow, itype *Arow, vtype *Aval, itype *Acol, itype *Alocal_row, vtype *Alocal_val, itype *Alocal_col, itype *nnz){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= nrow)
    return;

  itype shift = Arow[nstart];
  itype is = i + nstart;
  itype j_start = Arow[is];
  itype j_stop = Arow[is+1];

  int j;
  Alocal_row[i] = Arow[is] - shift;
  for(j=j_start; j<j_stop; j++){
    Alocal_val[j-shift] = Aval[j];
    Alocal_col[j-shift] = Acol[j];
  }

  if(i == nrow-1){
    *nnz = Arow[nrow+nstart] - shift;
    Alocal_row[nrow] = Arow[is+1] - shift;
  }

}

CSR* split_local(CSR *A){
  _MPI_ENV;
  assert(A->on_the_device && A->n == A->full_n);

  itype rowsxproc = 0;
  //Split A

  int nrows[nprocs];
  rowsxproc = A->n / nprocs;
  for(itype i=0; i<nprocs-1; i++) {
    nrows[i] = rowsxproc;
  }
  nrows[nprocs-1] = A->n - (rowsxproc * (nprocs-1));

  int nstart = 0;
  for(int j=0; j<myid; j++)
    nstart += nrows[j];

  CSR *Alocal = CSRm::init(nrows[myid], A->m, A->nnz, true, true, false, A->n, nstart);

  scalar<itype> *nnz = Scalar::init<itype>(-1, true);

  gridblock gb = gb1d(nrows[myid], BLOCKSIZE);
  _split_local<<<gb.g, gb.b>>>(nstart, nrows[myid], A->row, A->val, A->col, Alocal->row, Alocal->val, Alocal->col, nnz->val);

  int* h_nnz = Scalar::getvalueFromDevice(nnz);
  Scalar::free(nnz);

  Alocal->nnz = *h_nnz;

  return Alocal;
}


CSR* split_MatrixMPI(CSR *A){
  _MPI_ENV;

  itype colxproc[nprocs];
  itype rowsxproc = 0;

  if(ISMASTER){
    assert(!A->on_the_device);
    //Split A
    rowsxproc = A->n / nprocs;
    for(itype i=1; i<nprocs; i++) {
      colxproc[i-1] = A->row[i*rowsxproc] - A->row[(i-1)*rowsxproc];
    }
    colxproc[nprocs-1] = A->row[A->n] - A->row[(nprocs-1)*rowsxproc];
  }


  itype n, m;
  if(ISMASTER){
    n = A->n;
    m = A->m;
  }

  CHECK_MPI(
    MPI_Bcast(&n, sizeof(itype), MPI_BYTE, 0, MPI_COMM_WORLD)
  );

  CHECK_MPI(
    MPI_Bcast(&m, sizeof(itype), MPI_BYTE, 0, MPI_COMM_WORLD)
  );

  if( (nprocs>1) && myid==(nprocs-1) )
    // compute the number of rows for the last process
     rowsxproc = n - ( (n / nprocs) * (nprocs - 1) );
  else
    // compute the number of rows for the process
    rowsxproc = n / nprocs;

  itype mycol = 0;
  // send columns numbers to each process
  CHECK_MPI(
    MPI_Scatter(
      colxproc,
      sizeof(itype),
      MPI_BYTE,
      &mycol,
      sizeof(itype),
      MPI_BYTE,
      0,
      MPI_COMM_WORLD
    )
  );


  int chunks[nprocs], chunkn[nprocs];
  chop_array_MPI_same<itype>(nprocs, n, chunks, chunkn);
  itype rows_shift = chunks[myid] / sizeof(itype);

  CSR *Alocal = CSRm::init(rowsxproc, m, mycol, true, false, false, n, rows_shift);

  // get row pointers
  CHECK_MPI(
    MPI_Scatterv(
      myid ? NULL : A->row,
      chunkn,
      chunks,
      MPI_BYTE,
      Alocal->row,
      sizeof(itype) * rowsxproc,
      MPI_BYTE,
      0,
      MPI_COMM_WORLD
    )
  );
  // set the last pointer in the row array
  Alocal->row[rowsxproc] = Alocal->row[0] + mycol;

  // get columns
  for(int i=0; i<nprocs; i++) {
      chunkn[i] = (int)(colxproc[i] * sizeof(itype));
      chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }

  CHECK_MPI(
    MPI_Scatterv(
      myid ? NULL : A->col,
      chunkn,
      chunks,
      MPI_BYTE,
      Alocal->col,
      sizeof(itype) * mycol,
      MPI_BYTE,
      0,
      MPI_COMM_WORLD
    )
  );

  // get values
  for(int i=0; i<nprocs; i++) {
      chunkn[i] = (int)(colxproc[i] * sizeof(vtype));
      chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  CHECK_MPI(
    MPI_Scatterv(
      myid ? NULL : A->val,
      chunkn,
      chunks,
      MPI_BYTE,
      Alocal->val,
      sizeof(vtype) * mycol,
      MPI_BYTE,
      0,
      MPI_COMM_WORLD
    )
  );

  // shift row pointers
  if(myid>0) {
    itype shift = Alocal->row[0];
    for(int i=0; i<=Alocal->n; i++)
      Alocal->row[i] -= shift;
  }

  CSR *d_Alocal = CSRm::copyToDevice(Alocal);
  CSRm::free(Alocal);

  return d_Alocal;
}


CSR* join_MatrixMPI(CSR *Alocal){
  _MPI_ENV;

  assert(nprocs > 1);
  assert(!Alocal->on_the_device);

  itype row_ns[nprocs];

  //send rows sizes
  CHECK_MPI(
    MPI_Allgather(
      &Alocal->n,
      sizeof(itype),
      MPI_BYTE,
      &row_ns,
      sizeof(itype),
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  itype nnzs[nprocs];

  //send nnz sizes
  CHECK_MPI(
    MPI_Allgather(
      &Alocal->nnz,
      sizeof(itype),
      MPI_BYTE,
      &nnzs,
      sizeof(itype),
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  itype full_n = 0;
  itype full_nnz = 0;
  CSR *A;
  int chunkn[nprocs], chunks[nprocs];

  if(ISMASTER){

    for(int i=0; i<nprocs; i++){
      full_n += row_ns[i];
      full_nnz += nnzs[i];
    }

    assert(full_n == Alocal->full_n);

    A = CSRm::init(full_n, Alocal->m, full_nnz, true, false, false, full_n, 0);

    // gather rows
    for(int i=0; i<nprocs; i++){
      chunkn[i] = row_ns[i] * sizeof(itype);
      chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
    }
    chunkn[nprocs-1] += 1 * sizeof(itype);
  }

  itype rn = Alocal->n * sizeof(itype);
  if(myid == nprocs-1)
    rn += 1; // +1 for the last process

  CHECK_MPI(
    MPI_Gatherv(
      Alocal->row,
      rn,
      MPI_BYTE,
      myid ? NULL : A->row,
      chunkn,
      chunks,
      MPI_BYTE,
      0,
      MPI_COMM_WORLD
    )
  );


  if(ISMASTER){
    /* reset the row number */
    itype rowoffset = 0;
    itype th = row_ns[0];
    int j=0;
    for(int i=0; i<Alocal->full_n; i++){
        // next piece
        if(i >= th && (j<(nprocs)) ){
          rowoffset += nnzs[j];
          j++;
          th += row_ns[j];
        }
        A->row[i] += rowoffset;
      }

      A->row[A->full_n] = nnzs[0];
      for(int i=1; i<nprocs; i++) {
        A->row[A->full_n] += nnzs[i];
      }
  }
  // gather columns
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnzs[i] * sizeof(itype);
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  CHECK_MPI(
    MPI_Gatherv(
      Alocal->col,
      Alocal->nnz * sizeof(itype),
      MPI_BYTE,
      myid ? NULL : A->col,
      chunkn,
      chunks,
      MPI_BYTE,
      0,
      MPI_COMM_WORLD
    )
  );

  // gather value
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnzs[i] * sizeof(vtype);
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  CHECK_MPI(
    MPI_Gatherv(
      Alocal->val,
      Alocal->nnz * sizeof(vtype),
      MPI_BYTE,
      myid ? NULL : A->val,
      chunkn,
      chunks,
      MPI_BYTE,
      0,
      MPI_COMM_WORLD
    )
  );

  return A;

}


int stringCmp( const void *a, const void *b)
{
     return strcmp((const char*)a, (const char*)b);

}

#if false
#define MAXLEN 128
void  assignDeviceToProcess(int *p2myrank)
{
       char     host_name[MPI_MAX_PROCESSOR_NAME];
       char (*host_names)[MPI_MAX_PROCESSOR_NAME];
       char gpuname[MAXLEN];
       MPI_Comm nodeComm;


       int i, n, namelen, color, rank, nprocs, myrank,gpu_per_node;
       size_t bytes;
       int dev, err1;
       struct cudaDeviceProp deviceProp;

       /* Check if the device has been alreasy assigned */

       MPI_Comm_rank(MPI_COMM_WORLD, &rank);
       MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
       MPI_Get_processor_name(host_name,&namelen);

       bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
       host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);

       strcpy(host_names[rank], host_name);

       for (n=0; n<nprocs; n++)
       {
        MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
       }

       qsort(host_names, nprocs,  sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

       color = 0;

       for (n=0; n<nprocs; n++)
       {
         if(n>0&&strcmp(host_names[n-1], host_names[n])) color++;
         if(strcmp(host_name, host_names[n]) == 0) break;
       }

       MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);

       MPI_Comm_rank(nodeComm, &myrank);
       MPI_Comm_size(nodeComm, &gpu_per_node);

       p2myrank[0]=myrank;

        /* Find out how many DP capable GPUs are in the system and their device number */
       int deviceCount,slot=0;
       int *devloc;
       cudaGetDeviceCount(&deviceCount);
       printf("deviceCount: %d\n", deviceCount);
       devloc=(int *)malloc(deviceCount*sizeof(int));
       devloc[0]=0;
       for (dev = 0; dev < deviceCount; ++dev)
        {
          printf("%d %d\n", myrank, dev);
        cudaGetDeviceProperties(&deviceProp, dev);
           devloc[slot]=dev;
           slot++;
        }
       printf ("Assigning device %d  to process on node %s rank %d \n",devloc[myrank], host_name, rank );
       /* Assign device to MPI process, initialize BLAS and probe device properties */
       CHECK_DEVICE(cudaSetDevice(devloc[myrank]));
       snprintf(gpuname,sizeof(gpuname),"%d",devloc[myrank]);
       if(setenv("CUDA_VISIBLE_DEVICES",gpuname,1)<0) {
           fprintf(stderr,"Could not set CUDA_VISIBLE_DEVICES env variable\n");
           exit(1);
       }
       return;
}
void  assignDeviceToProcess(int localRank){
  int dev, deviceCount, slot=0;
  struct cudaDeviceProp deviceProp;

  CHECK_DEVICE(cudaGetDeviceCount(&deviceCount));

  if(!deviceCount) throw std::exception();

  int *devloc = new int[deviceCount];
  for(dev = 0; dev < deviceCount; ++dev){
      CHECK_DEVICE(cudaGetDeviceProperties(&deviceProp, dev));
      if(deviceProp.major > 1) devloc[slot++] = dev;
  }
  CHECK_DEVICE(cudaSetDevice(devloc[localRank % slot]));
  printf ("Assigning device %d to local rank: %d\n", devloc[localRank % slot], localRank );
}
#endif

void checkMatrixMPI(CSR *A, bool check_diagonal=true){
  _MPI_ENV;
  assert(A->on_the_device);
  CSR *h_Alocal = CSRm::copyToHost(A);
  CSR *h_Afull = join_MatrixMPI(h_Alocal);

  if(ISMASTER)
    CSRm::checkMatrix(h_Afull, check_diagonal);

  CSRm::free(h_Alocal);
  if(ISMASTER)
    CSRm::free(h_Afull);
}


bool _check_in_A(CSR *A, int i, int J){
  for (int j=A->row[i]; j<A->row[i+1]; j++){
      int c = A->col[j];
      if(c == J)
        return true;
    }
  return false;
}

void check_A_P_MPI(CSR *A_local, CSR *P_){
  _MPI_ENV;
  assert(A_local->on_the_device);
  CSR *h_Alocal = CSRm::copyToHost(A_local);
  CSR *A = join_MatrixMPI(h_Alocal);

  if(ISMASTER){
    CSR *P = CSRm::copyToHost(P_);
    CSRm::checkMatrix(A);

    for (int i=0; i < P->n; i++){
      for(int j=P->row[i]; j<P->row[i+1]; j++){
        if(!_check_in_A(A, i, P->col[j])){
          printf("AP_ERROR %d %d\n", P->col[j], i);
        }
      }
    }
    CSRm::free(A);
    CSRm::free(P);
  }

  CSRm::free(h_Alocal);

}


CSR* broadcast_FullMatrix(CSR *A){
  _MPI_ENV;

  if(ISMASTER)
    assert(!A->on_the_device);

  itype n, m, nnz;
  if(ISMASTER){
    n = A->n;
    m = A->m;
    nnz = A->nnz;
  }

  CHECK_MPI(
    MPI_Bcast(&n, sizeof(itype), MPI_BYTE, 0, MPI_COMM_WORLD)
  );

  CHECK_MPI(
    MPI_Bcast(&m, sizeof(itype), MPI_BYTE, 0, MPI_COMM_WORLD)
  );

  CHECK_MPI(
    MPI_Bcast(&nnz, sizeof(itype), MPI_BYTE, 0, MPI_COMM_WORLD)
  );

  if(!ISMASTER){
      A = CSRm::init(n, m, nnz, true, false, false, n, 0);
  }

  CHECK_MPI(
    MPI_Bcast(A->row, sizeof(itype)*(A->n+1), MPI_BYTE, 0, MPI_COMM_WORLD)
  );

  CHECK_MPI(
    MPI_Bcast(A->col, sizeof(itype)*A->nnz, MPI_BYTE, 0, MPI_COMM_WORLD)
  );

  CHECK_MPI(
    MPI_Bcast(A->val, sizeof(vtype)*A->nnz, MPI_BYTE, 0, MPI_COMM_WORLD)
  );

  CSR *d_A = CSRm::copyToDevice(A);
  CSRm::free(A);

  return d_A;
}


CSR* join_MatrixMPI_all(CSR *Alocal){
  _MPI_ENV;

  assert(nprocs > 1);
  assert(!Alocal->on_the_device);

  itype row_ns[nprocs];

  //send rows sizes
  CHECK_MPI(
    MPI_Allgather(
      &Alocal->n,
      sizeof(itype),
      MPI_BYTE,
      &row_ns,
      sizeof(itype),
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  itype nnzs[nprocs];

  //send nnz sizes
  CHECK_MPI(
    MPI_Allgather(
      &Alocal->nnz,
      sizeof(itype),
      MPI_BYTE,
      &nnzs,
      sizeof(itype),
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  itype full_n = 0;
  itype full_nnz = 0;
  CSR *A;
  int chunkn[nprocs], chunks[nprocs];

  for(int i=0; i<nprocs; i++){
    full_n += row_ns[i];
    full_nnz += nnzs[i];
  }

  assert(full_n == Alocal->full_n);

  A = CSRm::init(full_n, Alocal->m, full_nnz, true, false, false, full_n, 0);

  // gather rows
  for(int i=0; i<nprocs; i++){
    chunkn[i] = row_ns[i] * sizeof(itype);
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  chunkn[nprocs-1] += sizeof(itype);

  itype rn = Alocal->n * sizeof(itype);
  if(myid == nprocs-1)
    rn += 1; // +1 for the last process

  CHECK_MPI(
    MPI_Allgatherv(
      Alocal->row,
      rn,
      MPI_BYTE,
      A->row,
      chunkn,
      chunks,
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  itype rowoffset = 0;
  itype th = row_ns[0];
  int j=0;
  for(int i=0; i<Alocal->full_n; i++){
      // next piece
      if(i >= th && (j<(nprocs)) ){
        rowoffset += nnzs[j];
        j++;
        th += row_ns[j];
      }
      A->row[i] += rowoffset;
    }

    A->row[A->full_n] = nnzs[0];
    for(int i=1; i<nprocs; i++) {
      A->row[A->full_n] += nnzs[i];
    }

  // gather columns
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnzs[i] * sizeof(itype);
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  CHECK_MPI(
    MPI_Allgatherv(
      Alocal->col,
      Alocal->nnz * sizeof(itype),
      MPI_BYTE,
      A->col,
      chunkn,
      chunks,
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  // gather value
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnzs[i] * sizeof(vtype);
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  CHECK_MPI(
    MPI_Allgatherv(
      Alocal->val,
      Alocal->nnz * sizeof(vtype),
      MPI_BYTE,
      A->val,
      chunkn,
      chunks,
      MPI_BYTE,
      MPI_COMM_WORLD
    )
  );

  return A;

}
