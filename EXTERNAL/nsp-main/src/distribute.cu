#include "myMPI.h"
#include "cudamacro.h"
#include <cuda_runtime.h>
#include "CSR_T.cu"
#include <chrono>
#include "matrixIO.h"
#include "READ_MATRIX_MPI.cpp"

template <typename T>
vector<T>* aggregateVector(vector<T> *u_local, itype full_n, vector<T> *u=NULL){
  _MPI_ENV;

  vector<T> *h_u_local = Vector::copyToHost(u_local);

  vector<T> *h_u = Vector::init<T>(full_n, true, false);

  int row_ns[nprocs];
  int chunks[nprocs], chunkn[nprocs];

  for(int i=0; i<nprocs-1; i++)
    row_ns[i] = full_n / nprocs;
  row_ns[nprocs-1] = full_n - ( (full_n / nprocs) * (nprocs - 1) );

  for(int i=0; i<nprocs; i++){
    chunkn[i] = row_ns[i] * sizeof(T);
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }

  CHECK_MPI(
    MPI_Allgatherv(
      h_u_local->val,
      row_ns[myid] * sizeof(T),
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

  //int num_row_proc;

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

  //printf("\nSPLIT %d %d %d\n", myid, nstart, nrows[myid]);

  scalar<itype> *nnz = Scalar::init<itype>(0, true);

  gridblock gb = gb1d(rowsxproc, 1024);
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

  //cout << n << " " << m << endl;

  if( (nprocs>1) && myid==(nprocs-1) )
    // compute the number of rows for the last process
     rowsxproc = n - ( (n / nprocs) * (nprocs - 1) );
  else
    // compute the number of rows for the process
    rowsxproc = n / nprocs;

  //cout << rowsxproc << endl;

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
  chop_array_MPI<itype>(nprocs, n, chunks, chunkn);
  itype rows_shift = chunks[myid] / sizeof(itype);

  // printf("myid %d rows_shift %d\t%d %d %d %d\n",myid,rows_shift,chunks[0],chunks[1],chunks[2],chunks[3]);

  CSR *Alocal = CSRm::init(rowsxproc, m, mycol, true, false, false, n, rows_shift);

  printf("myid %d %d %d %d %d %d\n",myid,rowsxproc, m, mycol, n, rows_shift);

  // get row pointers
  //vector<itype> *myrows = Vector::init<itype>(rowsxproc+1, true, false);
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

  //vector<itype> *mycols = Vector::init<itype>(mycol, true, false);
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
  //vector<vtype> *myvals = Vector::init<vtype>(mycol, true, false);
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

  //for(int i=0; i<nprocs; i++)
  //  printf("\n%d %d nnzs, %d\n", i, myid, nnzs[i]);

/*
  // recover shift
  if(myid > 0){
    int shift = 0;
    for(int j=0; j<myid; j++)
      shift += nnzs[j];
    for(int i=0; i<=Alocal->n+1; i++)
      Alocal->row[i] += shift;
  }
*/

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
    chunkn[nprocs-1] += sizeof(itype);
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
    itype rowoffset=0;
    int j=0;
    for(int i=row_ns[0]; i<Alocal->full_n; i++) {
        if(( i % row_ns[j])==0 && (j<(nprocs-1))){
          rowoffset += nnzs[j];
          j++;
        }
      A->row[i]+=rowoffset;
      }
      A->row[A->full_n] = nnzs[0];
      for(int i=1; i<nprocs; i++) {
        A->row[A->full_n] += nnzs[i];
      }

  }

  // gather columns
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnzs[i] /* * sizeof(itype) */;
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  CHECK_MPI(
    MPI_Gatherv(
      Alocal->col,
      Alocal->nnz /* * sizeof(itype) */,
      MPI_INT,
      myid ? NULL : A->col,
      chunkn,
      chunks,
      MPI_INT,
      0,
      MPI_COMM_WORLD
    )
  );

  // gather value
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnzs[i] /*  * sizeof(vtype) */;
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  CHECK_MPI(
    MPI_Gatherv(
      Alocal->val,
      Alocal->nnz /* * sizeof(vtype) */,
      MPI_DOUBLE,
      myid ? NULL : A->val,
      chunkn,
      chunks,
      MPI_DOUBLE,
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
       //cudaGetDevice(&dev);
       //cudaGetDeviceProperties(&deviceProp, dev);
       //size_t free_bytes, total_bytes;
       //cudaMemGetInfo(&free_bytes, &total_bytes);
       //       printf("Host: %s Rank=%d Device= %d (%s)  ECC=%s  Free = %lu, Total = %lu\n",host_name,rank, devloc[myrank],deviceProp.name, deviceProp.ECCEnabled ? "Enabled " : "Disabled", (unsigned long)free_bytes, (unsigned long)total_bytes);
       //
       snprintf(gpuname,sizeof(gpuname),"%d",devloc[myrank]);
       if(setenv("CUDA_VISIBLE_DEVICES",gpuname,1)<0) {
           fprintf(stderr,"Could not set CUDA_VISIBLE_DEVICES env variable\n");
           exit(1);
       }
       return;
}
#else
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

//--------------------------
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
      //double v = A->val[j];
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


//------------------_______________________------------------------________________


CSR* broadcast_FullMatrix(CSR *A){
  _MPI_ENV;

/*
  CSR *A;
  if(ISMASTER){
    assert(_A->on_the_device);
    A = CSRm::copyToHost(_A);
    CSRm::free(_A);
  }
  */

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

//---------------------------------------------------------------------------------------

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

  /* reset the row number */
  itype rowoffset=0;
  int j=0;
  for(int i=row_ns[0]; i<Alocal->full_n; i++) {
      if(( i % row_ns[j])==0 && (j<(nprocs-1))){
        rowoffset += nnzs[j];
        j++;
      }
    A->row[i]+=rowoffset;
    }
    A->row[A->full_n] = nnzs[0];
    for(int i=1; i<nprocs; i++) {
      A->row[A->full_n] += nnzs[i];
    }

  // gather columns
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnzs[i] /* * sizeof(itype) */;
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  CHECK_MPI(
    MPI_Allgatherv(
      Alocal->col,
      Alocal->nnz,
      MPI_INT,
      A->col,
      chunkn,
      chunks,
      MPI_INT,
      MPI_COMM_WORLD
    )
  );

  // gather value
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnzs[i] /* * sizeof(vtype) */;
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  CHECK_MPI(
    MPI_Allgatherv(
      Alocal->val,
      Alocal->nnz /* * sizeof(vtype) */,
      MPI_DOUBLE,
      A->val,
      chunkn,
      chunks,
      MPI_DOUBLE,
      MPI_COMM_WORLD
    )
  );

  return A;

}

template <typename T, typename T_d, typename Tbuf, typename T_sc, typename T_rc>
void MMPI_Scatterv(Tbuf* send_buff, T_sc *send_count, T_d *send_displ, MPI_Datatype send_datatype, 
                   Tbuf* recv_buff, T_rc recv_count, MPI_Datatype recv_datatype) {
  _MPI_ENV;

  if(ISMASTER){
    for (int i = 1; i < nprocs; i++) {
      MPI_Send(send_buff + send_displ[i], send_count[i] * sizeof(T), MPI_BYTE, i, 0, MPI_COMM_WORLD);
    }
    memcpy(recv_buff, send_buff, recv_count*sizeof(T));
  }
  if (myid > 0) MPI_Recv(recv_buff, recv_count*sizeof(T), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}

// CSR* split_MatrixMPI(CSR *A, SHIFT *shift){
//   _MPI_ENV;
  
//   gstype n, m;
//   if(ISMASTER){
//     n = A->n;
//     m = A->m;
//   }

//   CHECK_MPI( MPI_Bcast(&n, sizeof(n), MPI_BYTE, 0, MPI_COMM_WORLD) );
//   CHECK_MPI( MPI_Bcast(&m, sizeof(m), MPI_BYTE, 0, MPI_COMM_WORLD) );
  
//   auto myrow    = *(shift->rowxproc  ); myrow    = 0;
//   auto mycol    = *(shift->colxproc  ); mycol    = 0;
//   auto mycol_sh = *(shift->cols_shift); mycol_sh = 0;  // gstype

//   CHECK_MPI(MPI_Scatter(shift->rowxproc  ,sizeof(shift->rowxproc  [0]),MPI_BYTE, &myrow   ,sizeof(shift->rowxproc  [0]),MPI_BYTE,0,MPI_COMM_WORLD));
//   CHECK_MPI(MPI_Scatter(shift->colxproc  ,sizeof(shift->colxproc  [0]),MPI_BYTE, &mycol   ,sizeof(shift->colxproc  [0]),MPI_BYTE,0,MPI_COMM_WORLD));
//   CHECK_MPI(MPI_Scatter(shift->cols_shift,sizeof(shift->cols_shift[0]),MPI_BYTE, &mycol_sh,sizeof(shift->cols_shift[0]),MPI_BYTE,0,MPI_COMM_WORLD));

//   CSR *Alocal = CSRm::init((stype)myrow, (gstype)m, (stype)mycol, true, false, false, n, (gstype)mycol_sh);

//   MMPI_Scatterv<itype>(myid ? NULL : A->row,shift->rowxproc,shift->displ_row,MPI_BYTE,Alocal->row,myrow,MPI_BYTE);
//   MMPI_Scatterv<itype>(myid ? NULL : A->col,shift->colxproc,shift->displ_col,MPI_BYTE,Alocal->col,mycol,MPI_BYTE);
//   MMPI_Scatterv<vtype>(myid ? NULL : A->val,shift->colxproc,shift->displ_col,MPI_BYTE,Alocal->val,mycol,MPI_BYTE);

//   Alocal->row[myrow] = Alocal->row[0] + mycol;

//   CSR *d_Alocal = CSRm::copyToDevice(Alocal);
//   CSRm::free(Alocal);

//   return d_Alocal;
// }

template <typename T, typename T_d, typename Tbuf, typename T_sc, typename T_rc>
void MMPI_Gatherv(Tbuf* send_buff, T_sc send_count, MPI_Datatype send_datatype, 
                  Tbuf* recv_buff, T_rc *recv_count, T_d *displ,
                  MPI_Datatype recv_datatype, int root, MPI_Comm comm) {
  _MPI_ENV;
  // pay attention to sizeof(T)
  MPI_Status status;
  if ( myid != root ){
    MPI_Send(send_buff, send_count, send_datatype, 0, 0, comm);
  }
  if(myid == root) {
    memcpy(recv_buff, send_buff, send_count * sizeof(T));
    for (int i = 0; i < nprocs; i++) {
      if (i != root) 
        CHECK_MPI(MPI_Recv(recv_buff + displ[i], recv_count[i], recv_datatype, i, 0, comm, &status));
    }
  }
}

template <typename Tst, typename Tgs, typename Ti>
CSRtype<Tst,Tgs,Ti>* join_MatrixMPI(CSR *Alocal){
  _MPI_ENV;
  
  assert(nprocs > 1);
  assert(!Alocal->on_the_device);

  Tst row_s[nprocs];
  Tst nnz_s[nprocs];

  CHECK_MPI(MPI_Allgather(&Alocal->n  ,sizeof(Tst),MPI_BYTE,&row_s,sizeof(Tst),MPI_BYTE,MPI_COMM_WORLD)); //send rows sizes
  CHECK_MPI(MPI_Allgather(&Alocal->nnz,sizeof(Tst),MPI_BYTE,&nnz_s,sizeof(Tst),MPI_BYTE,MPI_COMM_WORLD)); //send nnz  sizes

  Tgs full_n   = 0; // Tgs
  Tgs full_nnz = 0; // Tgs
  CSRtype<Tst, Tgs, Ti> *A;
  // CSR *A;
  int chunkn   [nprocs];
  int chunks   [nprocs]; // Tgs
  Tgs displ_row[nprocs]; // Tgs
  Tgs displ_col[nprocs]; // Tgs

  if(ISMASTER){
    for(int i=0; i<nprocs; i++){
      full_n   += (Tgs)row_s[i];
      full_nnz += (Tgs)nnz_s[i];
    }
    assert(full_n == Alocal->full_n);

    A = CSRm::init_CSR<Tst,Tgs,Ti>(full_n, Alocal->m, full_nnz, true, false, false, full_n, 0);

    // gather rows
    for(int i=0; i<nprocs; i++){
      chunkn[i]    = row_s[i] * sizeof(itype);
      chunks[i]    = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
      displ_row[i] = ((i==0)?0:(displ_row[i-1]+row_s[i-1]));
      displ_col[i] = ((i==0)?0:(displ_col[i-1]+nnz_s[i-1])); 
    }
    chunkn[nprocs-1] += sizeof(itype);
  }

  itype rn = Alocal->n * sizeof(itype);
  if(myid == nprocs-1)
    rn += 1; // +1 for the last process

  MMPI_Gatherv<itype>     (Alocal->row,Alocal->n,MPI_INT,myid ? NULL : A->row,row_s ,displ_row,MPI_INT,0,MPI_COMM_WORLD);
  // CHECK_MPI(MPI_Gatherv(Alocal->row,rn,MPI_BYTE,myid ? NULL : A->row,chunkn,chunks,   MPI_BYTE,0,MPI_COMM_WORLD));

  // printf("MMPI_Gatherv success! myid = %d\n",myid);

  if(ISMASTER){
    /* reset the row number */
    Tgs rowoffset=0;
    int j=0;
    for(Tgs i=row_s[0]; i<Alocal->full_n; i++) {
        if(( i % row_s[j])==0 && (j<(nprocs-1))){
          rowoffset += nnz_s[j];
          j++;
        }
      A->row[i]+=rowoffset;
      }
      A->row[A->full_n] = nnz_s[0];
      for(int i=1; i<nprocs; i++) {
        A->row[A->full_n] += nnz_s[i];
      }

  }

  // gather columns
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnz_s[i] /* * sizeof(itype) */;
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  // CHECK_MPI(MPI_Gatherv(Alocal->col,Alocal->nnz,MPI_INT ,myid ? NULL : A->col,chunkn,chunks,MPI_INT ,0,MPI_COMM_WORLD));
  MMPI_Gatherv<itype>  (Alocal->col,Alocal->nnz,MPI_INT,myid ? NULL : A->col,nnz_s,displ_col,MPI_INT,0,MPI_COMM_WORLD);

  // gather value
  for(int i=0; i<nprocs; i++){
    chunkn[i] = nnz_s[i] /*  * sizeof(vtype) */;
    chunks[i] = ((i==0)?0:(chunks[i-1]+chunkn[i-1]));
  }
  // CHECK_MPI(MPI_Gatherv(Alocal->val,Alocal->nnz,MPI_DOUBLE,myid ? NULL : A->val,chunkn,chunks,MPI_DOUBLE,0,MPI_COMM_WORLD));

  MMPI_Gatherv<vtype>(Alocal->val,Alocal->nnz,MPI_DOUBLE,myid ? NULL : A->val,nnz_s,displ_col,MPI_DOUBLE,0,MPI_COMM_WORLD);

  return A;

}





/***********************************************************************
                        LOAD MATRIX
************************************************************************/
template <typename T>
T* resize_array(T *arr, long int curr_size, long int new_size) {
    T *new_arr = (T*) malloc(new_size * sizeof(T));

    for (long int i = 0; i < curr_size && i < new_size; i++) {
        new_arr[i] = arr[i];
    }

    free(arr);

    return new_arr;
}


template <typename T>
void irow2iat(T nr,long int nt,T *irow, T *iat, long int shift=0)
{
  T k;
  T j,irow_old,irow_new;
  irow_old = 0;
  for (k=0;k<nt;k++) {
    irow_new = irow[k];
    if (irow_new > irow_old) {
      for (j=irow_old;j<irow_new;j++)
        iat[j] = k; // iat[j-shift+1] = k;
      irow_old = irow_new;
    }
  }
  // optional:
  k = nt;
  for (j=irow_old;j<nr+1;j++)
    iat[j] = k;  // iat[j-shift+1] = k;
}


template <typename T>
void apply_shift(T *arr, long int nt, T shift=0){
  for (long int i = 0; i < nt; i++){
    arr[i] -= shift;
  }
}

template <typename Ti,typename Tie>
void irow2iat2(Tie nr,Tie nt,Ti *irow, Ti *iat, long int shift=0){
  Tie ii=0;
  Ti row_old = irow[0];
  iat[0] = 0;
  for (Tie i = 0; i < nt; ){
    Ti row = irow[i];
    // fill empty rows
    while(row_old < row){
      iat[row_old-shift+1] = ii;
      row_old++;
    }
    // cumulative sum
    while(irow[i] == row){
      ii++;
      i++;
    }
    iat[row-shift+1] = ii;
    row_old = row;
  }
}

template <typename T>
void swap(T *xcol, T *ycol, vtype *xval, vtype *yval) { 
  T temp = *xcol; 
  vtype tempf = *xval;
  *xcol = *ycol; 
  *xval = *yval;
  *ycol = temp;
  *yval = tempf; 
}

template <typename T>
void bubbleSort(T arr[], vtype val[], T n) { 
  T i, j; 
  for (i = 0; i < n-1; i++){  
    // Last i elements are already in place 
    for (j = 0; j < n-i-1; j++) {
      if (arr[j] > arr[j+1]){
        swap(&arr[j], &arr[j+1], &val[j], &val[j+1]); 
      }
    }
  }
}

template <typename T>
void check_and_fix_order(T *Arow, T *Acol, vtype *Aval, T n) {
  T prev;
  T wrongo;
  for (T i = 0; i < n; i++){
    wrongo = 0;
    prev = Acol[Arow[i]];
    for (T j = Arow[i] + 1; j < Arow[i + 1]; j++){
      if (Acol[j] < prev){
        wrongo = 1;
        break;
      }else{
        prev = Acol[j];
      }
    }
    if (wrongo){
      bubbleSort(&Acol[Arow[i]], &Aval[Arow[i]], (Arow[i + 1] - Arow[i]));
    }
  }
}

template <typename Ti = int, typename Tv = double, typename Tie = long int>
CSR* load_MatrixMPI(const char *matrix_path, bool ascii = 0, bool header_type = 1, bool partition = 0){
  _MPI_ENV;

  Ti *irow,*ja;
  double *coef;
  Tie nr,nc,nt,ntrank;
  long int col_shift = 0;
  long int row_shift = 0;
  std::chrono::duration<double> elapsed_seconds;

  MPI_Barrier(MPI_COMM_WORLD);
  auto rstart = std::chrono::system_clock::now();
  if (ascii){
    if (READ_ASCII_MATRIX_MPI<Ti,Tie>(matrix_path, &nr, &nc, &nt, ntrank, &irow, &ja, &coef, myid, nprocs) != 0)
      printf("error in READ_ASCII_MATRIX_MPI, iExt\n");  
  }else{
    if (READ_BINARY_MATRIX_MPI<Ti,Tv,Tie>(matrix_path, &nr, &nc, &nt, ntrank, &irow, &ja, &coef, myid, nprocs, header_type, partition) != 0)
      printf("error in READ_BINARY_MATRIX_MPI_test, int\n");
  }
  auto rend  = std::chrono::system_clock::now();
  elapsed_seconds = rend - rstart;
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) printf("Time for reading [s]: %f\n",elapsed_seconds.count());

  long int rowxproc = irow[ntrank-1]-irow[0]+1;
  CHECK_MPI(MPI_Exscan(&rowxproc, &col_shift, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD));
  CHECK_MPI(MPI_Exscan(&ntrank  , &row_shift, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD));

  Ti *iat  = (Ti*) malloc((rowxproc+1) * sizeof(Ti));
  if (iat==NULL) printf("ERROR malloc iat: rowxproc=%ld, sizeof Ti=%ld, ntrank=%d\n",rowxproc,sizeof(Ti),ntrank);

  irow2iat2<Ti,Tie>((Ti)rowxproc,ntrank,irow,iat,irow[0]); free(irow);
  check_and_fix_order<Ti>  (iat, ja, coef, rowxproc);
  apply_shift<Ti>(ja,ntrank,col_shift);
  
  CSR *Alocal = CSRm::init((stype)rowxproc, (gstype)nc, (stype)ntrank, true, false, false, nr, (long int)col_shift);
  free(Alocal->val);
  Alocal->val = coef;
  for(unsigned long j=0; j<= rowxproc; j++) {
      Alocal->row[j] = (itype)iat[j];
  }
  for(unsigned long k=0; k<ntrank; k++) {
      Alocal->col[k] = (itype)ja[k];
  }
  free( iat);
  free(  ja);

  Alocal->row[rowxproc] = Alocal->row[0] + ntrank;

  CSR *d_Alocal = CSRm::copyToDevice(Alocal);
  CSRm::free(Alocal);

  return d_Alocal;
}
