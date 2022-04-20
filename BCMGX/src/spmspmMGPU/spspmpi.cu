#include <unistd.h>
#include "spmspmMGPU/spspmpi.h"
#include "utility/cudamacro.h"
#define NUM_THR 1024
#define BITXBYTE 8
#define CSRSEG
#include "matrix/matrixIO.h"

__global__
void _getNNZ(itype n, const itype * __restrict__ to_get_form_prow, const itype *__restrict__ row, itype *nnz_to_get_form_prow){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= n){ return; }
  itype j = to_get_form_prow[i];
  nnz_to_get_form_prow[i] = row[j+1] - row[j];
}

__forceinline__
__device__
int binsearch(const itype array[], itype size, itype value) {
  itype low, high, medium;
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
void _fillPRow( itype n, itype rows2bereceived, const itype * __restrict__ whichproc, itype *p_nnz_map,
  itype mypfirstrow, itype myplastrow, itype nzz_pre_local, itype Plocalnnz, const itype * __restrict__ local_row, itype *row ){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= n){ return; }
#if !defined(CSRSEG)
  if(i >= mypfirstrow && i <= myplastrow+1){
    row[i] = local_row[i-mypfirstrow] + nzz_pre_local;
    return;
  }
#else
  if(i > mypfirstrow && i <= myplastrow){
    return;
  }
  if(i == mypfirstrow && i == (myplastrow+1)){
    row[i] = local_row[i-mypfirstrow] + nzz_pre_local;
    return;
  }
#endif

  itype iv = binsearch(whichproc, rows2bereceived, i);

  itype shift = Plocalnnz * (i>myplastrow);
  if(iv > 0)
    row[i] = p_nnz_map[iv-1] * (iv > 0) + shift;
  else
    row[i] = shift;
}

__global__
void _fillPRowNoComm( itype n, itype mypfirstrow, itype myplastrow, itype Plocalnnz, const itype * __restrict__ local_row, itype *row){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= n){ return; }
  if(i >= mypfirstrow && i <= (myplastrow+1)){
    row[i] = local_row[i-mypfirstrow];
    return;
  }
  row[i] = Plocalnnz * (i>myplastrow);
}


__global__
void _getColMissingMap( itype nnz, itype mypfirstrow, itype myplastrow, itype *col, int *mask){
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= nnz){ return; }
    itype c = col[tid];

    int mask_idx = c / (sizeof(itype)*BITXBYTE);
    unsigned int m = 1 << ( (c%(sizeof(itype)*BITXBYTE)) );

    if(c < mypfirstrow || c > myplastrow){
       atomicOr(&mask[mask_idx], m);
    }
}

__global__
void _getColVal( itype n, itype *rcvprow, itype *nnz_per_row, const itype * __restrict__ row,
    const itype *__restrict__ col, const vtype * __restrict__ val, itype *col2get, vtype *val2get, itype row_shift ){
    itype q = blockDim.x * blockIdx.x + threadIdx.x;
    if(q >= n){ return; }
    itype I = rcvprow[q] - row_shift;
    itype start = row[I];
    itype end = row[I+1];
    for(itype i=start, j=nnz_per_row[q]; i<end; i++, j++){
        col2get[j] = col[i];
        val2get[j] = val[i];
    }
}

itype merge(itype a[], itype b[], itype c[], itype n1, itype n2) {
    itype i, j, k;

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = 0; // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (a[i] < b[j]) {
            c[k] = a[i];
            i++;
        } else if(a[i] > b[j]) {
            c[k] = b[j];
            j++;
        } else {
            c[k] = b[j];
            i++;
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there are any */
    while (i < n1) {
        c[k] = a[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there are any */
    while (j < n2) {
        c[k] = b[j];
        j++;
        k++;
    }
    return k;
}

int bswhichprocess(itype *P_n_per_process, int nprocs, itype e){
  unsigned int low, high, medium;
  low=0;
  high=nprocs;
  while(low<high) {
      medium=(high+low)/2;
      if(e > P_n_per_process[medium]) {
        low=medium+1;
      } else {
        high=medium;
      }
  }
  return low;
}

CSR* nsparseMGPU(CSR *Alocal, CSR *Pfull, csrlocinfo *Plocal) {
  _MPI_ENV;
  //static int cnt=0;
  sfCSR mat_a, mat_p, mat_c;
  assert(Alocal->on_the_device && Pfull->on_the_device);

  mat_a.M = Alocal->n;
  mat_a.N = Alocal->m;
  mat_a.nnz = Alocal->nnz;

  mat_a.d_rpt = Alocal->row;
  mat_a.d_col = Alocal->col;
  mat_a.d_val = Alocal->val;

  mat_p.M = Pfull->n;
  mat_p.N = Pfull->m;
  mat_p.nnz = Pfull->nnz;

  mat_p.d_rpt = Pfull->row;
  mat_p.d_col = Pfull->col;
  mat_p.d_val = Pfull->val;

  spgemm_csrseg_kernel_hash(&mat_a, &mat_p, &mat_c, Plocal);
	       
  mat_c.M=mat_a.M;
  mat_c.N=mat_p.N;

  //cnt++;
  CSR* C = CSRm::init(mat_c.M, mat_c.N, mat_c.nnz, false, true, false, Alocal->full_n, Alocal->row_shift);
  C->row = mat_c.d_rpt;
  C->col = mat_c.d_col;
  C->val = mat_c.d_val;

  return C;
}




vector<int> *get_missing_col( CSR *Alocal, CSR *Plocal ){
  _MPI_ENV;
  itype Pn, mypfirstrow, myplastrow;
  gridblock gb;
  
  if ( Plocal != NULL ){
    Pn = Plocal->full_n;
    mypfirstrow = Plocal->row_shift;
    myplastrow  = Plocal->n + Plocal->row_shift-1;
  }else{
    Pn = Alocal->full_n;
    mypfirstrow = Alocal->row_shift;
    myplastrow  = Alocal->n + Alocal->row_shift-1;
  }// P_n_per_process[i]: number of rows that process i have of matrix P 

  itype size_mask = (Pn+(((sizeof(int)*BITXBYTE))-1))/(sizeof(int)*BITXBYTE);
  vector<int> *dev_bitcol = Vector::init<int>(size_mask, true, true);
  Vector::fillWithValue(dev_bitcol, 0);

  if(Alocal->nnz){
  	gb = gb1d(Alocal->nnz, NUM_THR);
  	_getColMissingMap<<<gb.g, gb.b>>>( Alocal->nnz, mypfirstrow, myplastrow, Alocal->col, dev_bitcol->val );
  }else{
    printf("\n%d local->nnz == 0\n\n", myid);
  }
  vector<int> *_bitcol = Vector::copyToHost(dev_bitcol);

  Vector::free(dev_bitcol);
  return _bitcol;
}






void compute_rows_to_rcv_CPU( CSR *Alocal, CSR *Plocal, vector<int> *_bitcol){
  _MPI_ENV;

  if(nprocs == 1){ return ; }

  Alocal->rows_to_get = (rows_to_get_info *) Malloc( sizeof(rows_to_get_info) );

  itype Pn, mypfirstrow, myplastrow;
  itype *P_n_per_process;
  P_n_per_process = (itype*) Malloc(sizeof(itype)*nprocs);

  // send rows numbers to each process, Plocal->n local number of rows 
  if ( Plocal != NULL ){
    Pn = Plocal->full_n;
    mypfirstrow = Plocal->row_shift;
    myplastrow  = Plocal->n + Plocal->row_shift-1;
    CHECK_MPI( MPI_Allgather( &Plocal->n, sizeof(itype), MPI_BYTE, P_n_per_process, sizeof(itype), MPI_BYTE, MPI_COMM_WORLD ) );
  }else{
    Pn = Alocal->full_n;
    mypfirstrow = Alocal->row_shift;
    myplastrow  = Alocal->n + Alocal->row_shift-1;
    CHECK_MPI( MPI_Allgather( &Alocal->n, sizeof(itype), MPI_BYTE, P_n_per_process, sizeof(itype), MPI_BYTE, MPI_COMM_WORLD ) );
  }// P_n_per_process[i]: number of rows that process i owns of matrix P 

  itype *whichprow=NULL, *rcvpcolxrow=NULL, *rcvprow=NULL;

  int *displr, *displs, *scounts, *rcounts2, *scounts2, *displr2, *displs2;
  int rcounts[nprocs];
  unsigned int *rcvcntp;
  displr = (int*) Malloc(sizeof(int)*nprocs);
  rcounts2 = (int*) Malloc(sizeof(int)*nprocs);
  scounts2 = (int*) Malloc(sizeof(int)*nprocs);
  displs2 = (int*) Malloc(sizeof(int)*nprocs);
  displr2 = (int*) Malloc(sizeof(int)*nprocs);
  displs = (int*) Malloc(sizeof(int)*nprocs);
  scounts = (int*) Malloc(sizeof(int)*nprocs);
  rcvcntp = (unsigned int*) Malloc(sizeof(unsigned int)*nprocs);

  unsigned int countp[nprocs], offset[nprocs];
  unsigned int sothercol=0;
  int cntothercol=0;
  int whichproc;
  itype *othercol[1]={NULL};
  itype *bitcol[1];

  unsigned int i, j;
  itype size_mask = (Pn+(((sizeof(int)*BITXBYTE))-1))/(sizeof(int)*BITXBYTE);

  bitcol[0] = _bitcol->val;

  cntothercol = 0; // number of rows that we need from other processes
  for(i=0; i<size_mask; i++){
	int cc = __builtin_popcount(bitcol[0][i]);
	cntothercol += cc;
  }

  if(cntothercol>0) {
    othercol[0]=(itype *)Malloc(cntothercol*sizeof(itype));
    itype scratch, col;
    int k;
    for(i=0, j=0; i<size_mask; i++){
        if(bitcol[sothercol][i]) {
           scratch=bitcol[sothercol][i];	
	   while(scratch) {
	   	k=__builtin_ffs(scratch);
		col=(i*(sizeof(itype)*BITXBYTE))+(k-1);
		if(col<mypfirstrow||col>myplastrow) {
	   		othercol[sothercol][j]=col;
			j++;
		}
		scratch &= ~(1UL << (k-1));
	   }
	   if(j==cntothercol) { break; }
	}
    }
  } // othercol: contains the rows number we need to retrieve from other processes

  // the last list is in othercol[sothercol]
  for(i=0; i<nprocs; i++){
    countp[i]=0;
  }

  itype *aofwhichproc=(itype *)Malloc(sizeof(itype)*cntothercol); 

  itype cum_p_n_per_process[nprocs];
  cum_p_n_per_process[0]=P_n_per_process[0]-1;
  for(int i=1; i<nprocs; i++){
     cum_p_n_per_process[i]=cum_p_n_per_process[i-1] + P_n_per_process[i];
  }

  itype countall=0;
  
  for(j=0; j<cntothercol; j++) {
    whichproc = bswhichprocess(cum_p_n_per_process, nprocs, othercol[sothercol][j]);
    if(whichproc > (nprocs-1)){
      whichproc=nprocs-1;
    }
    countp[whichproc]++;
    aofwhichproc[countall]=whichproc;
    countall++;
  }
  offset[0]=0;
  for(i=1; i<nprocs; i++) {
    offset[i]=offset[i-1]+countp[i-1];
    countp[i-1]=0;
  }
  countp[nprocs-1]=0;
  if(countall>0) {
     whichprow=(itype *)Malloc(sizeof(itype)*countall); 
     rcvpcolxrow=(itype *)Malloc(sizeof(itype)*countall);
  }

  Alocal->rows_to_get->rows2bereceived=countall;

  for(j=0; j<cntothercol; j++) {
      whichproc=aofwhichproc[j];
      whichprow[offset[whichproc]+countp[whichproc]]=othercol[sothercol][j];
      countp[whichproc]++;
  }
  free(aofwhichproc);

  if(countp[myid]!=0) {
     fprintf(stderr,"self countp should be zero! %d\n",myid);
     exit(1);
  }
  free(othercol[sothercol]);

  if(MPI_Alltoall(countp,sizeof(itype),MPI_BYTE,rcvcntp,sizeof(itype),MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoall of P rows\n");
     exit(1);
  } 
  if(rcvcntp[myid]!=0) {
     fprintf(stderr,"self rcvcntp should be zero! %d\n",myid);
     exit(1);
  }
  countall=0;
  for(i=0; i<nprocs; i++) {
    rcounts2[i]=scounts[i]=countp[i]*sizeof(itype);
    displr2[i] =displs[i]=((i==0)?0:(displs[i-1]+scounts[i-1]));

    scounts2[i]=rcounts[i]=rcvcntp[i]*sizeof(itype);
    displs2[i] =displr[i]=((i==0)?0:(displr[i-1]+rcounts[i-1]));
    countall+=rcvcntp[i];
  }

  if(countall>0) {
     rcvprow=(itype *)Malloc(sizeof(itype)*countall);
  }

  if( MPI_Alltoallv(whichprow,scounts,displs,MPI_BYTE,rcvprow,rcounts,displr,MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoallv of whichprow rows\n");
     exit(1);
  }

  memset(scounts, 0, nprocs*sizeof(int));
  memset(displs, 0, nprocs*sizeof(int));
  vector<itype> *nnz_per_row_shift = NULL;
  itype total_row_to_rec = countall;
  countall = 0;
  if(total_row_to_rec){
    itype q = 0;
    itype tot_shift = 0;
    nnz_per_row_shift = Vector::init<itype>(total_row_to_rec, true, false); // no temp buff is used on the HOST only;
 
    q = 0;
    for(i=0; i<nprocs; i++){
      displs[i] = (i == 0) ? 0 : (displs[i-1]+scounts[i-1]);
      if(i == myid){
        continue;
      }
      for(j=0; j<rcvcntp[i]; j++) {
        scounts[i] += 1;
        nnz_per_row_shift->val[q] = tot_shift;
        tot_shift += 1;
        q++;
      }
      countall+=scounts[i];
      scounts[i]*=sizeof(itype);
      displs[i]=((i==0)?0:(displs[i-1]+scounts[i-1]));
    }
  }

  Alocal->rows_to_get->total_row_to_rec = total_row_to_rec;
  Alocal->rows_to_get->nnz_per_row_shift = nnz_per_row_shift;
  Alocal->rows_to_get->countall = countall;
  Alocal->rows_to_get->rcvprow = rcvprow;
  Alocal->rows_to_get->whichprow = whichprow;
  Alocal->rows_to_get->rcvpcolxrow = rcvpcolxrow;
  Alocal->rows_to_get->displr = displr;
  Alocal->rows_to_get->displs = displs;
  Alocal->rows_to_get->scounts = scounts;
  Alocal->rows_to_get->rcounts2 = rcounts2;
  Alocal->rows_to_get->scounts2 = scounts2;
  Alocal->rows_to_get->displs2 = displs2;
  Alocal->rows_to_get->displr2 = displr2;
  Alocal->rows_to_get->rcvcntp = rcvcntp;
  Alocal->rows_to_get->P_n_per_process = P_n_per_process;

  return ;
}


CSR* nsparseMGPU_noCommu(handles *h, CSR *Alocal, CSR *Plocal){
  
  _MPI_ENV;
  gridblock gb;
  itype mypfirstrow = Plocal->row_shift;
  itype myplastrow  = Plocal->n + Plocal->row_shift-1;
  
  vector<itype> *bareminp_row = Vector::init<itype>(Plocal->full_n+1, true, true);
  // no comunication; copy and fill on the right of Prow
  gb = gb1d(Plocal->full_n+1, NUM_THR);

  _fillPRowNoComm<<<gb.g, gb.b>>>( Plocal->full_n+1, mypfirstrow, myplastrow,
                                     Plocal->nnz, Plocal->row, bareminp_row->val );
  Plocal->row = bareminp_row->val;

  csrlocinfo Pinfo1p;
  Pinfo1p.fr=0;
  Pinfo1p.lr=Plocal->n;
  Pinfo1p.row=Plocal->row;
  Pinfo1p.col=NULL;
  Pinfo1p.val=Plocal->val;

  CSR *C = nsparseMGPU(Alocal, Plocal, &Pinfo1p);
  std::free(bareminp_row);
  return C;
}



CSR* nsparseMGPU_commu(handles *h, CSR *Alocal, CSR *Plocal){
  _MPI_ENV;

  csrlocinfo Pinfo1p;
  Pinfo1p.fr=0;
  Pinfo1p.lr=Plocal->n;
  Pinfo1p.row=Plocal->row;
#if !defined(CSRSEG)
  Pinfo1p.col=NULL;
#else
  Pinfo1p.col=Plocal->col;
#endif
  Pinfo1p.val=Plocal->val;

  if(nprocs == 1){
	return nsparseMGPU(Alocal, Plocal, &Pinfo1p );
  }
  static int cnt=0;
  itype *Pcol, *col2send=NULL;
  vtype *Pval;
  vtype *val2send=NULL;
  gridblock gb;
  unsigned int i, j, k;

  int *displr, *displs, *scounts, *rcounts2, *scounts2, *displr2, *displs2;
  int rcounts[nprocs];
  int rcounts_src[nprocs], displr_src[nprocs];
  int displr_target[nprocs];
  unsigned int *rcvcntp;
  itype *P_n_per_process, nnzp=0;
  CHECK_MPI(  MPI_Allreduce( &Plocal->nnz, &nnzp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ) );
  P_n_per_process = Alocal->rows_to_get->P_n_per_process;
  displr = Alocal->rows_to_get->displr;
  rcounts2 = Alocal->rows_to_get->rcounts2;
  scounts2 = Alocal->rows_to_get->scounts2;
  displs2 = Alocal->rows_to_get->displs2;
  displr2 = Alocal->rows_to_get->displr2;
  rcvcntp = Alocal->rows_to_get->rcvcntp;
  displs = Alocal->rows_to_get->displs;
  scounts = Alocal->rows_to_get->scounts;

  itype mycolp = Plocal->nnz;   // number of nnz stored by the process
  itype Pm = Plocal->m;         // number of columns in P  
  itype mypfirstrow = Plocal->row_shift;
  itype myplastrow  = Plocal->n + Plocal->row_shift-1;
  itype Pn = Plocal->full_n;

  vector<itype> *nnz_per_row_shift = NULL;
  nnz_per_row_shift = Alocal->rows_to_get->nnz_per_row_shift;
 
  itype *p2rcvprow;
  itype countall = 0;
  countall = Alocal->rows_to_get->countall;
  itype q = 0;

  memset(rcounts, 0, nprocs*sizeof(int)); 

  itype *whichprow=NULL, *rcvpcolxrow=NULL, *rcvprow=NULL;
  rcvprow = Alocal->rows_to_get->rcvprow;
  whichprow = Alocal->rows_to_get->whichprow;
  rcvpcolxrow = Alocal->rows_to_get->rcvpcolxrow;

  itype *dev_col2send = idevtemp1;
  vtype *dev_val2send = vdevtemp1;
  if(countall>0){
    col2send = iAtemp1;
    val2send = vAtemp1;
    itype *dev_rcvprow;

    // sync call to make async stream1 stream2 one event cp1 
    vector<itype> *dev_nnz_per_row_shift = NULL;
    if(nnz_per_row_shift->n>0) {
	    if(nnz_per_row_shift->n>8000000) {
		    fprintf(stderr,"Task %d, n=%d\n",myid,nnz_per_row_shift->n); 
		    exit(0);
	    }
    dev_nnz_per_row_shift = Vector::init<itype>(nnz_per_row_shift->n, false, true); 
    dev_nnz_per_row_shift->val = idevtemp2;

    CHECK_DEVICE(cudaMemcpyAsync(dev_nnz_per_row_shift->val, nnz_per_row_shift->val, dev_nnz_per_row_shift->n * sizeof(itype), cudaMemcpyHostToDevice,h->stream1 ));
    CHECK_DEVICE( cudaMalloc( (void**) &dev_rcvprow, dev_nnz_per_row_shift->n * sizeof(itype)) );
    CHECK_DEVICE( cudaMemcpyAsync(dev_rcvprow, rcvprow, dev_nnz_per_row_shift->n * sizeof(itype), cudaMemcpyHostToDevice, h->stream1));
    }
    q = 0;
    for(i=0; i<nprocs; i++){
      if(i==myid) { continue; }
      // shift
      p2rcvprow = &rcvprow[displr[i]/sizeof(itype)]; //recvprow will be modified
      for(j=0; j<rcvcntp[i]; j++){
        p2rcvprow[j] = 1; // recycle rcvprow to send the number of columns in each row
        q++;
      }
    }
    cnt++;
    if(nnz_per_row_shift->n>0) {
    gb = gb1d(dev_nnz_per_row_shift->n, NUM_THR);
    _getColVal<<<gb.g, gb.b, 0, h->stream1>>>( dev_nnz_per_row_shift->n, dev_rcvprow, dev_nnz_per_row_shift->val, Plocal->row, Plocal->col, Plocal->val, dev_col2send, dev_val2send, mypfirstrow );
    cudaStreamSynchronize( h->stream1 );
    }

    CHECK_DEVICE( cudaMemcpyAsync(col2send, dev_col2send, countall * sizeof(itype), cudaMemcpyDeviceToHost,h->stream1 ));
    CHECK_DEVICE( cudaMemcpyAsync(val2send, dev_val2send, countall * sizeof(vtype), cudaMemcpyDeviceToHost,h->stream2 ));

    if(nnz_per_row_shift->n>0) {
    dev_nnz_per_row_shift->val = NULL;
    std::free(dev_nnz_per_row_shift);
    }
  }

  if(MPI_Alltoallv(rcvprow,scounts2,displs2,MPI_BYTE,rcvpcolxrow,rcounts2,displr2,MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
    fprintf(stderr,"Error in MPI_Alltoallv of rcvprow\n");
    exit(1);
  }
  vector<itype> *bareminp_row = Vector::init<itype>(Plocal->full_n+1, true, true);
 
  itype nzz_pre_local = 0;

  itype rows2bereceived = Alocal->rows_to_get->rows2bereceived; 
  if(rows2bereceived){
    // we have rows from other process
    bool flag = true;
    itype *dev_whichproc = NULL;
    itype r = 0;

    vector<itype> *P_nnz_map = Vector::init<itype>(rows2bereceived, true, false);
    k = 0;
    int whichproc;

    itype cum_p_n_per_process[nprocs];
    cum_p_n_per_process[0]=P_n_per_process[0]-1;
    for(int j=1; j<nprocs; j++){
     cum_p_n_per_process[j]=cum_p_n_per_process[j-1] + P_n_per_process[j];
    }
    for(i=0; i<rows2bereceived; i++){r = whichprow[i];
      // count nnz per process for comunication
      whichproc = bswhichprocess(cum_p_n_per_process, nprocs, r);
      if(whichproc>(nprocs-1)){
        whichproc=nprocs-1;
      }
      rcounts[whichproc] += rcvpcolxrow[i];
  	  // after local add shift
      if(r > mypfirstrow && flag){
          nzz_pre_local = k;
          flag = false;
      }
      k += rcvpcolxrow[i];
      P_nnz_map->val[i] = k;
    }

    if(flag){
      nzz_pre_local = P_nnz_map->val[rows2bereceived-1];
    }
    CHECK_DEVICE(  cudaMalloc( (void**) &dev_whichproc, rows2bereceived * sizeof(itype) );  );
    CHECK_DEVICE(  cudaMemcpy(dev_whichproc, whichprow, rows2bereceived * sizeof(itype), cudaMemcpyHostToDevice); );

  	vector<itype> *dev_P_nnz_map = Vector::copyToDevice(P_nnz_map);

    gb = gb1d(Plocal->full_n+1, NUM_THR);

    _fillPRow<<<gb.g, gb.b>>>( Plocal->full_n+1, rows2bereceived, dev_whichproc, dev_P_nnz_map->val,
                               mypfirstrow, myplastrow, nzz_pre_local, Plocal->nnz, Plocal->row, bareminp_row->val );

    Vector::free(dev_P_nnz_map);
    Vector::free(P_nnz_map);
    cudaFree(dev_whichproc);
  }else{ // no comunication; copy and fill on the right of Prow
    gb = gb1d(Plocal->full_n+1, NUM_THR);

    _fillPRowNoComm<<<gb.g, gb.b>>>( Plocal->full_n+1, mypfirstrow, myplastrow,
                                     Plocal->nnz, Plocal->row, bareminp_row->val );
  }
  if(rcounts[myid]!=0) {
     fprintf(stderr,"task: %d, unexpected rcount[%d]=%d. It should be zero\n",myid,myid,rcounts[myid]);
     exit(1);
  }


  int totcell=0;
  static int s_totcell;
  memcpy(rcounts_src, rcounts, nprocs*sizeof(itype));
  for(i=0; i<nprocs; i++) {
      totcell += rcounts[i];
      displr_target[i]=(i==0)?0:(displr_target[i-1]+(i==(myid+1)?mycolp:rcounts_src[i-1]));
      rcounts[i]*=sizeof(itype);
      displr[i]=(i==0)?0:(displr[i-1]+rcounts[i-1]);
      displr_src[i]=displr[i]/sizeof(itype);
  }
  if (iPtemp1 == NULL && totcell > 0){ // first allocation
    MY_CUDA_CHECK( cudaMallocHost( &iPtemp1, totcell*sizeof(itype) ) );
    vPtemp1 = (vtype*) Malloc( totcell*sizeof(vtype) );
    s_totcell = totcell;
  }
  if (totcell > s_totcell){ // not enough space
    MY_CUDA_CHECK( cudaFreeHost(iPtemp1));
    printf("[Realloc] --- totcell: %d s_totcell: %d\n",totcell,s_totcell);
    MY_CUDA_CHECK( cudaMallocHost(&iPtemp1, sizeof (itype) * totcell ) );
    vPtemp1 = (vtype*) Realloc( vPtemp1, totcell * sizeof(vtype) );
    s_totcell = totcell; 
  }
  Pcol = iPtemp1;
  Pval = vPtemp1;
  if (countall>0){
      cudaStreamSynchronize (h->stream1);
      dev_col2send = NULL;
  }
  if(MPI_Alltoallv(col2send,scounts,displs,MPI_BYTE,Pcol,rcounts,displr,MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoallv of col2send\n");
     exit(1);
  }

  CSR *baremin_Plocal = NULL;
  baremin_Plocal = CSRm::init(Pn, Pm, nnzp, true, true, false, Pn, Alocal->row_shift);
  for(i=0; i<nprocs; i++){
	if(rcounts_src[i]>0) {
	   CHECK_DEVICE( cudaMemcpyAsync(baremin_Plocal->col+displr_target[i], Pcol+displr_src[i], rcounts_src[i] * sizeof(itype), cudaMemcpyHostToDevice, h->stream1)  );
	}
  }

  col2send = NULL;
  for(i=0; i<nprocs; i++) {
      scounts[i]*=(sizeof(vtype)/sizeof(itype));
      displs[i]*=(sizeof(vtype)/sizeof(itype));
      rcounts[i]*=(sizeof(vtype)/sizeof(itype));
      displr[i]*=(sizeof(vtype)/sizeof(itype));
  }
  if (countall > 0) {
      cudaStreamSynchronize (h->stream2);
      dev_val2send = NULL;
  }
  if(MPI_Alltoallv(val2send,scounts,displs,MPI_BYTE,Pval,rcounts,displr,MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoallv of val2send\n");
     exit(1);
  }
  val2send = NULL;

  for(i=0; i<nprocs; i++){
	if(rcounts_src[i]>0) {
		CHECK_DEVICE(  cudaMemcpy(baremin_Plocal->val+displr_target[i], Pval+displr_src[i], rcounts_src[i] * sizeof(vtype), cudaMemcpyHostToDevice)  );
	}
  }
  CHECK_DEVICE( cudaFree(baremin_Plocal->row) );
  baremin_Plocal->row = bareminp_row->val;

#if !defined(CSRSEG) 
  CHECK_DEVICE( cudaMemcpy(baremin_Plocal->val + nzz_pre_local, Plocal->val, Plocal->nnz * sizeof(vtype), cudaMemcpyDeviceToDevice);  );
  CHECK_DEVICE( cudaMemcpy(baremin_Plocal->col + nzz_pre_local, Plocal->col, Plocal->nnz * sizeof(itype), cudaMemcpyDeviceToDevice);  );
#endif

  csrlocinfo Plocalinfo;
  Plocalinfo.fr=mypfirstrow;
  Plocalinfo.lr=myplastrow;
  Plocalinfo.row=Plocal->row;
#if !defined(CSRSEG)
  Plocalinfo.col=NULL;
#else
  Plocalinfo.col=Plocal->col;
#endif
  Plocalinfo.val=Plocal->val;

  cudaDeviceSynchronize();
  CSR *C = nsparseMGPU(Alocal, baremin_Plocal, &Plocalinfo);

  Pcol = NULL;
  Pval = NULL; //memory will free up in AMG
  CSRm::free(baremin_Plocal);
  std::free(bareminp_row); 
  CSRm::free_rows_to_get(Alocal); 
  return C;
}

