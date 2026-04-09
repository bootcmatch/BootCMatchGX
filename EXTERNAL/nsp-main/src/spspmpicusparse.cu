#include "spspmpi.h"

#define SPSP_SMART_HOST_TO_DEVICE 1
#define NO_A_COL 1
#define NO_PROW 1
#define MAKE_PROW_GPU 1
#define NUM_THR 1024
#define BITXBYTE 8

int compactcol(int *, int *, itype);

#define STOP CHECK_DEVICE(cudaDeviceSynchronize());MPI_Finalize();exit(0);
//------------------------------------------------------------------------------

__global__
void _getNNZ(itype n, itype *to_get_form_prow, itype * row, itype *nnz_to_get_form_prow){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;
  itype j = to_get_form_prow[i];
  nnz_to_get_form_prow[i] = row[j+1] - row[j];

}

//------------------------------------------------------------------------------
__forceinline__
__device__
int binsearch(itype array[], itype size, itype value) {
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
void _fillPRow(
  itype n,
  itype rows2bereceived,

  itype *whichproc,
  itype *p_nnz_map,

  itype mypfirstrow,
  itype myplastrow,
  itype nzz_pre_local,
  itype Plocalnnz,

  itype *local_row,
  itype *row
){

  itype i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= n)
		return;


  // copy local
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
  row[i] = p_nnz_map[iv-1] * (iv > 0) + shift;
}

//------------------------------------------------------------------------------

__global__
void _fillPRowNoComm(
  itype n,
  itype mypfirstrow,
  itype myplastrow,
  itype Plocalnnz,
  itype *local_row,
  itype *row
){

  itype i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= n)
		return;

  // copy local
  if(i >= mypfirstrow && i <= myplastrow+1){
    row[i] = local_row[i-mypfirstrow];
    return;
  }
  row[i] = Plocalnnz * (i>myplastrow);
}

//------------------------------------------------------------------------------

__global__
void _getColMissing(
	itype nnz,

	itype mypfirstrow,
	itype myplastrow,

	itype *col,
	int *mask
){
	itype tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid >= nnz)
		return;

	itype c = col[tid];
#if 0
	int mask_idx = c / (sizeof(itype)*BITXBYTE);
	unsigned int m = 1 << ( (c%(sizeof(itype)*BITXBYTE)) );
#endif

 	if(c < mypfirstrow || c > myplastrow) {
            mask[c]=c;
#if 0            
			atomicOr(&mask[mask_idx], m);
#endif
    } 

}

//------------__-----_---------___--------__----__--_____------_-__-__-_-__-__--

__global__
void _getColVal(
	itype n,
	itype *rcvprow,
	itype *nnz_per_row,

	itype *row,
	itype *col,
	vtype *val,

	itype *col2get,
	vtype *val2get,

	itype row_shift
){

	itype q = blockDim.x * blockIdx.x + threadIdx.x;

	if(q >= n)
		return;

	itype I = rcvprow[q] - row_shift;

	itype start = row[I];
	itype end = row[I+1];


	for(itype i=start, j=nnz_per_row[q]; i<end; i++, j++){
		col2get[j] = col[i];
		val2get[j] = val[i];
	}
}

//------------__-----_---------___--------__----__--_____------_-__-__-_-__-__--


void *Malloc(size_t sz) {

	void *ptr;

	if (!sz) {
		printf("Allocating zero bytes...\n");
		exit(EXIT_FAILURE);
	}
	ptr = (void *)malloc(sz);
	if (!ptr) {
		fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	memset(ptr, 0, sz);
	return ptr;
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

    /* Copy the remaining elements of L[], if there
       are any */
    while (i < n1) {
        c[k] = a[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there
       are any */
    while (j < n2) {
        c[k] = b[j];
        j++;
        k++;
    }
    return k;
}

//------------__-----_---------___--------__----__--_____------_-__-__-_-__-__--
int spgemmcusparse(int A_num_rows, int A_num_cols, int A_nnz,
               int *dA_csrOffsets, int *dA_columns, double *dA_values,
               int B_num_rows, int B_num_cols, int B_nnz,
               int *dB_csrOffsets, int *dB_columns, double *dB_values,
               int *p2C_nnz, int **p2dC_csrOffsets, int **p2dC_columns, double **p2dC_values);

// version0: A is local, P is FULL
CSR* nsparseMGPU_version0(CSR *Alocal, CSR *Pfull, csrlocinfo *Plocal, double &loc_time) {
  _MPI_ENV;
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

  double start;
  MPI_Barrier(MPI_COMM_WORLD);
  cudaDeviceSynchronize();
  if(ISMASTER) start = MPI_Wtime();

  //cudaProfilerStart();
  //spgemm_csrseg_kernel_hash(&mat_a, &mat_p, &mat_c, Plocal);
 spgemmcusparse(mat_a.M, mat_a.N,  mat_a.nnz,
               mat_a.d_rpt, mat_a.d_col, mat_a.d_val,
 	       mat_p.M, mat_p.N,  mat_p.nnz,
               mat_p.d_rpt, mat_p.d_col, mat_p.d_val,
               &mat_c.nnz, &(mat_c.d_rpt), &(mat_c.d_col) , &(mat_c.d_val)); 
  mat_c.M=mat_a.M;
  mat_c.N=mat_p.N;
  //cudaProfilerStop();

  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  if(ISMASTER) loc_time = (MPI_Wtime() - start);  

  CSR* C = CSRm::init(mat_c.M, mat_c.N, mat_c.nnz, false, true, false, Alocal->full_n, Alocal->row_shift);
  C->row = mat_c.d_rpt;
  C->col = mat_c.d_col;
  C->val = mat_c.d_val;

  return C;
}

CSR* nsparseMGPU_version1(handles *h, CSR *Alocal, CSR *Plocal, double &loc_time){
  _MPI_ENV;
	gridblock gb;
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
		return nsparseMGPU_version0(Alocal, Plocal, &Pinfo1p,loc_time);
	}else{
    if (ISMASTER) printf("cusparse version must be modified for nprocs > 1, exiting\n");
    exit(1);
  }

	double starttime, commtime, endtime, precommtime, infocommtime, copytime, cpucopytime, deviceToHost, deviceToHost_start;

  itype Pn = Plocal->full_n, Am = Alocal->m, Pm = Plocal->m;

  //copy to HOST Alocal

	if(ISMASTER)
		deviceToHost_start = MPI_Wtime();


	// copy Alocal to hos ----------------__----_---_----_-__--____---------__-__---_--_-__-_--_--_-_
  itype rowsaxproc = Alocal->n; // == A->n/nprocs

#if NO_A_COL == 0
	itype *locarow = (itype*) malloc( (rowsaxproc + 1) * sizeof(itype) );
	CHECK_HOST(locarow);

	CHECK_DEVICE(
		cudaMemcpy(locarow, Alocal->row, (rowsaxproc + 1) * sizeof(itype), cudaMemcpyDeviceToHost);
	);

	itype *locacol = (itype*) malloc( (Alocal->nnz) * sizeof(itype) );
	CHECK_HOST(locacol);
	CHECK_DEVICE(
		cudaMemcpy(locacol, Alocal->col, Alocal->nnz * sizeof(itype), cudaMemcpyDeviceToHost);
	);
#endif
	//----------------__----_---_----_-__--____---------__-__---_--_-__-_--_--_-__--_-__-------_--_
  //copy to HOST Plocal
  //CSR *h_Plocal = CSRm::copyToHost(Plocal);
	//CSR *h_Plocal = CSRm::init(Plocal->n, Plocal->m, Plocal->nnz, true, false, false, Plocal->full_n, Plocal->row_shift);

	itype *locprow;

#if SPSP_SMART_HOST_TO_DEVICE == 0
	itype *locpcol;
	vtype *locpval;

	CHECK_DEVICE(
		cudaMallocHost(&locprow, (Plocal->n+1) * sizeof(itype));
	);
	CHECK_DEVICE(
		cudaMallocHost(&locpcol, Plocal->nnz * sizeof(itype));
	);
	CHECK_DEVICE(
		cudaMallocHost(&locpval, Plocal->nnz * sizeof(vtype));
	);

	cudaMemcpyAsync(locprow, Plocal->row, (Plocal->n+1) * sizeof(itype), cudaMemcpyDeviceToHost, h->stream1);
	cudaMemcpyAsync(locpcol, Plocal->col, Plocal->nnz * sizeof(itype), cudaMemcpyDeviceToHost, h->stream2);
	cudaMemcpyAsync(locpval, Plocal->val, Plocal->nnz * sizeof(vtype), cudaMemcpyDeviceToHost, h->stream2);
#else

#if NO_PROW == 0
	CHECK_DEVICE(
		cudaMallocHost(&locprow, (Plocal->n+1) * sizeof(itype));
	);
	cudaMemcpyAsync(locprow, Plocal->row, (Plocal->n+1) * sizeof(itype), cudaMemcpyDeviceToHost, h->stream1);
#endif
#endif


	//--__-__-----------------------_---_------------___---______----------------__-_------_--__-_-

	if(ISMASTER)
		deviceToHost = MPI_Wtime() - deviceToHost_start;

	if(myid==0) {
		  starttime=MPI_Wtime();
	    infocommtime=starttime;
	    precommtime=starttime;
	  }

  itype mycolp = Plocal->nnz; //colpxproc[myid]
  // collect total nnz for P
  itype nnzp = 0;

	CHECK_MPI(
    MPI_Allreduce(
      &Plocal->nnz,
      &nnzp,
      1,
      MPI_INT,
      MPI_SUM,
      MPI_COMM_WORLD
    )
  );
  //------------------------


  // send baremin main-------------------------------------------------------------
  itype *Prow, *Pcol;
  vtype *Pval;
  itype *whichprow=NULL, *rcvpcolxrow=NULL, *rcvprow=NULL, *col2send=NULL;
  vtype *val2send=NULL;

  int rcounts[nprocs], scounts[nprocs];
  int displr[nprocs],  displs[nprocs];
  int rcounts2[nprocs], scounts2[nprocs];
  int displr2[nprocs],  displs2[nprocs];

  // changed
  //itype mypfirstrow = Plocal->row_shift;/* check if every task already has a subset of the second matrix and believes it is the whole matrix! */
  //itype myplastrow = Plocal->n + Plocal->row_shift;

	itype mypfirstrow = myid * (Pn/nprocs);
	assert(mypfirstrow == Plocal->row_shift);

	itype myplastrow = ((myid==(nprocs-1))? Pn : (myid+1)*(Pn/nprocs)) -1;
	assert(myplastrow == (Plocal->n + Plocal->row_shift-1));

	//*
	//itype mypfirstrow=myid*(Pn/nprocs); /* check if every task already has a subset of the second matrix and believes it is the whole matrix! */
	//itype myplastrow= ((myid==(nprocs-1))? Pn : (myid+1)*(Pn/nprocs)) -1;
  //

  unsigned int countp[nprocs], offset[nprocs], rcvcntp[nprocs];
  unsigned int sothercol=0;
  int cntothercol=0;
  int whichproc;
  itype *othercol[1]={NULL};

  itype *bitcol[1];

unsigned int ii, i, j, k, l;
#if NO_A_COL == 0
  bitcol[0]=(itype *) Malloc( ((Pn+(((sizeof(itype)*BITXBYTE))-1))/(sizeof(itype)*BITXBYTE))*sizeof(itype) );

  for(i=0, cntothercol=0; i<rowsaxproc; i++) {
   for(j=locarow[i]; j<locarow[i+1]; j++) {
     if(locacol[j]<mypfirstrow||locacol[j]>myplastrow) {
			 ii=locacol[j];

		if((bitcol[sothercol][ii/(sizeof(itype)*BITXBYTE)]&(1<<((ii%(sizeof(itype)*BITXBYTE)))))==0) {
		  bitcol[sothercol][ii/(sizeof(itype)*BITXBYTE)]|=(1<<((ii%(sizeof(itype)*BITXBYTE))));
		  cntothercol++;
		}
    }
   }
  }
#else
	itype size_mask = (Pn+(((sizeof(int)*BITXBYTE))-1))/(sizeof(int)*BITXBYTE);
	vector<int> *dev_acol = Vector::init<int>(Pn, true, true);
	vector<int> *dev_othercol = Vector::init<int>(Pn, true, true);
	Vector::fillWithValue(dev_acol, -1);
	gb = gb1d(Alocal->nnz, NUM_THR);
	_getColMissing<<<gb.g, gb.b>>>(
		Alocal->nnz,
		mypfirstrow,
		myplastrow,
		Alocal->col,
		dev_acol->val
	);
    cudaDeviceSynchronize();
//    printf("task %d after getColMissing, Pn=%d\n",myid,Pn);
 //   MPI_Barrier(MPI_COMM_WORLD);
	cntothercol=compactcol(dev_acol->val,dev_othercol->val,Pn);
//    printf("task %d, cntothercol=%d, Pn=%d\n",myid,cntothercol,Pn);

    if(cntothercol>0) {
        othercol[0]=(itype *)Malloc(cntothercol*sizeof(itype));
        cudaMemcpy(othercol[0],dev_othercol->val,cntothercol*sizeof(int),cudaMemcpyDeviceToHost);
    }
	Vector::free(dev_acol);
	Vector::free(dev_othercol);

//	vector<int> *_bitcol = Vector::copyToHost(dev_bitcol);

//	bitcol[0] = _bitcol->val;
	//CHECK_DEVICE(cudaDeviceSynchronize());MPI_Finalize();exit(0);

	// count
//	cntothercol = 0;
//  for(i=0; i<size_mask; i++){
//			int cc = __builtin_popcount(bitcol[0][i]);
//			cntothercol += cc;
//	}
//

#endif
#if 0    
  if(cntothercol>0) {
  othercol[0]=(itype *)Malloc(cntothercol*sizeof(itype));
  for(i=0, j=0; i<mypfirstrow; i++) {
    if(bitcol[sothercol][i/(sizeof(itype)*BITXBYTE)]&(1<<((i%(sizeof(itype)*BITXBYTE))))) {
        if(j==cntothercol) {
	  fprintf(stderr,"task %d, invalid value %d %d\n",myid,j,cntothercol);
	  exit(1);
	}
	othercol[sothercol][j]=i;
	j++;
    }
  }
  for(i=myplastrow+1; i<Pn; i++) {
    if(bitcol[sothercol][i/(sizeof(itype)*BITXBYTE)]&(1<<((i%(sizeof(itype)*BITXBYTE))))) {
        if(j==cntothercol) {
	  fprintf(stderr,"task %d, invalid value %d %d\n",myid,j,cntothercol);
	  exit(1);
	}
	othercol[sothercol][j]=i;
	j++;
    }
  }
  if(j!=cntothercol) {
    fprintf(stderr,"task %d: error expected %d found %d\n",myid,cntothercol,j);
    exit(1);
  }
  }
#endif


#if NO_A_COL == 0
  free(bitcol[0]);
#else
//	Vector::free(_bitcol);
#endif
	printf("\n%d A2\n", myid);


 /* the last list is in othercol[sothercol] */
  for(i=0; i<nprocs; i++) {countp[i]=0;}
  itype countall=0;
  for(j=0; j<cntothercol; j++) {
	whichproc=othercol[sothercol][j]/(Pn/nprocs);
	if(whichproc>(nprocs-1)) { whichproc=nprocs-1; }
	countp[whichproc]++;
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
  itype rows2bereceived=countall;

  for(j=0; j<cntothercol; j++) {
  	whichproc=othercol[sothercol][j]/(Pn/nprocs);
  	if(whichproc>(nprocs-1)) { whichproc=nprocs-1; }
  	whichprow[offset[whichproc]+countp[whichproc]]=othercol[sothercol][j];
  	countp[whichproc]++;
  }

  if(countp[myid]!=0) {
     fprintf(stderr,"self countp should be zero! %d\n",myid);
     exit(1);
  }
  free(othercol[sothercol]);
	if(myid==0) {
	precommtime=MPI_Wtime();
	}

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


  if(MPI_Alltoallv(whichprow,scounts,displs,MPI_BYTE,rcvprow,rcounts,displr,MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoallv of whichprow rows\n");
     exit(1);
  }

/*

	rcvcntp: quante righe di P serve a A
	rcvprow: la lista righe che serve per ogni processo

*/

	CHECK_DEVICE( cudaStreamSynchronize(h->stream1) );

	itype total_row_to_rec = 0;
	for(i=0; i<nprocs; i++){
		total_row_to_rec += rcvcntp[i];
	}

	vector<itype> *nnz_per_row_shift = NULL;
	if(total_row_to_rec > 0)
		nnz_per_row_shift = Vector::init<itype>(total_row_to_rec, true, false);

  itype *p2rcvprow;
  countall=0;
	itype q = 0;
	itype tot_shift = 0;

  for(i=0; i<nprocs; i++){
    scounts[i] = 0;
    displs[i] = 0;
  }

#if NO_PROW == 0

  if(total_row_to_rec){
    for(i=0; i<nprocs; i++){
      scounts[i]=0;

      displs[i] = (i == 0) ? 0 : (displs[i-1]+scounts[i-1]);

      if(i == myid)
  			continue;

      p2rcvprow = &rcvprow[ displr[i] / sizeof(itype) ];

  		// da inviare
      for(j=0; j<rcvcntp[i]; j++) {
        // locale
        if((p2rcvprow[j]<mypfirstrow) || (p2rcvprow[j]>myplastrow)) {
          fprintf(stderr,"task %d: unexpected request %d from task %d\n",myid,p2rcvprow[j],i);
          exit(1);
        }
  			// quante col per per ogni riga in p2rcvprow
        //locprow[(p2rcvprow[j]-mypfirstrow)+1]-locprow[p2rcvprow[j]-mypfirstrow]
  			itype number_nnz_per_rec_row = locprow[ (p2rcvprow[j]-mypfirstrow)+1 ] - locprow[p2rcvprow[j]-mypfirstrow];
        scounts[i] += number_nnz_per_rec_row;

  			nnz_per_row_shift->val[q] = tot_shift;
  			tot_shift += number_nnz_per_rec_row;
  			q++;
      }
  		// tutto in byte
  		// quanti elementi
      countall+=scounts[i];
      scounts[i]*=sizeof(itype);
      displs[i]=((i==0)?0:(displs[i-1]+scounts[i-1]));
    }
  }
#else

  vector<itype> *nnz_to_get_form_prow = NULL;

  if(total_row_to_rec){
    vector<itype> *to_get_form_prow = Vector::init<itype>(total_row_to_rec, true, false);
    q = 0;
    // get index of rows to query to prow
    for(i=0; i<nprocs; i++){
      if(i == myid)
        continue;
      p2rcvprow = &rcvprow[ displr[i] / sizeof(itype) ];
      // da inviare
      for(j=0; j<rcvcntp[i]; j++){
        to_get_form_prow->val[q] = p2rcvprow[j]-mypfirstrow;
        q++;
      }
    }

    vector<itype> *dev_to_get_form_prow = Vector::copyToDevice(to_get_form_prow);
    vector<itype> *dev_nnz_to_get_form_prow = Vector::init<itype>(total_row_to_rec, true, true);

    gb = gb1d(total_row_to_rec, NUM_THR);
    _getNNZ<<<gb.g, gb.b>>>(total_row_to_rec, dev_to_get_form_prow->val, Plocal->row, dev_nnz_to_get_form_prow->val);

    nnz_to_get_form_prow = Vector::copyToHost(dev_nnz_to_get_form_prow);

    q = 0;
    for(i=0; i<nprocs; i++){
      scounts[i]=0;
      displs[i] = (i == 0) ? 0 : (displs[i-1]+scounts[i-1]);
      if(i == myid)
        continue;
      // da inviare
      for(j=0; j<rcvcntp[i]; j++) {
        itype number_nnz_per_rec_row = nnz_to_get_form_prow->val[q];
        scounts[i] += number_nnz_per_rec_row;
        nnz_per_row_shift->val[q] = tot_shift;
        tot_shift += number_nnz_per_rec_row;
        q++;
      }
      // tutto in byte
      // quanti elementi
      countall+=scounts[i];
      scounts[i]*=sizeof(itype);
      displs[i]=((i==0)?0:(displs[i-1]+scounts[i-1]));
    }

    Vector::free(dev_to_get_form_prow);
    Vector::free(to_get_form_prow);
    Vector::free(dev_nnz_to_get_form_prow);
    //Vector::free(nnz_to_get_form_prow);
  }

#endif

  if(countall>0) {
     col2send=(itype *)Malloc(sizeof(itype)*countall);
     val2send=(vtype *)Malloc(sizeof(vtype)*countall);
  }


#if SPSP_SMART_HOST_TO_DEVICE == 0

	CHECK_DEVICE( cudaStreamSynchronize(h->stream2) );

  for(i=0; i<nprocs; i++){

    countp[i]=0;
    if(i==myid)
			continue;

		// shift
    p2rcvprow = &rcvprow[displr[i]/sizeof(itype)];
		// same 363
    for(j=0; j<rcvcntp[i]; j++){
      //for(k=locprow[p2rcvprow[j]-mypfirstrow]-locprow[0]; k<(locprow[(p2rcvprow[j]-mypfirstrow)+1]-locprow[0]); k++){
			for(k=locprow[p2rcvprow[j]-mypfirstrow]; k<(locprow[(p2rcvprow[j]-mypfirstrow)+1]); k++){

				col2send[ (displs[i] / sizeof(itype)) + countp[i] ]=locpcol[k];

		    val2send[(displs[i]/sizeof(itype))+countp[i]]=locpval[k];
		    countp[i]++;
      }
			// ?
      p2rcvprow[j] = locprow[(p2rcvprow[j]-mypfirstrow)+1]-locprow[p2rcvprow[j]-mypfirstrow]; /* recycle rcvprow to send the number of columns in each row */
    }
  }
#else

CHECK_DEVICE( cudaStreamSynchronize(h->stream2) );

	if(countall>0){
		//itype *col2send2 =(itype *)Malloc(sizeof(itype)*countall);
		//vtype *val2send2 = (vtype *)Malloc(sizeof(vtype)*countall);

	 vector<itype> *dev_nnz_per_row_shift = Vector::copyToDevice(nnz_per_row_shift);

	 itype *dev_rcvprow;
	 CHECK_DEVICE( cudaMalloc( (void**) &dev_rcvprow, dev_nnz_per_row_shift->n * sizeof(itype)) );
	 CHECK_DEVICE( cudaMemcpy(dev_rcvprow, rcvprow, dev_nnz_per_row_shift->n * sizeof(itype), cudaMemcpyHostToDevice) );

   q = 0;
		for(i=0; i<nprocs; i++){
			if(i==myid)
				continue;
			// shift
			p2rcvprow = &rcvprow[displr[i]/sizeof(itype)];
			for(j=0; j<rcvcntp[i]; j++){
				//p2rcvprow[j] = locprow[(p2rcvprow[j]-mypfirstrow)+1]-locprow[p2rcvprow[j]-mypfirstrow]; // recycle rcvprow to send the number of columns in each row
        p2rcvprow[j] = nnz_to_get_form_prow->val[q]; // recycle rcvprow to send the number of columns in each row
        q++;
			}
		}

    //p2rcvprow = nnz_to_get_form_prow->val;

		itype *dev_col2send = NULL;
		vtype *dev_val2send = NULL;
		CHECK_DEVICE( cudaMalloc( (void**) &dev_col2send, countall * sizeof(itype)) );
		CHECK_DEVICE( cudaMalloc( (void**) &dev_val2send, countall * sizeof(vtype)) );

		gb = gb1d(dev_nnz_per_row_shift->n, NUM_THR);

		_getColVal<<<gb.g, gb.b>>>(
			dev_nnz_per_row_shift->n,

			dev_rcvprow,
			dev_nnz_per_row_shift->val,

			Plocal->row,
			Plocal->col,
			Plocal->val,

			dev_col2send,
			dev_val2send,

			mypfirstrow
		);

		CHECK_DEVICE( cudaMemcpy(col2send, dev_col2send, countall * sizeof(itype), cudaMemcpyDeviceToHost) );
		CHECK_DEVICE( cudaMemcpy(val2send, dev_val2send, countall * sizeof(vtype), cudaMemcpyDeviceToHost) );


		Vector::free(dev_nnz_per_row_shift);
		cudaFree(dev_col2send);
		cudaFree(dev_val2send);

		//CHECK_DEVICE( cudaDeviceSynchronize() );
	}
#endif

  if(MPI_Alltoallv(rcvprow,scounts2,displs2,MPI_BYTE,rcvpcolxrow,rcounts2,displr2,MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoallv of rcvprow\n");
     exit(1);
  }

  Prow=(itype *)Malloc(sizeof(itype)*(Pn+1));
  Prow[0]=0;

  for(i=0; i<nprocs; i++) {
    rcounts[i]=0;
  }

#if MAKE_PROW_GPU == 0
  for(i=1, j=1, k=0, l=0; i<=Pn; i++) {
		if(whichprow!=NULL && i== (whichprow[l]+1)&&(l<rows2bereceived)) {
			k+=rcvpcolxrow[l];
			whichproc=whichprow[l]/(Pn/nprocs);

    	if(whichproc>(nprocs-1)){
				whichproc=nprocs-1;
			}
	   rcounts[whichproc]+=rcvpcolxrow[l];
	   l++;
    }

    if((i>mypfirstrow)&&(i<=(myplastrow+1))) { /* it is a row of mine */
      k+=(locprow[j]-locprow[j-1]);
      j++;
    }
    Prow[i]=k;
  }
#else
//____----____-------_____-----_____----____----____---____---______----___----_-------_____-----_____----____----____---____---______----___----_

  /*
	for(i=1, j=1, k=0, l=0; i<=Pn; i++) {
		if(whichprow!=NULL &&i==(whichprow[l]+1)&&(l<rows2bereceived)){
			k+=rcvpcolxrow[l];
			whichproc=whichprow[l]/(Pn/nprocs);

			if(whichproc>(nprocs-1))
				whichproc=nprocs-1;
		 // how many to send every process
		 //rcounts[whichproc] += rcvpcolxrow[l];
		 l++;
		}

		if((i>mypfirstrow)&&(i<=(myplastrow+1))) { // it is a row of mine
			k+=(locprow[j]-locprow[j-1]);
			j++;
		}
		//Prow[i]=k;
	}
  */

  vector<itype> *bareminp_row = Vector::init<itype>(Plocal->full_n+1, true, true);

  itype nzz_pre_local = 0;

  if(rows2bereceived){
    // we have rows from other process
  	bool flag = true;
    itype *dev_whichproc = NULL;
    itype r = 0;

    vector<itype> *P_nnz_map = Vector::init<itype>(rows2bereceived, true, false);

  	k = 0;
  	for(i=0; i<rows2bereceived; i++){

  		r = whichprow[i];

      // count nnz per process for comunication
      whichproc=r/(Pn/nprocs);
      if(whichproc>(nprocs-1))
        whichproc=nprocs-1;
      rcounts[whichproc] += rcvpcolxrow[i];
      //-------------------------------------------

  		// after local add shift
  		if(r > mypfirstrow && flag){
  			//k += Plocal->nnz;
        nzz_pre_local = k;
  			flag = false;
  		}
  		k += rcvpcolxrow[i];
  		P_nnz_map->val[i] = k;
  	}

    if(flag)
      nzz_pre_local = P_nnz_map->val[rows2bereceived-1];

    CHECK_DEVICE(
      cudaMalloc( (void**) &dev_whichproc, rows2bereceived * sizeof(itype) );
    );
    CHECK_DEVICE(
      cudaMemcpy(dev_whichproc, whichprow, rows2bereceived * sizeof(itype), cudaMemcpyHostToDevice);
    )

  	vector<itype> *dev_P_nnz_map = Vector::copyToDevice(P_nnz_map);

    gb = gb1d(Plocal->full_n+1, NUM_THR);

    _fillPRow<<<gb.g, gb.b>>>(
      Plocal->full_n+1,
      rows2bereceived,

      dev_whichproc,
      dev_P_nnz_map->val,

      mypfirstrow,
      myplastrow,
      nzz_pre_local,
      Plocal->nnz,

      Plocal->row,
      bareminp_row->val
    );

    Vector::free(dev_P_nnz_map);
    Vector::free(P_nnz_map);
    cudaFree(dev_whichproc);

  }else{
    // no comunication; copy and fill on the right of Prow
    gb = gb1d(Plocal->full_n+1, NUM_THR);

    _fillPRowNoComm<<<gb.g, gb.b>>>(
      Plocal->full_n+1,
      mypfirstrow,
      myplastrow,
      Plocal->nnz,
      Plocal->row,
      bareminp_row->val
    );
    /*
    if(false){
      vector<itype> *h_bareminp_row = Vector::copyToHost(bareminp_row);
      for(i=0; i<Plocal->full_n+1; i++){
        //if(i < mypfirstrow || i > myplastrow+1)
        //  printf("EXT %d %d %d---", mypfirstrow, i, myplastrow);
        // printf("%d]-%d %d %d %d", myid, i, Prow[i], h_bareminp_row->val[i], Prow[i]-h_bareminp_row->val[i]);

        if(Prow[i] != h_bareminp_row->val[i]){
          printf("||&&&&\n");
          assert(false);
        }
        //else
        //printf("\n");
      }
    }
  */
  }

//____----____-------_____-----_____----____----____---____---______----___----_-------_____-----_____----____----____---____---______----___----_
#endif

  if(rcounts[myid]!=0) {
     fprintf(stderr,"task: %d, unexpected rcount[%d]=%d. It should be zero\n",myid,myid,rcounts[myid]);
     exit(1);
  }

  for(i=0; i<nprocs; i++) {
      rcounts[i]*=sizeof(itype);
			displr[i]=(i==0)?0:(displr[i-1]+(i==(myid+1)?(mycolp*sizeof(itype)):rcounts[i-1]));
      //displr[i]=(i==0 || i==(nprocs-1))?0:(displr[i-1]+(i==(myid+1)?(mycolp*sizeof(itype)):rcounts[i-1]));
      // printf("task %d: scounts[%d]=%d, displs[%d]=%d, rcounts[%d]=%d, displr[%d]=%d, countall=%d\n",myid,i,scounts[i],i,displs[i],i,rcounts[i],i,displr[i],countall);
  }
	if(myid==0) {
	infocommtime=MPI_Wtime();
	}
  free(whichprow);
  free(rcvpcolxrow);
  free(rcvprow);

  // @REDO
  Pcol=(itype *)Malloc(sizeof(itype)*nnzp);
  Pval=(vtype *)Malloc(sizeof(vtype)*nnzp);
  if(MPI_Alltoallv(col2send,scounts,displs,MPI_BYTE,Pcol,rcounts,displr,MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoallv of col2send\n");
     exit(1);
  }
  free(col2send);
  for(i=0; i<nprocs; i++) {
      scounts[i]*=(sizeof(vtype)/sizeof(itype));
      displs[i]*=(sizeof(vtype)/sizeof(itype));
      rcounts[i]*=(sizeof(vtype)/sizeof(itype));
      displr[i]*=(sizeof(vtype)/sizeof(itype));
  }

  if(MPI_Alltoallv(val2send,scounts,displs,MPI_BYTE,Pval,rcounts,displr,MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoallv of val2send\n");
     exit(1);
  }

	free(val2send);

	if(myid==0) {
		copytime=MPI_Wtime();
	}
	CSR *baremin_Plocal = NULL;

#if SPSP_SMART_HOST_TO_DEVICE == 1
	itype total_recv = 0;

	baremin_Plocal = CSRm::init(Pn, Pm, nnzp, true, true, false, Pn, Alocal->row_shift);

	for(i=0; i<nprocs; i++){
		displr[i] /= sizeof(vtype);
		rcounts[i] /= sizeof(vtype);
		total_recv += rcounts[i];
	}

	for(i=0; i<nprocs; i++){
		if(rcounts[i]>0) {
			CHECK_DEVICE(
				cudaMemcpy(baremin_Plocal->val+displr[i], Pval+displr[i], rcounts[i] * sizeof(vtype), cudaMemcpyHostToDevice)
			);
			CHECK_DEVICE(
				cudaMemcpy(baremin_Plocal->col+displr[i], Pcol+displr[i], rcounts[i] * sizeof(itype), cudaMemcpyHostToDevice)
			);
		}
	}

#if MAKE_PROW_GPU == 1
  baremin_Plocal->row = bareminp_row->val;
#else
	CHECK_DEVICE(
		cudaMemcpy(baremin_Plocal->row, Prow,  (baremin_Plocal->n+1) * sizeof(itype), cudaMemcpyHostToDevice)
	);
#endif

#else

  CSR *h_baremin_Plocal = CSRm::init(Pn, Pm, nnzp, false, false, false, Pn, Alocal->row_shift);
  h_baremin_Plocal->row = Prow;
  h_baremin_Plocal->col = Pcol;
  h_baremin_Plocal->val = Pval;
  baremin_Plocal = CSRm::copyToDevice(h_baremin_Plocal);

#endif
#if !defined(CSRSEG) 
	// copy local
	CHECK_DEVICE(
		cudaMemcpy(baremin_Plocal->val + nzz_pre_local, Plocal->val, Plocal->nnz * sizeof(vtype), cudaMemcpyDeviceToDevice);
	);

	CHECK_DEVICE(
		cudaMemcpy(baremin_Plocal->col + nzz_pre_local, Plocal->col, Plocal->nnz * sizeof(itype), cudaMemcpyDeviceToDevice);
	);

#endif
	if(myid==0) {
		commtime=MPI_Wtime();
  }
  
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

  CSR *C = nsparseMGPU_version0(Alocal, baremin_Plocal, &Plocalinfo, loc_time);

	if(myid==0) {
		endtime=MPI_Wtime();
	}

	if(ISMASTER)
		printf("@META deviceToHost=%f Kern+Comm=%f, Comm=%f, PreComm=%f, InfoComm=%f, DataComm=%f, Copy=%f\n", deviceToHost, endtime-starttime,commtime-starttime,precommtime-starttime,infocommtime-precommtime,copytime-infocommtime,commtime-copytime);

  printf("\ncheck free\n");

  free(Pcol);
  free(Prow);
  free(Pval);

	CSRm::free(baremin_Plocal);
#if NO_PROW == 0
	CHECK_DEVICE( cudaFreeHost(locprow) );
#else
  if(nnz_to_get_form_prow != NULL)
    Vector::free(nnz_to_get_form_prow);
#endif
#if SPSP_SMART_HOST_TO_DEVICE == 0
	CHECK_DEVICE( cudaFreeHost(locpcol) );
	CHECK_DEVICE( cudaFreeHost(locpval) );
#endif

  return C;
}

//------------------------------------------------------------------------------
