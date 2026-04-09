#include "spspmpi.h"
#include "csrseg.h"
#include "cudamacro.h"
#include "matrixIO.h"
#define SPSP_SMART_HOST_TO_DEVICE 1
#define NO_A_COL 1
#define NO_PROW 1
#define MAKE_PROW_GPU 1
#define NUM_THR 1024
#define BITXBYTE 8
//#define CSRSEG 
#undef CSRSEG


#define STOP CHECK_DEVICE(cudaDeviceSynchronize());MPI_Finalize();exit(0);
//------------------------------------------------------------------------------

//------------__-----_---------___--------__----__--_____------_-__-__-_-__-__--
__global__
void _getNNZ(itype n, itype *to_get_form_prow, itype * row, itype *nnz_to_get_form_prow){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;
  itype j = to_get_form_prow[i];
  nnz_to_get_form_prow[i] = row[j+1] - row[j];

}

__global__
void _completedP_rows (itype completedP_n, itype* new_rows) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (id < completedP_n)
        new_rows[id] = id;
}

__global__
void _completedP_rows2( itype completedP_n, itype rows_pre_local, itype local_rows, itype nzz_pre_local, itype Plocal_nnz, itype* Plocal_row, itype* P_nnz_map, itype* completedP_row ) {
    itype id = blockDim.x * blockIdx.x + threadIdx.x;
    itype nnz_map_val;

    if(id < completedP_n) {
        if (id < rows_pre_local) {
            nnz_map_val = (id == 0) ? 0 : P_nnz_map[id-1];
            completedP_row[id] = nnz_map_val;
        } else {
            if (id < rows_pre_local + local_rows) {
                completedP_row[id] = nzz_pre_local + Plocal_row[id-rows_pre_local]; // id - rows_pre_local
            } else {
                nnz_map_val = (id-local_rows == 0) ? 0 : P_nnz_map[id-local_rows-1];
                completedP_row[id] = Plocal_nnz + nnz_map_val; // id - local_rows
            }
        }
    }
}


__global__
void apply_mask_permut_GPU_noSideEffects_glob (itype nnz, const itype *col, int shrinking_permut_len, const itype* shrinking_permut, itype* comp_col) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    int number_of_permutations = shrinking_permut_len, start, med, end, flag;
    
    if (id < nnz) {
        flag = 1;
        start = 0;
        end = number_of_permutations;
        while (flag && (end >= start) ) {
            med = start + (end - start)/2;
            if (col[id] == shrinking_permut[med]) {
                comp_col[id] = med;
                flag = 0;
            }else{
                if (col[id] < shrinking_permut[med])
                    end = med -1;
                else
                    start = med +1;
            }
        }
    }
    
    return;
}

vector<itype>* apply_mask_permut_GPU_noSideEffects (const CSR *Alocal, const vector<itype>* shrinking_permut) {
    assert(Alocal->on_the_device);
    assert(shrinking_permut->on_the_device);
    
    vector<itype>* comp_col;
    if (Alocal->custom_alloced) {
        comp_col = Vector::init<itype>(Alocal->nnz, true, true);
    } else {
        comp_col = Vector::init<itype>(Alocal->nnz, true, true);
    }
    // ---------------------------------------------
    
    
    gridblock gb;
    gb = gb1d(Alocal->nnz, NUM_THR);
    apply_mask_permut_GPU_noSideEffects_glob<<<gb.g, gb.b>>>(Alocal->nnz, Alocal->col, shrinking_permut->n, shrinking_permut->val, comp_col->val);
    
    return(comp_col);
}

vector<int> *get_shrinked_col( CSR *Alocal, CSR *Plocal ){
  _MPI_ENV;
  stype myplastrow;
  // ----------------- NOTE temp 4 debug -------------------------------
//   int static Ncall = 0;
//   Ncall ++;
//   printf("[%d] get_shrinked_col call number %d\n", myid, Ncall);
  // -------------------------------------------------------------------
  
  //gridblock gb;
  int *getmct_4shrink(itype *,itype,itype,itype,int,int*,int**,int*,int*,int);
  
  if ( Plocal != NULL ){
    myplastrow  = Plocal->n -1;
  }else{
    myplastrow  = Alocal->n -1;
  }// P_n_per_process[i]: number of rows that process i have of matrix P
  
  if(Alocal->nnz==0) { return NULL; }
  int uvs;
  int first_or_last = 0;
  if(myid == 0) {
     first_or_last=-1;
  }
  if(myid == (nprocs-1)) {
     first_or_last=1;
  }
  
  int *ptr = getmct_4shrink( Alocal->col, Alocal->nnz, 0, myplastrow, first_or_last, &uvs, &(Alocal->bitcol), &(Alocal->bitcolsize), &(Alocal->post_local), NUM_THR);
  vector<int> *_bitcol = Vector::init<int>(uvs, false, true);
  _bitcol->val=ptr;
  return _bitcol;
}


bool shrink_col(CSR* A, CSR* P) {
    vector<int> *get_shrinked_col( CSR*, CSR* );
    if (!(A->shrinked_flag)) {
        if ( P != NULL ) {    // product compatibility check
            assert( A->m == P->full_n );
        } else {
            assert( A->m == A->full_n );
        }
  
        vector<itype>* shrinking_permut = get_shrinked_col( A, P );
        assert ( shrinking_permut->n >= (P!=NULL ? P->n : A->n) );
        vector<itype>* shrinkedA_col = apply_mask_permut_GPU_noSideEffects (A, shrinking_permut);
        
        A->shrinked_flag = true;
        A->shrinked_m = shrinking_permut->n;
        A->shrinked_col = shrinkedA_col->val;

        Vector::free(shrinking_permut);
        std::free(shrinkedA_col);
        return (true);
    }else{
        return (false);
    }
}

CSR* get_shrinked_matrix(CSR* A, CSR* P) {
    
    if (!(A->shrinked_flag)) {
        assert( shrink_col(A, P) );
    } else {
        bool test = (P!=NULL) ? (P->row_shift == A->shrinked_firstrow) : (A->row_shift == A->shrinked_firstrow);
        test = test && ((P!=NULL) ? (P->row_shift+P->n == A->shrinked_lastrow) : (A->row_shift+A->n == A->shrinked_lastrow));
        assert ( test ); // NOTE: check Pfirstrow, Plastrow
    }
    
    CSR* A_ = CSRm::init(A->n, A->shrinked_m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);
    A_->row = A->row;
    A_->val = A->val;
    A_->col = A->shrinked_col;
    
    return(A_);
}
#if 0
__global__
void _getColVal( itype n, itype *rcvprow, itype *nnz_per_row, const itype * __restrict__ row,
    const itype *__restrict__ col, const vtype * __restrict__ val, itype *col2get, vtype *val2get, itype Pn, itype Pnnz ){
    itype q = blockDim.x * blockIdx.x + threadIdx.x;
    if(q >= n){ return; }
    itype I = rcvprow[q];
    itype start = row[I];
    itype end = (I<(Pn-1)) ? row[I+1] : Pnnz ;
    for(itype i=start, j=nnz_per_row[q]; i<end; i++, j++){
        col2get[j] = col[i];
        val2get[j] = val[i];
    }
}
#else
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
#endif
//------------__-----_---------___--------__----__--_____------_-__-__-_-__-__--
void *Realloc(void *pptr, size_t sz) {
	void *ptr;
	if (!sz) {
	    printf("Allocating zero bytes...\n");
	    exit(EXIT_FAILURE);
	}
	ptr = (void *)realloc(pptr, sz);
	if (!ptr) {
		fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	return ptr;
}

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

//------------__-----_---------___--------__----__--_____------_-__-__-_-__-__--

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
#ifdef NSP2_NSPARSE
  nsp_spgemm_kernel_hash(&mat_a, &mat_p, &mat_c);
  //printf("NSP2 version\n");
#else
  spgemm_csrseg_kernel_hash(&mat_a, &mat_p, &mat_c, Plocal);
  //printf("NSPARSE version\n");
#endif
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

int bswhichprocess(gstype *P_n_per_process, int nprocs, gstype e){
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

vector<int> *get_missing_col( CSR *Alocal, CSR *Plocal ){
  _MPI_ENV;
  stype myplastrow;
  if(nprocs == 1){ 
     vector<int> *_bitcol = Vector::init<int>(1, true, false);
     return _bitcol; 
  }

// ----------------- NOTE temp 4 debug -------------------------------
//   int static Ncall = 0;
//   Ncall ++;
//   printf("[%d] get_missing_col call number %d\n", myid, Ncall);
// -------------------------------------------------------------------
  
  //gridblock gb;
  int *getmct(itype *,itype,itype,itype,int *,int**,int*,int);
  
  if ( Plocal != NULL ){
    myplastrow  = Plocal->n -1;
  }else{
    myplastrow  = Alocal->n -1;
  }// P_n_per_process[i]: number of rows that process i have of matrix P 

  if(Alocal->nnz==0) { return NULL; }
  int uvs;
  int *ptr = getmct( Alocal->col, Alocal->nnz, 0, myplastrow, &uvs, &(Alocal->bitcol), &(Alocal->bitcolsize), NUM_THR);
  if(uvs == 0){ 
     vector<int> *_bitcol = Vector::init<int>(1, true, false);
     return _bitcol; 
  } else {
    vector<int> *_bitcol = Vector::init<int>(uvs, false, false);
    _bitcol->val=ptr;
    return _bitcol;
  }
}

static itype *iPtemp1;
static vtype *vPtemp1;
#if defined(COMPUTECOLINDICES)
extern itype *hProw, *hPcol, hPnnz, hPn, hPshift;
#endif
CSR* nsparseMGPU_version1(handles *h, CSR *Alocal, CSR *Plocal, double &loc_time){

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
  
  if(nprocs == 1){ return nsparseMGPU_version0(Alocal, Plocal, &Pinfo1p, loc_time); }
#if !defined(ADDCOLSHIFT) && !defined(USESHRINKEDMATRIX)
  fprintf(stderr,"Either ADDCOLSHIFT or USESHRINKEDMATRIX must be defined\n");
  exit(1);
#endif

  vector<int> *_bitcol = NULL;
  // if (myid == 0) printf("start get_missing_col\n");
  // MPI_Barrier(MPI_COMM_WORLD);

  _bitcol = get_missing_col( Alocal, NULL );

  // MPI_Barrier(MPI_COMM_WORLD);
  // if (myid == 0) printf("finish get_missing_col\n");

  gstype row_shift[nprocs], ends[nprocs];

  itype *P_n_per_process;
  P_n_per_process = (itype*) Malloc(sizeof(itype)*nprocs);

  // send rows numbers to each process, Plocal->n local number of rows 
  if ( Plocal != NULL ){
    CHECK_MPI( MPI_Allgather( &Plocal->n, sizeof(itype), MPI_BYTE, P_n_per_process, sizeof(itype), MPI_BYTE, MPI_COMM_WORLD ) );
  }else{
    CHECK_MPI( MPI_Allgather( &Alocal->n, sizeof(itype), MPI_BYTE, P_n_per_process, sizeof(itype), MPI_BYTE, MPI_COMM_WORLD ) );
  }// P_n_per_process[i]: number of rows that process i owns of matrix P 

  itype *rcvpcolxrow=NULL, *rcvprow=NULL;

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

  unsigned int i, j, k;
  cntothercol=_bitcol->n;
  othercol[sothercol]=_bitcol->val;

  // the last list is in othercol[sothercol]
  for(i=0; i<nprocs; i++){
    countp[i]=0;
  }

  itype *aofwhichproc=(itype *)Malloc(sizeof(itype)*cntothercol); 

  gstype cum_p_n_per_process[nprocs];
  row_shift[0]=0;
  ends[0] = P_n_per_process[0];  
  cum_p_n_per_process[0]=P_n_per_process[0]-1;
  for(int i=1; i<nprocs; i++){
     cum_p_n_per_process[i]=cum_p_n_per_process[i-1] + P_n_per_process[i];
     row_shift[i]=row_shift[i-1]+P_n_per_process[i-1];
     ends[i] = ends[i-1]+ P_n_per_process[i];
  }
  assert(ends[nprocs-1] == Alocal->full_n);

  itype countall=0;
  
  for(j=0; j<cntothercol; j++) {
    whichproc = bswhichprocess(cum_p_n_per_process, nprocs, othercol[sothercol][j]+Alocal->row_shift);
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
  itype *whichprow4byte=NULL;
  gstype *whichprow=NULL;   
  if(countall>0) {
     whichprow4byte =(itype *)Malloc(sizeof(itype)*countall);
     whichprow =(gstype *)Malloc(sizeof(gstype)*countall);     
     rcvpcolxrow=(itype *)Malloc(sizeof(itype)*countall);
  }

  itype rows2bereceived=countall;


  for(j=0; j<cntothercol; j++) {
      whichproc=aofwhichproc[j];
      whichprow4byte[offset[whichproc]+countp[whichproc]]=othercol[sothercol][j]+
		                                     (Alocal->row_shift-(whichproc?ends[whichproc-1]:0));
      countp[whichproc]++;
  }
  free(aofwhichproc);

  if(countp[myid]!=0) {
     fprintf(stderr,"self countp should be zero! %d\n",myid);
     exit(1);
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

  if( MPI_Alltoallv(whichprow4byte,scounts,displs,MPI_BYTE,rcvprow,rcounts,displr,MPI_BYTE,MPI_COMM_WORLD) != MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoallv of whichprow rows\n");
     exit(1);
  }
  
  k=0;
  for(int i=0; i<nprocs; i++) {
    	    for(int j=0; j<(scounts[i]/sizeof(itype)); j++) {
	        	    whichprow[k]=whichprow4byte[k]+row_shift[i];
			    k++;
	    }
  }
  free(whichprow4byte);

  memset(scounts, 0, nprocs*sizeof(int));
  memset(displs, 0, nprocs*sizeof(int));
  vector<itype> *nnz_per_row_shift = NULL;
  // total_row_to_rec actually store the total rows to send, the sum of the number of rows we must send to each process i
  // rcvcntp[i] = number of rows to send to process i
  itype total_row_to_rec = countall;
  countall = 0;
  if(total_row_to_rec){
    nnz_per_row_shift = Vector::init<itype>(total_row_to_rec, true, false); // no temp buff is used on the HOST only;
#if 0 
    itype q = 0;
    itype tot_shift = 0;
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
#endif    
  }

  itype *Pcol, *col2send=NULL;
  vtype *Pval;
  vtype *val2send=NULL;
  gridblock gb;

  int rcounts_src[nprocs], displr_src[nprocs];
  int displr_target[nprocs];

  stype mycolp = Plocal->nnz;   // number of nnz stored by the process
  itype Pm = Plocal->m;         // number of columns in P  
  gstype mypfirstrow = Plocal->row_shift;
  gstype myplastrow  = Plocal->n + Plocal->row_shift-1;
 
  itype *p2rcvprow;
  
  itype q = 0;

  memset(rcounts, 0, nprocs*sizeof(int)); 

  itype *dev_col2send = NULL;
  vtype *dev_val2send = NULL;
 
  if(total_row_to_rec){

    vector<itype> *dev_nnz_per_row_shift = NULL;
    itype *dev_rcvprow;    
    if(nnz_per_row_shift->n>0) {
        dev_nnz_per_row_shift = Vector::init<itype>(nnz_per_row_shift->n, true, true); 
        CHECK_DEVICE( cudaMalloc( (void**) &dev_rcvprow, dev_nnz_per_row_shift->n * sizeof(itype)) );
        CHECK_DEVICE(cudaMemcpyAsync(dev_nnz_per_row_shift->val, nnz_per_row_shift->val, dev_nnz_per_row_shift->n * sizeof(itype), cudaMemcpyHostToDevice,h->stream1 ));
        CHECK_DEVICE( cudaMemcpyAsync(dev_rcvprow, rcvprow, dev_nnz_per_row_shift->n * sizeof(itype), cudaMemcpyHostToDevice, h->stream1)); 
    }
    q = 0;
    vector<itype> *to_get_form_prow = Vector::init<itype>(dev_nnz_per_row_shift->n, true, false);
    for(i=0; i<nprocs; i++){
      if(i==myid) { continue; }
      // shift
      p2rcvprow = &rcvprow[displr[i]/sizeof(itype)]; //recvprow will be modified
      for(j=0; j<rcvcntp[i]; j++){
        to_get_form_prow->val[q] = p2rcvprow[j] /* -mypfirstrow */;
        q++;
      }
    }

    vector<itype> *dev_to_get_form_prow = Vector::copyToDevice(to_get_form_prow);
    vector<itype> *dev_nnz_to_get_form_prow = Vector::init<itype>(dev_nnz_per_row_shift->n, true, true);

    assert( Plocal->on_the_device && (Plocal->row != NULL) );

    gb = gb1d(total_row_to_rec, NUM_THR);
    _getNNZ<<<gb.g, gb.b>>>(total_row_to_rec, dev_to_get_form_prow->val, Plocal->row, dev_nnz_to_get_form_prow->val);
#if 0    
    _getNNZ<<<gb.g, gb.b>>>(total_row_to_rec, dev_to_get_form_prow->val, Plocal->row, Plocal->nnz, Plocal->n, dev_nnz_to_get_form_prow->val);
#endif
    vector<itype> *nnz_to_get_form_prow = Vector::copyToHost(dev_nnz_to_get_form_prow);
    itype tot_shift = 0;
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
    // MPI_Barrier(MPI_COMM_WORLD);
    // for (int j = 0; j < nprocs; j++){
    //   if (myid == j){
    //     printf("%d      ",myid);
    //     for (int i = 0; i < nprocs; i++){
    //       if (displs[i] < 0) {
    //         for (int k = 0; k < i; k++){
    //           printf("%d ",scounts[k]);
    //         }
    //         printf("\n");
    //         break;
    //         // printf("%d %d\t\t",i,displs[i]);
    //       }
    //     }
    //     printf("\n");
    //   }
    //   MPI_Barrier(MPI_COMM_WORLD);
    // }
    

    countall = 0;
    vector<itype>*  nnz_per_row_shift = Vector::init<itype>(dev_nnz_per_row_shift->n, true, false);
    for (int i=0; i<nnz_to_get_form_prow->n; i++) {
      nnz_per_row_shift->val[i] = countall;
      countall += nnz_to_get_form_prow->val[i];
      rcvprow[i] = nnz_to_get_form_prow->val[i];
    }

    col2send=(itype *)Malloc(sizeof(itype)*countall);
    val2send=(vtype *)Malloc(sizeof(vtype)*countall);

    // size_t free, total; 
    // cudaMemGetInfo( &free, &total );

    // if (countall * (sizeof(itype) + sizeof(vtype)) > free){
    //   std::cout << "Not enough memory, GPU " << myid << " memory: free=" << free / 1024 / 1024 << " MB, total=" << total / 1024 / 1024 << " MB" \
    //   << " memory needed: " << (countall * (sizeof(itype) + sizeof(vtype))) / 1024 / 1024 << " MB" << std::endl;
    //   exit(1);
    // }
    // if (myid == 0) printf("line 649\n");
    // MPI_Barrier(MPI_COMM_WORLD);

    CHECK_DEVICE( cudaMalloc( (void**) &dev_col2send, countall * sizeof(itype)) );
    CHECK_DEVICE( cudaMalloc( (void**) &dev_val2send, countall * sizeof(vtype)) );

    CHECK_DEVICE( cudaMemcpy(dev_nnz_per_row_shift->val, nnz_per_row_shift->val, dev_nnz_per_row_shift->n * sizeof(itype), cudaMemcpyHostToDevice) );

    if(nnz_per_row_shift->n>0) {
        gb = gb1d(nnz_to_get_form_prow->n, NUM_THR);
        _getColVal<<<gb.g, gb.b, 0, h->stream1>>>( nnz_to_get_form_prow->n, dev_to_get_form_prow->val, dev_nnz_per_row_shift->val, Plocal->row, Plocal->col, Plocal->val, dev_col2send, dev_val2send, 0 /* mypfirstrow */);
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
  // if (myid == 0) printf("line 676\n");
  // MPI_Barrier(MPI_COMM_WORLD);
 
  itype nzz_pre_local = 0;
  itype rows_pre_local = 0;
  vector<itype> *dev_P_nnz_map = NULL;
  
  if(rows2bereceived){
    // we have rows from other process
    bool flag = true;
//     itype *dev_whichproc = NULL;
    gsstype r = 0;

    vector<itype> *P_nnz_map = Vector::init<itype>(rows2bereceived, true, false);
    k = 0;
    int whichproc;

    gstype cum_p_n_per_process[nprocs];
    cum_p_n_per_process[0]=P_n_per_process[0]-1;
    for(int j=1; j<nprocs; j++){
     cum_p_n_per_process[j]=cum_p_n_per_process[j-1] + P_n_per_process[j];
    }
    for(i=0; i<rows2bereceived; i++){
      r = whichprow[i];
      // count nnz per process for comunication
      whichproc = bswhichprocess(cum_p_n_per_process, nprocs, r);
      if(whichproc>(nprocs-1)){
        whichproc=nprocs-1;
      }
      rcounts[whichproc] += rcvpcolxrow[i];
  	  // after local add shift
      if(r > mypfirstrow && flag){
          nzz_pre_local = k;
          rows_pre_local = i;
          flag = false;
      }
      k += rcvpcolxrow[i];
      P_nnz_map->val[i] = k;
    }

    if(flag){
      nzz_pre_local = P_nnz_map->val[rows2bereceived-1];
      rows_pre_local = rows2bereceived;
    }
    
    dev_P_nnz_map = Vector::copyToDevice(P_nnz_map);
    Vector::free(P_nnz_map);
  }
  // if (myid == 0) printf("line 724\n");
  // MPI_Barrier(MPI_COMM_WORLD);
  
  if(rcounts[myid]!=0) {
     fprintf(stderr,"task: %d, unexpected rcount[%d]=%d. It should be zero\n",myid,myid,rcounts[myid]);
     exit(1);
  }


  int totcell=0;
  static int s_totcell_new;
  memcpy(rcounts_src, rcounts, nprocs*sizeof(itype));
  for(i=0; i<nprocs; i++) {
      totcell += rcounts[i];
      displr_target[i]=(i==0)?0:(displr_target[i-1]+(i==(myid+1)?mycolp:rcounts_src[i-1]));
      rcounts[i]*=sizeof(itype);
      displr[i]=(i==0)?0:(displr[i-1]+rcounts[i-1]);
      displr_src[i]=displr[i]/sizeof(itype);
  }
  // if (myid == 0) printf("line 743\n");
  // MPI_Barrier(MPI_COMM_WORLD);
  
  if (iPtemp1 == NULL && totcell > 0){ // first allocation
    MY_CUDA_CHECK( cudaMallocHost( &iPtemp1, totcell*sizeof(itype) ) );
    vPtemp1 = (vtype*) Malloc( totcell*sizeof(vtype) );
    s_totcell_new = totcell;
  }
  if (totcell > s_totcell_new){ // not enough space
    MY_CUDA_CHECK( cudaFreeHost(iPtemp1));
    printf("[Realloc] --- totcell: %d s_totcell_new: %d\n",totcell,s_totcell_new);
    MY_CUDA_CHECK( cudaMallocHost(&iPtemp1, sizeof (itype) * totcell ) );
    vPtemp1 = (vtype*) Realloc( vPtemp1, totcell * sizeof(vtype) );
    s_totcell_new = totcell; 
  }
  Pcol = iPtemp1;
  Pval = vPtemp1;
  if (countall>0){
      cudaStreamSynchronize (h->stream1);
      dev_col2send = NULL;
  }
  // // if (myid == 0) printf("line 764\n");
  // // MPI_Barrier(MPI_COMM_WORLD);

  // // Check arguments
  // if (col2send == NULL || Pcol == NULL) {
  //     printf("Error: Send or receive buffer is null\n");
  //     MPI_Abort(MPI_COMM_WORLD, 1);
  // }
  // int sendcount_sum = 0;
  // int recvcount_sum = 0;
  // for (int i = 0; i < nprocs; ++i) {
  //     if (scounts[i] < 0 || rcounts[i] < 0 || displs[i] < 0 || displr[i] < 0) {
  //         printf("Error: scount=%d rcount=%d displs=%d displr=%d\n", scounts[i], rcounts[i], displs[i], displr[i]);
  //         MPI_Abort(MPI_COMM_WORLD, 1);
  //     }
  //     sendcount_sum += scounts[i];
  //     recvcount_sum += rcounts[i];
  // }
  // if (sendcount_sum != recvcount_sum) {
  //     printf("Error: Sum of send counts does not match sum of receive counts\n");
  //     MPI_Abort(MPI_COMM_WORLD, 1);
  // }

  if(MPI_Alltoallv(col2send,scounts,displs,MPI_BYTE,Pcol,rcounts,displr,MPI_BYTE,MPI_COMM_WORLD)!=MPI_SUCCESS) {
     fprintf(stderr,"Error in MPI_Alltoallv of col2send\n");
     exit(1);
  }
  // CHECK_MPI(MPI_Alltoallv(col2send,scounts,displs,MPI_BYTE,Pcol,rcounts,displr,MPI_BYTE,MPI_COMM_WORLD));
  // if (myid == 0) printf("line 771\n");
  // MPI_Barrier(MPI_COMM_WORLD);
  k=0;
  for(int i=0; i<nprocs; i++) {
    	    for(int j=0; j<rcounts_src[i]; j++) {
	        	    Pcol[k]=Pcol[k]+((i?ends[i-1]:0)-Plocal->row_shift);
			    k++;
	    }
  }
  // if (myid == 0) printf("line 778\n");
  // MPI_Barrier(MPI_COMM_WORLD);

  CSR *completedP = NULL;
  itype completedP_n = Plocal->n + rows2bereceived; //Alocal->rows_to_get->total_row_to_rec; (?NOTE?)
  itype completedP_nnz = Plocal->nnz + totcell;

#if defined(ADDCOLSHIFT)
  completedP = CSRm::init(completedP_n, Pm, completedP_nnz, true, true, false, completedP_n, Alocal->row_shift);
#endif
#if defined(USESHRINKEDMATRIX)
  completedP = CSRm::init(completedP_n, Pm, completedP_nnz, true, true, false, Pm, Alocal->row_shift);
#endif  
  // --------------------------------------------------------------------------------------
  
  gb = gb1d(completedP_n +1, NUM_THR);
  _completedP_rows2<<<gb.g, gb.b>>>( completedP_n +1, rows_pre_local, myplastrow - mypfirstrow +1, nzz_pre_local, Plocal->nnz, Plocal->row, dev_P_nnz_map->val, completedP->row );

// _completedP_rows<<<gb.g, gb.b>>>( completedP_n +1, completedP->row );
  if(rows2bereceived) {
    Vector::free(dev_P_nnz_map);
  }
  
  for(i=0; i<nprocs; i++){
    if(rcounts_src[i]>0) {
        CHECK_DEVICE( cudaMemcpyAsync(completedP->col+displr_target[i], Pcol+displr_src[i], rcounts_src[i] * sizeof(itype), cudaMemcpyHostToDevice, h->stream1)  );
    }
  }
  // if (myid == 0) printf("line 806\n");
  // MPI_Barrier(MPI_COMM_WORLD);

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
        CHECK_DEVICE(  cudaMemcpy(completedP->val+displr_target[i], Pval+displr_src[i], rcounts_src[i] * sizeof(vtype), cudaMemcpyHostToDevice)  );
    }
  }
  
#if !defined(CSRSEG)
  CHECK_DEVICE( cudaMemcpy(completedP->val + nzz_pre_local, Plocal->val, Plocal->nnz * sizeof(vtype), cudaMemcpyDeviceToDevice);  );
  CHECK_DEVICE( cudaMemcpy(completedP->col + nzz_pre_local, Plocal->col, Plocal->nnz * sizeof(itype), cudaMemcpyDeviceToDevice);  );
#endif

  // if (myid == 0) printf("line 837\n");
  // MPI_Barrier(MPI_COMM_WORLD);

  csrlocinfo Plocalinfo;
#if !defined(CSRSEG)
  Plocalinfo.fr=mypfirstrow;
  Plocalinfo.lr=myplastrow;
  Plocalinfo.row=completedP->row;
  Plocalinfo.col=NULL;
  Plocalinfo.val=completedP->val;
#else
  Plocalinfo.fr=nzz_pre_local;
  Plocalinfo.lr=nzz_pre_local + Plocal->nnz;
  Plocalinfo.row=Plocal->row;
  Plocalinfo.col=completedP->col + nzz_pre_local;
  Plocalinfo.val=Plocal->val;
#endif

  // if (myid == 0) printf("line 855\n");
  // MPI_Barrier(MPI_COMM_WORLD);

  CSR* Alocal_ = get_shrinked_matrix(Alocal, Plocal);
  
  cudaDeviceSynchronize();
  if (Alocal_->m != completedP->n) {
      fprintf(stderr, "[%d] Alocal_->m = %d != %d = completedP->n (totcell = %d, Plocal->n = %d, countall = %d, rows2berecived = %d)\n", myid, Alocal_->m, completedP->n, totcell, Plocal->n, countall, Alocal->rows_to_get->rows2bereceived );
  }
  assert( Alocal_->m == completedP->n );
#if defined(ADDCOLSHIFT)  
  CSRm::shift_cols(completedP, completedP->row_shift);
#endif  
#if defined(USESHRINKEDMATRIX)    
  CSR* Plocal_ = get_shrinked_matrix(completedP,NULL);
#endif
#if defined(ADDCOLSHIFT)  
  CSR *C = nsparseMGPU_version0(Alocal_, completedP, &Plocalinfo, loc_time);
#endif
#if defined(USESHRINKEDMATRIX)  
  // if (myid == 0) printf("start nsparseMGPU_version0\n");
  // double start;
  // MPI_Barrier(MPI_COMM_WORLD);
  // cudaDeviceSynchronize();
  // if(ISMASTER) start = MPI_Wtime();

  CSR *C = nsparseMGPU_version0(Alocal_, Plocal_, &Plocalinfo, loc_time);

  // cudaDeviceSynchronize();
  // MPI_Barrier(MPI_COMM_WORLD);
  // if(ISMASTER) loc_time = (MPI_Wtime() - start);
#endif
#if defined(COMPUTECOLINDICES)
  cudaError_t err;
  hProw=(itype *)malloc((completedP->n + 1) * sizeof(itype));
  if(hProw==NULL) {
     fprintf(stderr,"Could not get %d bytes for Prow\n",(completedP->n + 1) * sizeof(itype));
     exit(1);
  }
  hPcol=(itype *)malloc(completedP->nnz * sizeof(itype));
  if(hPcol==NULL) {
     fprintf(stderr,"Could not get %d bytes for Pcol\n",completedP->nnz * sizeof(itype));
     exit(1);
  }
  hPnnz=completedP->nnz;
  hPn=completedP->n;
  hPshift=completedP->row_shift;    
  err = cudaMemcpy(hProw, completedP->row, (completedP->n + 1) * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
  err = cudaMemcpy(hPcol, completedP->col, completedP->nnz * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
#endif

//  if(myid) {  
//    CHECK_DEVICE( cudaMemcpy(C->col, Plocal->col, Plocal->nnz * sizeof(itype), cudaMemcpyDeviceToDevice);  );
//  }

  Pcol = NULL;
  Pval = NULL; //memory will free up in AMG
  CSRm::free_rows_to_get(Alocal);
  
  std::free(completedP);
  
  Alocal_->col = NULL;
  Alocal_->row = NULL;
  Alocal_->val = NULL;
  std::free(Alocal_);
  
  return C;
}

//------------------------------------------------------------------------------