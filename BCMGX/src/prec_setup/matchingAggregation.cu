float SUITOR_TIME = 0;
float TOTAL_MUL_TIME = 0;
float TOTAL_MATCH_TIME = 0;
float TOTAL_SETUP_TIME = 0;
float TOTAL_MEM_TIME = 0;
float TOTAL_RELAX_TIME = 0;
float TOTAL_SHIFTED_CSRVEC=0;
float TOTAL_MAKE_P=0;
float TOTAL_TRA_P=0;
float TOTAL_MAKEAHW_TIME=0;
float TOTAL_MATCHINGPAIR_TIME=0;
float TOTAL_OTHER_TIME=0;
int   DETAILED_TIMING=0;
extern char idstring[];
#include "matchingAggregation.h"

#include "utility/memoryPools.cu"
#include "prec_setup/suitor.cu"
#include "prec_setup/matching.cu"
#include "matchingPairAggregation.cu"

#include "utility/cudamacro.h"
int MUL_NUM = 0;
int I = 0;

#include "utility/function_cnt.h"

#define FTCOARSE_INC 100
#define COARSERATIO_THRSLD 1.2

#include "basic_kernel/halo_communication/local_permutation.h"
#include "basic_kernel/custom_cudamalloc/custom_cudamalloc.h"
#include "utility/timing.h"

itype *iPtemp1;
vtype *vPtemp1;
itype *iAtemp1;
vtype *vAtemp1;
itype *idevtemp1;
vtype *vdevtemp1;
itype *idevtemp2;
// --------- TEST ----------
itype * dev_rcvprow_stat;
vtype * completedP_stat_val;
itype * completedP_stat_col;
itype * completedP_stat_row;
// -------- AH glob --------
itype * AH_glob_row;
itype * AH_glob_col;
vtype * AH_glob_val;
// -------------------------
int * buffer_4_getmct;
int sizeof_buffer_4_getmct = 0;
unsigned int * idx_4shrink;
bool alloced_idx = false;
// ------ cuCompactor ------
int * glob_d_BlocksCount;
int * glob_d_BlocksOffset;
// -------------------------

void relaxPrepare(handles *h, int level, CSR *A, hierarchy *hrrch, buildData *amg_data, int force_relax_type=-1){
  PUSH_RANGE(__func__, 5)
    
  int relax_type;

  if(force_relax_type != -1)
    relax_type = force_relax_type;
  else
    relax_type = amg_data->CRrelax_type;

  if(relax_type == 0){
    // jacobi
    if(hrrch->D_array[level] != NULL)
      Vector::free(hrrch->D_array[level]);
    hrrch->D_array[level] = CSRm::diag(A);

  }else if(relax_type == 4){
    // L1 smoother
    if(hrrch->D_array[level] != NULL)
      Vector::free(hrrch->D_array[level]);
    hrrch->D_array[level] = CSRm::diag(A);

    if(hrrch->M_array[level] != NULL)
      Vector::free(hrrch->M_array[level]);
    hrrch->M_array[level] = CSRm::absoluteRowSum(A, NULL);
  }
  
  POP_RANGE
}

vector<itype>* makePCol_CPU(vector<itype> *mask, itype *ncolc){

  vector<itype> *col = Vector::init<itype>(mask->n, true, false);

  for(itype v=0; v<mask->n; v++){
    itype u = mask->val[v];
    if((u>=0) && (v != u) && (v < u)){
      col->val[v] = ncolc[0];
      col->val[u] = ncolc[0];
      ncolc[0]++;
    }
  }

  for(itype v=0; v<mask->n; v++){
    if(mask->val[v] == -2){
	     col->val[v] = ncolc[0]-1;
	  }else if (mask->val[v] == -1){
	     col->val[v] = ncolc[0];
	     ncolc[0]++;
	   }
  }
  return col;
}


__global__
void __setPsRow4prod(itype n, itype *row, itype nnz, itype start, itype stop){

  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if ( i >= n ) {
     return;
  }

  if(i < start){
    row[i] = 0;
  }

  if(i > stop){
    row[i] = nnz;
  }
}

CSR* matchingAggregation(handles *h, buildData *amg_data, CSR *A, vector<vtype> **w, CSR **P, CSR **R, int level){
  PUSH_RANGE(__func__, 5)
    
  _MPI_ENV;
  TIMER_DEF;
  static int cnt=0;
  CSR *Ai_ = A, *Ai = NULL;

  CSR *Ri_ = NULL;
  vector<vtype> *wi_ = *w, *wi = NULL;

  double size_coarse, size_precoarse;
  double coarse_ratio;

  for(int i=0; i<amg_data->sweepnumber; i++){
    CSR *Pi_;
    if (0 && myid==0) fprintf(stderr,"Task %d reached line %d \n",myid,__LINE__);

    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TIME::start();
      TIMER_START;
    }
    matchingPairAggregation(h, Ai_, wi_, &Pi_, &Ri_, (i==0)); /* routine with the real work. It calls the suitor procedure */
    if (0 && myid==0) fprintf(stderr,"Task %d reached line %d \n",myid,__LINE__);
    char MName[256];
    sprintf(MName,"Pi%d_%s",cnt,idstring);
    CSRm::printMM(Pi_,MName);
    sprintf(MName,"Ri%d_%s",cnt,idstring);
    
    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TOTAL_MATCHINGPAIR_TIME += TIME::stop();
      TIMER_STOP;
      TOTAL_MATCHINGPAIR_TIME += TIMER_ELAPSED;
    }

    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TIME::start();
      TIMER_START;
    }
    char AiName[256], APName[256];
    sprintf(APName,"AP%d_%s",cnt,idstring);
    sprintf(AiName,"Ai%d_%s",cnt++,idstring);
    // --------------- PICO ------------------
    CSR *AP;
    AP  = nsparseMGPU_commu_new(h, Ai_, Pi_, false);
    CSRm::shift_cols(Ri_, -AP->row_shift);
    Ri_->col_shifted=-AP->row_shift;  
    Ai = nsparseMGPU_noCommu_new(h, Ri_, AP);  //, (i+1 < amg_data->sweepnumber ? false : true));
    if(myid!=0 && Ai->col_shifted==0) {
     CSRm::shift_cols(Ai, -(Ai->row_shift) );
     Ai->col_shifted=-(Ai->row_shift);
    }
    if(myid!=0 && AP->col_shifted==0) { /* This is only for debugging */
     CSRm::shift_cols(AP, -(AP->row_shift) );
     AP->col_shifted=-(AP->row_shift);
    }
    CSRm::printMM(Ri_,MName);
    CSRm::printMM(Ai,AiName);
    CSRm::printMM(AP,APName);
	       
    if (0 && myid==0) fprintf(stderr,"Task %d reached line %d \n",myid,__LINE__);
    // ---------------------------------------
        
    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TOTAL_MUL_TIME += TIME::stop();
      TIMER_STOP;
      TOTAL_MUL_TIME += TIMER_ELAPSED;
      MUL_NUM += 2;
    }

    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TIME::start();
      TIMER_START;
    }
    CSRm::free(AP);
    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TOTAL_OTHER_TIME += TIME::stop();
      TIMER_STOP;
      TOTAL_OTHER_TIME += TIMER_ELAPSED;
    }
    // ------------- custom cudaMalloc -------------
    // Vectorinit_CNT
    // wi = Vector::init<vtype>(Ai->n, true, true);
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    wi = Vector::init<vtype>(Ai->n, false, true);
    wi->val = CustomCudaMalloc::alloc_vtype(Ai->n, 1);
    // ---------------------------------------------

    if(DETAILED_TIMING && ISMASTER){
         cudaDeviceSynchronize();
//          TIME::start();
         TIMER_START;
    }
    CSRm::shifted_CSRVector_product_adaptive_miniwarp2(Ri_, wi_, wi, 0 /* Ai_->row_shift */);
    if(DETAILED_TIMING && ISMASTER){
        cudaDeviceSynchronize();
//         TOTAL_SHIFTED_CSRVEC += TIME::stop();
        TIMER_STOP;
        TOTAL_SHIFTED_CSRVEC += TIMER_ELAPSED;
    }

    size_precoarse = Ai_->full_n;
    size_coarse = Ai->full_n;
    coarse_ratio = size_precoarse / size_coarse;
    
    if (coarse_ratio <= COARSERATIO_THRSLD){
      amg_data->ftcoarse = FTCOARSE_INC;
    }
    
    bool brk_flag = (i+1 >= amg_data->sweepnumber) || (size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize);

    if(i == 0){
      *P = Pi_;
    }else{

      if(DETAILED_TIMING && ISMASTER){
         cudaDeviceSynchronize();
//          TIME::start();
         TIMER_START;
      }

      CSRm::shift_cols(*P, -(Pi_->row_shift) );      
      (*P)->m = (unsigned long)Pi_->n;
      csrlocinfo Pinfo1p; 
      Pinfo1p.fr=0;
      Pinfo1p.lr=Pi_->n;
      Pinfo1p.row=Pi_->row;
      Pinfo1p.col=NULL;
      Pinfo1p.val=Pi_->val;

      CSR *tmpP = *P; 
      *P = nsparseMGPU(*P, Pi_, &Pinfo1p, brk_flag);
      CSRm::free(tmpP);
      if(DETAILED_TIMING && ISMASTER){
        cudaDeviceSynchronize();
//         TOTAL_MUL_TIME += TIME::stop();
        TIMER_STOP;
        TOTAL_MUL_TIME += TIMER_ELAPSED;
        MUL_NUM += 1;
      }

      if(DETAILED_TIMING && ISMASTER){
     	cudaDeviceSynchronize();
//         TIME::start();
        TIMER_START;
      }
      //Ri_->row -= Ri_->row_shift;
      CSRm::free(Ri_);
      Ri_ = NULL;
      CSRm::free(Pi_);
      CSRm::free(Ai_);
      if(DETAILED_TIMING && ISMASTER){
         cudaDeviceSynchronize();
//          TOTAL_OTHER_TIME += TIME::stop();
         TIMER_STOP;
         TOTAL_OTHER_TIME += TIMER_ELAPSED;
      }
    }
    // ------------- custom cudaMalloc -------------
    //Vector::free(wi_);
    std::free(wi_);
    // ---------------------------------------------

    if(size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize){
      break;
    }

    Ai_ = Ai;
    wi_ = wi;
    if(myid!=0 && Ai_->col_shifted==0) {
    	CSRm::shift_cols(Ai_, -(Ai_->row_shift) );
        Ai_->col_shifted=-(Ai_->row_shift);
    }
  }

  *w = wi;
  if (0 && myid==0) fprintf(stderr,"Task %d reached line %d \n",myid,__LINE__);
  if(Ri_ == NULL){

    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TIME::start();
      TIMER_START;
    }
    //*R = CSRm::T(h->cusparse_h0, *P);
    if(nprocs > 1){
      //itype ms[nprocs];
      gstype  m_shifts[nprocs];
      // send columns numbers to each process
      m_shifts[myid]=Ai->row_shift;
      CSRm::shift_cols(*P, -m_shifts[myid]);
      
      gstype swp_m = (*P)->m;
      if (myid == nprocs-1){
          (*P)->m = Ai->n;
      }else{
          (*P)->m = Ai->n /* m_shifts[myid+1]-m_shifts[myid] */;
      }

      *R = CSRm::T_multiproc(h->cusparse_h0, *P, Ai->n, true);
    
      (*P)->m = swp_m;
      CSRm::shift_cols(*P, m_shifts[myid]);
      
      //(*R)->row += m_shifts[myid];
      //(*R)->n = Ai->n;
      (*R)->m = (*P)->full_n;
      (*R)->full_n = (*P)->m;
      CSRm::shift_cols(*R, (*P)->row_shift);
      (*R)->row_shift = m_shifts[myid];
    }else{
      *R = CSRm::T(h->cusparse_h0, *P);
    }
    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TOTAL_TRA_P += TIME::stop();
      TIMER_STOP;
      TOTAL_TRA_P += TIMER_ELAPSED;
    }
  }else{
    *R = Ri_;
  }
  
  POP_RANGE
  
  if(myid!=0 && Ai->col_shifted==0) {
     CSRm::shift_cols(Ai, -(Ai->row_shift) );
     Ai->col_shifted=-(Ai->row_shift);
  }

  return Ai;
}


hierarchy* adaptiveCoarsening(handles *h, buildData *amg_data, const params p, bool precondition_flag){
  PUSH_RANGE(__func__,4)
    
  _MPI_ENV;
  TIMER_DEF;
  
  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TIME::start();
      TIMER_START;
  }
  CSR *A = amg_data->A;

  // init memory pool
  MemoryPool::initContext(A->full_n, A->n);
  iPtemp1 = NULL;
  vPtemp1 = NULL;
 
  MY_CUDA_CHECK( cudaMallocHost(&iAtemp1, sizeof (itype) * p.mem_alloc_size ) );
  MY_CUDA_CHECK( cudaMallocHost(&vAtemp1, sizeof (vtype) * p.mem_alloc_size ) );
  cudaMalloc_CNT
  MY_CUDA_CHECK( cudaMalloc(&idevtemp1, sizeof (itype) * p.mem_alloc_size ) ); 
  cudaMalloc_CNT
  MY_CUDA_CHECK( cudaMalloc(&vdevtemp1, sizeof (vtype) * p.mem_alloc_size ) ); 
  cudaMalloc_CNT
  MY_CUDA_CHECK( cudaMalloc(&idevtemp2, sizeof (itype) * p.mem_alloc_size ) );
  // -------- AH glob --------
  cudaMalloc_CNT
  MY_CUDA_CHECK( cudaMalloc(&AH_glob_row, sizeof (itype) * (A->n +1) ) ); 
  cudaMalloc_CNT
  MY_CUDA_CHECK( cudaMalloc(&AH_glob_col, sizeof (itype) * A->nnz ) );
  cudaMalloc_CNT
  MY_CUDA_CHECK( cudaMalloc(&AH_glob_val, sizeof (vtype) * A->nnz ) );;
  // -------------------------
  
  vector<vtype> *w = amg_data->w;
  //vector<vtype> *w_temp = Vector::clone(w);
  vector<vtype> *w_temp;

  // -----  CustomCudaMalloc ---- //
  if( w->on_the_device ) {
    w_temp = Vector::init<vtype>(w->n, false, true);
    w_temp->val = CustomCudaMalloc::alloc_vtype(w->n, 1);
    cudaError_t err;
    err = cudaMemcpy(w_temp->val, w->val, w_temp->n * sizeof(vtype), cudaMemcpyDeviceToDevice);
    CHECK_DEVICE(err);
  }else{
    w_temp = Vector::clone(w);
  }    
  // ----------------------------//

  CSR *P = NULL, *R = NULL;
  hierarchy *hrrch = AMG::Hierarchy::init(amg_data->maxlevels + 1);
  hrrch->A_array[0] = A;
  
  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TOTAL_MEM_TIME += TIME::stop();
      TIMER_STOP;
      TOTAL_MEM_TIME += TIMER_ELAPSED;
  }

  // compute comunication patterns for solver
  if(nprocs > 1){
    if(DETAILED_TIMING && ISMASTER){
//       TIME::start();
        TIMER_START;
    }
    halo_info hi = haloSetup(hrrch->A_array[0], NULL);
//    printf("Task %d, halo done for %x level=%d\n",myid,hrrch->A_array[0],0);
    
    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TOTAL_SETUP_TIME += TIME::stop();
      TIMER_STOP;
      TOTAL_SETUP_TIME += TIMER_ELAPSED;
    }
    hrrch->A_array[0]->halo = hi;
  }

  vtype avcoarseratio = 0.;
  int level = 0;
  if (precondition_flag) {
    if(DETAILED_TIMING && ISMASTER){
  //      TIME::start();
        TIMER_START;
    }
    relaxPrepare(h, level, hrrch->A_array[level], hrrch, amg_data);
    if(DETAILED_TIMING && ISMASTER){
        cudaDeviceSynchronize();
  //       TOTAL_RELAX_TIME += TIME::stop();
        TIMER_STOP;
        TOTAL_RELAX_TIME += TIMER_ELAPSED;
    }
  }

  amg_data->ftcoarse = 1;

  if(precondition_flag){
    for(level=1; level < amg_data->maxlevels;){
      if(0 && myid==0) fprintf(stderr,"Task %d entering level %d\n",myid,level);
      if(DETAILED_TIMING && ISMASTER){
//           TIME::start();
          TIMER_START;
      }
      hrrch->A_array[level] = matchingAggregation(h, amg_data, hrrch->A_array[level-1], &w_temp, &P, &R, level-1);
      if(0 && myid==0) fprintf(stderr,"Task %d out of matchingAggregation\n",myid);
      if(DETAILED_TIMING && ISMASTER){
        cudaDeviceSynchronize();
//         TOTAL_MATCH_TIME += TIME::stop();
        TIMER_STOP;
        TOTAL_MATCH_TIME += TIMER_ELAPSED;
      }
 
      if(nprocs > 1){
        if(DETAILED_TIMING && ISMASTER){
//           TIME::start();
            TIMER_START;
        }
	if(myid!=0 && hrrch->A_array[level]->col_shifted==0) {
  	      CSRm::shift_cols(hrrch->A_array[level], -(hrrch->A_array[level]->row_shift) );
	      hrrch->A_array[level]->col_shifted=-(hrrch->A_array[level]->row_shift);
	}
        halo_info hi = haloSetup(hrrch->A_array[level], NULL);
//        printf("Task %d, halo done for %x, level=%d\n",myid,hrrch->A_array[level],level);
        if(0 && myid==0) fprintf(stderr,"Task %d out of haloSetup\n",myid);
        if(DETAILED_TIMING && ISMASTER){
            cudaDeviceSynchronize();
//             TOTAL_SETUP_TIME += TIME::stop();
            TIMER_STOP;
            TOTAL_SETUP_TIME += TIMER_ELAPSED;
        }
        hrrch->A_array[level]->halo = hi;
      }

      if(!amg_data->agg_interp_type){
       if(DETAILED_TIMING && ISMASTER){
//           TIME::start();
           TIMER_START;
       }
       relaxPrepare(h, level, hrrch->A_array[level], hrrch, amg_data);
       if(0 && myid==0)  fprintf(stderr,"Task %d out of relaxPrepare\n",myid);
       if(DETAILED_TIMING && ISMASTER){
          cudaDeviceSynchronize();
//           TOTAL_RELAX_TIME += TIME::stop();
          TIMER_STOP;
          TOTAL_RELAX_TIME += TIMER_ELAPSED;
       }
      }

      hrrch->P_array[level-1] = P;
      hrrch->R_array[level-1] = R;
      
      // --------------- PICO ------------------
      bool shrink_col(CSR*, CSR*);
//      printf("Task %d, shrinking col of matrix %x, level %d\n",myid,hrrch->A_array[level-1],level-1);    
      shrink_col(hrrch->A_array[level-1], NULL);
      if(myid!=0 && hrrch->P_array[level-1]->col_shifted==0) {
           CSRm::shift_cols(hrrch->P_array[level-1], -(hrrch->A_array[level]->row_shift));
	   hrrch->P_array[level-1]->col_shifted=-(hrrch->A_array[level]->row_shift);
      }
      shrink_col(hrrch->P_array[level-1], hrrch->A_array[level]);
      
      if (level != hrrch->num_levels-1) {
        if(myid!=0 && hrrch->R_array[level-1]->col_shifted==0) {
	   hrrch->R_array[level-1]->bitcol=NULL;
	   hrrch->R_array[level-1]->bitcolsize=0; 
           CSRm::shift_cols(hrrch->R_array[level-1], -(hrrch->A_array[level-1]->row_shift));
     	   hrrch->R_array[level-1]->col_shifted=-(hrrch->A_array[level-1]->row_shift);
        }

        shrink_col(hrrch->R_array[level-1], hrrch->A_array[level-1]);
      } 
      // ---------------------------------------

      if(nprocs > 1){

        if(DETAILED_TIMING && ISMASTER){
//             TIME::start();
            TIMER_START;
        }
        halo_info hi = haloSetup(hrrch->A_array[level], hrrch->P_array[level-1]);
        if(0 && myid==0) fprintf(stderr,"Task %d out of haloSetup 2\n",myid);
	
        if(DETAILED_TIMING && ISMASTER){
            cudaDeviceSynchronize();
//             TOTAL_SETUP_TIME += TIME::stop();
            TIMER_STOP;
            TOTAL_SETUP_TIME += TIMER_ELAPSED;
        }
        hrrch->P_array[level-1]->halo = hi;
        
      }
      
      if(nprocs > 1 && (level != hrrch->num_levels-1)) {
        
        if(DETAILED_TIMING && ISMASTER){
//             TIME::start();
            TIMER_START;
        }
        halo_info hi = haloSetup(hrrch->A_array[level-1], hrrch->R_array[level-1]); // BUG: haloSetup(hrrch->R_array[level-1], hrrch->A_array[level-1]);
        if(0 && myid==0) fprintf(stderr,"Task %d out of haloSetup 3\n",myid);
        if(DETAILED_TIMING && ISMASTER){
            cudaDeviceSynchronize();
//             TOTAL_SETUP_TIME += TIME::stop();
            TIMER_STOP;
            TOTAL_SETUP_TIME += TIMER_ELAPSED;
        }
        hrrch->R_array[level-1]->halo = hi;
        
      }
      
      vtype size_coarse = hrrch->A_array[level]->full_n;

      vtype coarse_ratio = hrrch->A_array[level-1]->full_n / size_coarse;
      avcoarseratio = avcoarseratio + coarse_ratio;
      level++;

      if(size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize){
        break;
      }
      if(0 && myid==0) fprintf(stderr,"Task %d end level %d\n",myid,level);
    }
//    printf("Task %d, shrinking col of matrix %x, level %d\n",myid,hrrch->A_array[level-1],level-1);    
//#if LOCAL_COARSEST==0
    if( p.coarsesolver_type == 0){
      shrink_col(hrrch->A_array[level-1], NULL);
    }
//#endif
  } else {
    bool shrink_col(CSR*, CSR*);
    shrink_col(hrrch->A_array[level], NULL);
  }
  if(0 && myid==0) fprintf(stderr,"Task %d end loop on levels\n",myid);

  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TIME::start();
      TIMER_START;
  }

// ### Start Free ASYNC
//  MY_CUDA_CHECK( cudaFreeAsync(idevtemp1, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(idevtemp2, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(vdevtemp1, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(dev_rcvprow_stat, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(completedP_stat_val, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(completedP_stat_col, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(completedP_stat_row, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(AH_glob_row, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(AH_glob_col, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(AH_glob_val, h->stream_free));
//  MY_CUDA_CHECK( cudaFreeAsync(buffer_4_getmct, h->stream_free));
//  if (alloced_idx == true) {
//    MY_CUDA_CHECK( cudaFreeAsync(idx_4shrink, h->stream_free));
//  }
// ### End Free ASYNC  


  if (precondition_flag) {
    AMG::Hierarchy::finalize_level(hrrch, level);
    AMG::Hierarchy::finalize_cmplx(hrrch);
    AMG::Hierarchy::finalize_wcmplx(hrrch);
    hrrch->avg_cratio = avcoarseratio / (level-1);

    if(ISMASTER){
      AMG::Hierarchy::printInfo(hrrch);
      Eval::printMetaData("agg;level_number", level, 0);
      Eval::printMetaData("agg;avg_coarse_ratio", hrrch->avg_cratio, 1);
      Eval::printMetaData("agg;OpCmplx", hrrch->op_cmplx, 1);
      Eval::printMetaData("agg;OpCmplxW", hrrch->op_wcmplx, 1);
      Eval::printMetaData("agg;coarsest_size", hrrch->A_array[level-1]->full_n, 0);
      Eval::printMetaData("agg;total_mul_num", MUL_NUM, 0);
    }
  }
  MY_CUDA_CHECK( cudaFreeHost(iPtemp1));
  free(vPtemp1);
  MY_CUDA_CHECK( cudaFreeHost(iAtemp1));
  MY_CUDA_CHECK( cudaFreeHost(vAtemp1));
  MY_CUDA_CHECK( cudaFree(idevtemp1));
  MY_CUDA_CHECK( cudaFree(idevtemp2));
  MY_CUDA_CHECK( cudaFree(vdevtemp1));
 // ----------------- TEST --------------------
  MY_CUDA_CHECK( cudaFree(dev_rcvprow_stat));
  MY_CUDA_CHECK( cudaFree(completedP_stat_val));
  MY_CUDA_CHECK( cudaFree(completedP_stat_col));
  MY_CUDA_CHECK( cudaFree(completedP_stat_row));
 // --------------- AH glob -------------------
  MY_CUDA_CHECK( cudaFree(AH_glob_row));
  MY_CUDA_CHECK( cudaFree(AH_glob_col));
  MY_CUDA_CHECK( cudaFree(AH_glob_val));
 // -------------------------------------------
  MY_CUDA_CHECK( cudaFree(buffer_4_getmct));
  if (alloced_idx == true) {
    MY_CUDA_CHECK( cudaFree(idx_4shrink));
  }
  // ------------ cuCompactor ------------------
  MY_CUDA_CHECK( cudaFree(glob_d_BlocksCount));
  MY_CUDA_CHECK( cudaFree(glob_d_BlocksOffset));
  // -------------------------------------------
  
  // ------------- custom cudaMalloc -------------
//   Vector::free(w_temp);
  std::free(w_temp);
  // ---------------------------------------------

  MemoryPool::freeContext();
  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
//       TOTAL_MEM_TIME += TIME::stop();
      TIMER_STOP;
      TOTAL_MEM_TIME += TIMER_ELAPSED;
  }
  if(DETAILED_TIMING && ISMASTER){
    Eval::printMetaData("agg;SUITOR_TIME", SUITOR_TIME / 1000.0, 1);
    Eval::printMetaData("agg;total_mul_time", TOTAL_MUL_TIME / 1000.0, 1);
    Eval::printMetaData("agg;total_setup_time", TOTAL_SETUP_TIME / 1000.0, 1);
    Eval::printMetaData("agg;total_mem_time", TOTAL_MEM_TIME / 1000.0, 1);
    Eval::printMetaData("agg;total_relax_time", TOTAL_RELAX_TIME / 1000.0, 1);
    Eval::printMetaData("agg;total_shifted_csrvec", TOTAL_SHIFTED_CSRVEC/ 1000.0, 1);
    Eval::printMetaData("agg;total_make_p", TOTAL_MAKE_P/ 1000.0, 1);
    Eval::printMetaData("agg;total_traspose_p", TOTAL_TRA_P/ 1000.0, 1);
    Eval::printMetaData("agg;total_makeAH_W", TOTAL_MAKEAHW_TIME/ 1000.0, 1);
    Eval::printMetaData("agg;total_matchingPairAggregation", TOTAL_MATCHINGPAIR_TIME/ 1000.0, 1);
    Eval::printMetaData("agg;total_matchingAggregation", TOTAL_MATCH_TIME/ 1000.0, 1);
    Eval::printMetaData("agg;total_OtherTime", TOTAL_OTHER_TIME/ 1000.0, 1);
    
  }

  POP_RANGE
  return hrrch;
}

