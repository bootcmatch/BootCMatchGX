#include "solver/bcmgx.h"

#include <cuda_profiler_api.h>

#include "utility/distribuite.h"

#if !defined(TIMING_H)
#define TIMING_H
#include <sys/time.h>

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+((temp_2.tv_usec-temp_1 .tv_usec)/(1.e6)))
#endif

#include "utility/function_cnt.h"
#include "basic_kernel/custom_cudamalloc/custom_cudamalloc.h"

#include "prec_apply/GAMG_cycle.h"
#include "prec_setup/bootstrap.cu"

// ---------------------------------
// #include "solver/FCG.cu"
#include "solver/FCG.h"
extern float TOTAL_CSRVECTOR_TIME;
extern float TOTAL_NORMMPI_TIME;
extern float TOTAL_PRECONDAPPLY_TIME;
extern float TOTAL_AXPY2_TIME;
extern float TOTAL_TRIPLEPROD_TIME;
extern float TOTAL_DOUBLEMERGED_TIME;
extern float TOTAL_NORMPI2_TIME;
extern float TOTAL_RESTPRE_TIME;
// ---------------------------------

#include "basic_kernel/smoother/relax.h"
overlappedSmootherList *osl;

vector<vtype>* bcmgx(CSR *Alocal, vector<vtype> *rhs, const params p, bool precondition_flag){
  PUSH_RANGE(__func__,2)
    
  _MPI_ENV;
  TIMER_DEF;
  handles *h = Handles::init();
  
// ------------------ custom_cudamalloc --------------------
  // Parametri di inizializazione con unico blocco: [ 9, 5 ]
  CustomCudaMalloc::init((Alocal->nnz)*5, (Alocal->nnz)*3);
  //CustomCudaMalloc::init((Alocal->nnz)*5, (Alocal->nnz)*3);

  //CustomCudaMalloc::init((Alocal->nnz)*2, (Alocal->nnz)*2, 1);
  CustomCudaMalloc::init((Alocal->nnz)*1, (Alocal->nnz)*1, 1);
  
  CustomCudaMalloc::init((Alocal->nnz)*3, (Alocal->nnz)*3, 2);
  //CustomCudaMalloc::init((Alocal->nnz)*3, (Alocal->nnz)*2, 2);
// ---------------------------------------------------------

  if(ISMASTER)
    std::cout << "\n\nBUILDING....:\n\n";

  if( myid == 0){
      cudaProfilerStart();
  }

  bootBuildData *bootamg_data;
  bootamg_data = AMG::BootBuildData::initByParams(Alocal, p);

  buildData *amg_data;
  amg_data = bootamg_data->amg_data;

  applyData *amg_cycle;
  amg_cycle = AMG::ApplyData::initByParams(p);
  AMG::ApplyData::setGridSweeps(amg_cycle, amg_data->maxlevels);

  if(VERBOSE > 0){
    AMG::BootBuildData::print(bootamg_data);
    AMG::BuildData::print(amg_data);
    AMG::ApplyData::print(amg_cycle);
  }

  Relax::initContext(Alocal->n);
  GAMGcycle::initContext(Alocal->n);

  float boot_time;

  // start bootstrap process
  CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
  if(ISMASTER)
    TIME::start();
  TIMER_START;

  boot *boot_amg = Bootstrap::bootstrap(h, bootamg_data, amg_cycle, p, precondition_flag);
  CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
  
  if(ISMASTER){
//    cudaDeviceSynchronize();
    boot_time = TIME::stop();
    TIMER_STOP;
    fprintf(stdout,"New bootstrap timer: %f in seconds\n",TIMER_ELAPSED);
  }
//  fprintf(stderr,"Task %d: Bootstrap done\n",myid);
  if( myid == 0){
      cudaProfilerStop();
  }

  hierarchy *H = boot_amg->H_array[0];
  vector<vtype> *Sol = NULL;

  int precon = 1;
  int num_iter = 0;

  Vectorinit_CNT
  Sol = Vector::init<vtype>(Alocal->n, true, true);
  Vector::fillWithValue(Sol, 0.);

 // ------------------ custom_cudamalloc --------------------
    CustomCudaMalloc::free(1);
    CustomCudaMalloc::free(2);
 // ---------------------------------------------------------

  CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );

  float setup_time_add;
  if (precondition_flag) {

    if(ISMASTER)
      TIME::start();
      TIMER_START;

    itype levelc = H->num_levels - 1;

    if(nprocs > 1){

    //#if LOCAL_COARSEST == 1
    if (p.coarsesolver_type == 1){
	  if(H->A_array[levelc]->col_shifted) {
           CSRm::shift_cols(H->A_array[levelc], -(H->A_array[levelc]->col_shifted));
	  }
        CSR * h_Ac_local = CSRm::copyToHost(H->A_array[levelc]);
        CSR *h_Ac = join_MatrixMPI_all(h_Ac_local);

        CSRm::free(h_Ac_local);
        CSR *Ac = CSRm::copyToDevice(h_Ac);
        CSRm::free(h_Ac);
        CSRm::free(H->A_array[levelc]);
        H->A_array[levelc] = Ac;

        // todo wasted computation
        relaxPrepare(h, levelc, H->A_array[levelc], H, amg_data, amg_data->coarse_solver);
	  if(H->R_array[levelc-1]->col_shifted) {
           CSRm::shift_cols(H->R_array[levelc-1], -(H->R_array[levelc-1]->col_shifted));
	  }
        h_Ac_local = CSRm::copyToHost(H->R_array[levelc-1]);
        h_Ac = join_MatrixMPI_all(h_Ac_local);
        CSRm::free(h_Ac_local);
        H->R_array[levelc-1] = CSRm::copyToDevice(h_Ac);
        CSRm::free(h_Ac);

	  if(H->P_array[levelc-1]->col_shifted) {
           CSRm::shift_cols(H->P_array[levelc-1], -(H->P_array[levelc-1]->col_shifted));
	  }
    //#endif
    }

        for(int i=0; i<H->num_levels-1; i++){
          H->R_local_array[i] = H->R_array[i];
          H->P_local_array[i] = H->P_array[i];
        }

      itype hn;
    //#if LOCAL_COARSEST == 1
    if ( p.coarsesolver_type == 1){
        hn = H->num_levels-1;
    }else{
    //#else
        hn = H->num_levels;
    //#endif
    }

      // pre-computation required for overlapped smoother (matrix/vector product) this can be performed during the building phase by using a differen stream
    #ifdef OVERLAPPED_SMO
      osl = init_overlappedSmootherList(hn);
      for(int i=0; i<hn; i++){
        setupOverlappedSmoother(H->A_array[i], &osl->oss[i]);
      }
    #endif

      }

      if(ISMASTER){
        setup_time_add = TIME::stop();
      }

  }

  if(ISMASTER)
    std::cout << "\n\nSOLVING....:\n\n";

  if(ISMASTER)
   TIME::start();
  
  vtype residual = 0.;
  residual = flexibileConjugateGradients_v3(
    H->A_array[0],
    h,
    Sol,
    rhs, 
    bootamg_data, 
    boot_amg, 
    amg_cycle, 
    precon, 
    p.itnlim, 
    p.rtol, 
    &num_iter,
    precondition_flag, p.coarsesolver_type);

  CHECK_MPI( MPI_Barrier(MPI_COMM_WORLD) );
  
  
  if(ISMASTER){
    cudaDeviceSynchronize();
    float solve_time = TIME::stop();
    TIMER_STOP;
    printf("New solver timer: %f (in seconds)\n",TIMER_ELAPSED);
    if (precondition_flag) {
      Eval::printMetaData("agg;hierarchy_levels_num", boot_amg->n_hrc, 0);
      Eval::printMetaData("agg;final_estimated_ratio", boot_amg->estimated_ratio, 1);
    }
    Eval::printMetaData("sol;num_iteration", num_iter, 0);
    Eval::printMetaData("sol;residual", residual, 1);
    Eval::printMetaData("time;solve_time", solve_time, 1);
    std::cout << "solve_time: " << (solve_time / 1000) << " sec \n";

    float time_for_iteration = solve_time / num_iter;
    std::cout << "time_for_iteration: " << (time_for_iteration / 1000) << " sec \n";
    Eval::printMetaData("time;bootstrap_time", (boot_time)/1000, 1);
    if (precondition_flag) {
      Eval::printMetaData("time;setup_time", (setup_time_add)/1000, 1);
      if(DETAILED_TIMING) {
        Eval::printMetaData("agg;csrvector_time per iter", TOTAL_CSRVECTOR_TIME/num_iter/1000,1);
        Eval::printMetaData("agg;normpi_time per iter", TOTAL_NORMMPI_TIME/num_iter/1000,1);
        Eval::printMetaData("agg;precondapply_time per iter", TOTAL_PRECONDAPPLY_TIME/num_iter/1000,1);
        Eval::printMetaData("agg;relax_time per iter", TOTAL_SOLRELAX_TIME/num_iter/1000,1);
        Eval::printMetaData("agg;restpre_time per iter", TOTAL_RESTPRE_TIME/num_iter/1000,1);
        Eval::printMetaData("agg;restgamg_time per iter", TOTAL_RESTGAMG_TIME/num_iter/1000,1);
        Eval::printMetaData("agg;axpy2_time per iter", TOTAL_AXPY2_TIME/num_iter/1000,1);
        Eval::printMetaData("agg;tripleprod_time per iter", TOTAL_TRIPLEPROD_TIME/num_iter/1000,1);
        Eval::printMetaData("agg;doublemerged_time per iter", TOTAL_DOUBLEMERGED_TIME/num_iter/1000,1);
        Eval::printMetaData("agg;normpi2_time per iter", TOTAL_NORMPI2_TIME/num_iter/1000,1);
      }
    }
  }

 // ------------------ custom_cudamalloc --------------------
    CustomCudaMalloc::free();  //Freed in sample_main.cu
 // ---------------------------------------------------------


  POP_RANGE
  return Sol;
}
