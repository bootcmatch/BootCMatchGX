#include "solver.h"

halo_info *H_halo_info;
halo_info *H_R_halo_info;
halo_info *H_P_halo_info;
overlappedSmootherList *osl;

#include "matrix/distribuite.cu"
#include "solver/relaxation.cu"
#include "solver/relaxation_sm.cu"

void relax(handles *h, int k, int level, hierarchy *hrrch, vector<vtype> *f, int relax_type, vtype relax_weight, vector<vtype> *u, vector<vtype> **u_, bool forward=true){
  _MPI_ENV;

  if(nprocs == 1){
    relaxCoarsest(h, k, level, hrrch, f, relax_type, relax_weight, u, u_, 0);
    return;
  }

  if(relax_type == 0){
    SMOOTHER(h->cusparse_h0, h->cublas_h, k, hrrch->A_array[level], u, u_, f, hrrch->D_array[level], relax_weight, level);
  }else if(relax_type == 4){
    SMOOTHER(h->cusparse_h0, h->cublas_h, k, hrrch->A_array[level], u, u_, f, hrrch->M_array[level], relax_weight, level);
  }
}

#include "solver/GAMG_cycle.cu"
#include "solver/FCG.cu"
#include "solver/bootstrap.cu"


vector<vtype>* solve(CSR *Alocal, vector<vtype> *rhs, const params p){
  _MPI_ENV;

  handles *h = Handles::init();

  if(ISMASTER)
    std::cout << "\n\nBUILDING....:\n\n";

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

  Relax::initContext(Alocal->full_n);
  GAMGcycle::initContext(Alocal->n);

  float boot_time;

  // start bootstrap process
  CHECK_MPI(
		  MPI_Barrier(MPI_COMM_WORLD)
	   );
  if(ISMASTER)
    TIME::start();

  boot *boot_amg = Bootstrap::bootstrap(h, bootamg_data, amg_cycle, p);

  if(ISMASTER){
    boot_time = TIME::stop();
  }
  fprintf(stderr,"Task %d: Bootstrap done\n",myid);

hierarchy *H = boot_amg->H_array[0];
vector<vtype> *Sol = NULL;

int precon = 1;
int num_iter = 0;

Sol = Vector::init<vtype>(Alocal->full_n, true, true);
Vector::fillWithValue(Sol, 0.);

if(ISMASTER)
  std::cout << "\n\nSOLVING....:\n\n";
CHECK_MPI(
	  MPI_Barrier(MPI_COMM_WORLD)
   );

if(ISMASTER)
  TIME::start();

itype levelc = H->num_levels - 1;

if(nprocs > 1){

#if LOCAL_COARSEST == 1

    CSR * h_Ac_local = CSRm::copyToHost(H->A_array[levelc]);
    CSR *h_Ac = join_MatrixMPI_all(h_Ac_local);

    CSRm::free(h_Ac_local);
    CSR *Ac = CSRm::copyToDevice(h_Ac);
    CSRm::free(h_Ac);
    CSRm::free(H->A_array[levelc]);
    H->A_array[levelc] = Ac;

    // todo wasted computation
    relaxPrepare(h, levelc, H->A_array[levelc], H, amg_data, amg_data->coarse_solver);

    h_Ac_local = CSRm::copyToHost(H->R_array[levelc-1]);
    h_Ac = join_MatrixMPI_all(h_Ac_local);
    CSRm::free(h_Ac_local);
    H->R_array[levelc-1] = CSRm::copyToDevice(h_Ac);
    CSRm::free(h_Ac);
#endif

    for(int i=0; i<H->num_levels-1; i++){
      H->R_local_array[i] = H->R_array[i];
      H->P_local_array[i] = H->P_array[i];
    }

  itype hn;
#if LOCAL_COARSEST == 1
    hn = H->num_levels-1;
#else
    hn = H->num_levels;
#endif

  // pre-computation required for overlapped smoother (matrix/vector product) this can be performed during the building phase by using a differen stream
#ifdef OVERLAPPED_SMO
  osl = init_overlappedSmootherList(hn);
  for(int i=0; i<hn; i++){
    setupOverlappedSmoother(H->A_array[i], &osl->oss[i]);
  }
#endif

  }

  float setup_time_add;
  if(ISMASTER){
    setup_time_add = TIME::stop();
  }

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
    &num_iter);

  if(ISMASTER){
    cudaDeviceSynchronize();
    float solve_time = TIME::stop();
    Eval::printMetaData("agg;hierarchy_levels_num", boot_amg->n_hrc, 0);

    Eval::printMetaData("agg;final_estimated_ratio", boot_amg->estimated_ratio, 1);
    Eval::printMetaData("sol;num_iteration", num_iter, 0);
    Eval::printMetaData("sol;residual", residual, 1);
    Eval::printMetaData("time;solve_time", solve_time, 1);
    std::cout << "solve_time: " << (solve_time / 1000) << " sec \n";

    float time_for_iteration = solve_time / num_iter;
    std::cout << "time_for_iteration: " << (time_for_iteration / 1000) << " sec \n";
    Eval::printMetaData("time;bootstrap_time", (boot_time)/1000, 1);
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

  return Sol;
}
