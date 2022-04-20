#pragma once

namespace Bootstrap{

    void innerIterations(handles *h, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle){

      buildData *amg_data = bootamg_data->amg_data;
      CSR *A = amg_data->A;
      vector<vtype> *w = amg_data->w;
      // current solution: at the end it will point to the new smooth vector */
      vector<vtype> *x = Vector::init<vtype>(A->n, true, true);
      Vector::fillWithValue(x, 1.);

      // rhs
      vector<vtype> *rhs = Vector::init<vtype>(A->n, true, true);
      Vector::fillWithValue(rhs, 0.);

      vtype normold = CSRm::vectorANorm(h->cublas_h, A, x);
      vtype normnew;

      vtype conv_ratio;

      for(int i=1; i<=bootamg_data->solver_it; i++){
        preconditionApply(h, bootamg_data, boot_amg, amg_cycle, rhs, x);
        normnew = CSRm::vectorANorm(h->cublas_h, A, x);
        conv_ratio = normnew / normold;
        normold = normnew;
      }

      std::cout << "\n conv_ratio " << conv_ratio << "\n";

      vtype alpha = 1. / normnew;

      printf("current smooth vector A-norm=%e\n", normnew);

      Vector::scale(h->cublas_h, x, alpha);
      Vector::copyTo(w, x);

      boot_amg->estimated_ratio = conv_ratio;

      Vector::free(x);
      Vector::free(rhs);
  }

#ifndef TIMING_H
#define TIMING_H
#include <sys/time.h>

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+((temp_2.tv_usec-temp_1 .tv_usec)/(1.e6)))
#endif

boot* bootstrap(handles *h, bootBuildData *bootamg_data, applyData *apply_data, const params p){
    _MPI_ENV;
    TIMER_DEF;
    double initfinalize_time=0;
    TIMER_START;
    boot *boot_amg = AMG::Boot::init(bootamg_data->max_hrc, 1.0);
    TIMER_STOP;
    initfinalize_time+=TIMER_ELAPSED;
    buildData *amg_data;
    amg_data = bootamg_data->amg_data;

    int num_hrc = 0;
    while(boot_amg->estimated_ratio > bootamg_data->conv_ratio && num_hrc < bootamg_data->max_hrc){

      TIMER_START;
      boot_amg->H_array[num_hrc] = adaptiveCoarsening(h, amg_data, p); /* this is always done (look at AMG folder) */
      TIMER_STOP;
      if(ISMASTER) {
        printf("adaptiveCoarsening time: %g\n",TIMER_ELAPSED);
      }
      num_hrc++;
      boot_amg->n_hrc = num_hrc;

      if(VERBOSE > 0)
        printf("Built new hierarchy. Current number of hierarchies:%d\n", num_hrc);


      if(num_hrc == 1){
	TIMER_START;
        // init FGC buffers
        FCG::initPreconditionContext(boot_amg->H_array[0]); /* this is always done */
	TIMER_STOP;
	if(ISMASTER) {
        	printf("initPreconditionerContext time: %g\n",TIMER_ELAPSED);
	}
        if(apply_data->cycle_type == 3)
          assert(false);
      }else{
          assert(false);
      }
    }
    TIMER_START;
    AMG::Boot::finalize(boot_amg, num_hrc);
    TIMER_STOP;
    if(ISMASTER) {
        	printf("initfinalize time: %g\n",TIMER_ELAPSED);
    }

    return boot_amg;
  }

}
