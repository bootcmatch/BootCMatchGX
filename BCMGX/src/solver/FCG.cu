#pragma once

#include "FCG.h"
#include "basic_kernel/halo_communication/local_permutation.h"
#include "basic_kernel/halo_communication/extern2.h"
#include "basic_kernel/halo_communication/extern.h"
#include "basic_kernel/matrix/vector.cu"
#include "prec_apply/GAMG_cycle.h"

#include <string.h>
#include <cmath>

#include "utility/utils.h"

float TOTAL_CSRVECTOR_TIME=0.;
float TOTAL_NORMMPI_TIME=0.;
float TOTAL_PRECONDAPPLY_TIME=0.;
float TOTAL_AXPY2_TIME=0.;
float TOTAL_TRIPLEPROD_TIME=0.;
float TOTAL_DOUBLEMERGED_TIME=0.;
float TOTAL_NORMPI2_TIME=0.;
float TOTAL_RESTPRE_TIME=0.;

FCGPreconditionContext FCG::context;

void FCG::initPreconditionContext(hierarchy *hrrch){
    PUSH_RANGE(__func__, 4)

    FCG::context.hrrch = hrrch;
    int num_levels = hrrch->num_levels;

    FCG::context.max_level_nums = num_levels;
    FCG::context.max_coarse_size = (itype*) malloc( num_levels * sizeof(int));
    assert(FCG::context.max_coarse_size != NULL);

    vectorCollection<vtype> *RHS_buffer = Vector::Collection::init<vtype>(num_levels);

    vectorCollection<vtype> *Xtent_buffer_local = Vector::Collection::init<vtype>(num_levels);
    vectorCollection<vtype> *Xtent_buffer_2_local = Vector::Collection::init<vtype>(num_levels);

    // !skip the first
    for(int i=0; i<num_levels; i++){
      itype n_i = hrrch->A_array[i]->n;
      itype n_i_full = hrrch->A_array[i]->full_n;
      FCG::context.max_coarse_size[i] = n_i;
      Vectorinit_CNT
      RHS_buffer->val[i] = Vector::init<vtype>(n_i, true, true);
      Vectorinit_CNT
      Xtent_buffer_local->val[i] = Vector::init<vtype>( (i!=num_levels-1) ? n_i : n_i_full , true, true);
      Vectorinit_CNT
      Xtent_buffer_2_local->val[i] = Vector::init<vtype>( (i!=num_levels-1) ? n_i : n_i_full , true, true);
      Vector::fillWithValue(Xtent_buffer_local->val[i], 0.);
      Vector::fillWithValue(Xtent_buffer_2_local->val[i], 0.);
    }

    FCG::context.RHS_buffer = RHS_buffer;

    FCG::context.Xtent_buffer_local = Xtent_buffer_local;
    FCG::context.Xtent_buffer_2_local = Xtent_buffer_2_local;

    POP_RANGE
}

void FCG::setHrrchBufferSize(hierarchy *hrrch){
    int num_levels = hrrch->num_levels;
    assert(num_levels <= FCG::context.max_level_nums);

    for(int i=0; i<num_levels; i++){
      itype n_i = hrrch->A_array[i]->n;
      itype n_i_full = hrrch->A_array[i]->full_n;

      if(n_i > FCG::context.max_coarse_size[i]){
        // make i-level's buffer bigger

        FCG::context.max_coarse_size[i] = n_i;
        Vector::free(FCG::context.RHS_buffer->val[i]);
        Vectorinit_CNT
        FCG::context.RHS_buffer->val[i] = Vector::init<vtype>(n_i, true, true);


        Vector::free(FCG::context.Xtent_buffer_local->val[i]);
        Vector::free(FCG::context.Xtent_buffer_2_local->val[i]);

        if (i == num_levels-1) {
            Vectorinit_CNT
            FCG::context.Xtent_buffer_local->val[i] = Vector::init<vtype>(n_i_full, true, true);
            Vectorinit_CNT
            FCG::context.Xtent_buffer_2_local->val[i] = Vector::init<vtype>(n_i_full, true, true);
        } else {
            Vectorinit_CNT
            FCG::context.Xtent_buffer_local->val[i] = Vector::init<vtype>(n_i, true, true);
            Vectorinit_CNT
            FCG::context.Xtent_buffer_2_local->val[i] = Vector::init<vtype>(n_i, true, true);
        }


      }else{
        FCG::context.RHS_buffer->val[i]->n = n_i;

        FCG::context.Xtent_buffer_local->val[i]->n =    (i == num_levels-1) ? n_i_full : n_i ;
        FCG::context.Xtent_buffer_2_local->val[i]->n =  (i == num_levels-1) ? n_i_full : n_i ;
      }
    }
}

void FCG::freePreconditionContext(){
    free(FCG::context.max_coarse_size);
    Vector::Collection::free(FCG::context.RHS_buffer);

    Vector::Collection::free(FCG::context.Xtent_buffer_local);
    Vector::Collection::free(FCG::context.Xtent_buffer_2_local);
}

__global__
void _triple_innerproduct( itype n, vtype *r, vtype *w, vtype *q, vtype *v, vtype *alpha_beta_gamma, itype shift ){
  __shared__ vtype alpha_shared[FULL_WARP];
  __shared__ vtype beta_shared[FULL_WARP];
  __shared__ vtype gamma_shared[FULL_WARP];

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;
  int warp = threadIdx.x / FULL_WARP;
  int lane = tid % FULL_WARP;
  int i = tid;

  //if(i >= n){
  //  if(lane == 0){
  //    alpha_shared[warp] = 0.;
  //    beta_shared[warp] = 0.;
  //    gamma_shared[warp] = 0.;
  //  }
  //  return;
  //}

  if(threadIdx.x < FULL_WARP){
     alpha_shared[threadIdx.x] = 0.;
     beta_shared[threadIdx.x] = 0.;
     gamma_shared[threadIdx.x] = 0.;
  } 
  __syncthreads();
  if(i >= n) { return; }

  vtype v_i = v[i+shift];
  vtype alpha_i = r[i] * v_i;
  vtype beta_i = w[i] * v_i;
  vtype gamma_i = q[i] * v_i;

  #pragma unroll
  for(int k=FULL_WARP >> 1; k > 0; k = k >> 1){
    alpha_i += __shfl_down_sync(FULL_MASK, alpha_i, k);
    beta_i += __shfl_down_sync(FULL_MASK, beta_i, k);
    gamma_i += __shfl_down_sync(FULL_MASK, gamma_i, k);
  }

  if(lane == 0){
    alpha_shared[warp] = alpha_i;
    beta_shared[warp] = beta_i;
    gamma_shared[warp] = gamma_i;
  }

  __syncthreads();

  if(warp == 0){
    #pragma unroll
    for(int k=FULL_WARP >> 1; k > 0; k = k >> 1){
      alpha_shared[lane] += __shfl_down_sync(FULL_MASK, alpha_shared[lane], k);
      beta_shared[lane] += __shfl_down_sync(FULL_MASK, beta_shared[lane], k);
      gamma_shared[lane] += __shfl_down_sync(FULL_MASK, gamma_shared[lane], k);
    }

    if(lane == 0){
      atomicAdd(&alpha_beta_gamma[0], alpha_shared[0]);
      atomicAdd(&alpha_beta_gamma[1], beta_shared[0]);
      atomicAdd(&alpha_beta_gamma[2], gamma_shared[0]);
    }
  }
}

void triple_innerproduct(vector<vtype> *r, vector<vtype> *w, vector<vtype> *q, vector<vtype> *v, vtype *alpha, vtype *beta, vtype *gamma, itype shift){
  PUSH_RANGE(__func__,4)
    
  assert(r->n == w->n &&  w->n == q->n);

  Vectorinit_CNT
  vector<vtype> *alpha_beta_gamma = Vector::init<vtype>(3, true, true);
  Vector::fillWithValue(alpha_beta_gamma, 0.);

  gridblock gb = gb1d(r->n, BLOCKSIZE);

  _triple_innerproduct<<<gb.g, gb.b>>>(r->n, r->val, w->val, q->val, v->val, alpha_beta_gamma->val, shift);

  vector<vtype> *alpha_beta_gamma_host = Vector::copyToHost(alpha_beta_gamma);

  vtype abg[3];

  CHECK_MPI( MPI_Allreduce(
    alpha_beta_gamma_host->val,
    abg,
    3,
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD
  ) );
  *alpha = abg[0];
  *beta = abg[1];
  *gamma = abg[2];


  Vector::free(alpha_beta_gamma);
  POP_RANGE
}


__global__
void _double_merged_axpy(itype n, vtype *x0, vtype *x1, vtype *x2, vtype alpha_0, vtype alpha_1, itype shift){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  vtype xi1_local = alpha_0 * x0[i+shift] + x1[i+shift];
  x2[i+shift] = alpha_1 * xi1_local + x2[i+shift];
  x1[i+shift] = xi1_local;
  
}


void double_merged_axpy(vector<vtype> *x0, vector<vtype> *x1, vector<vtype> *y, vtype alpha_0, vtype alpha_1, itype n, itype shift){
  PUSH_RANGE(__func__,4)
  
  gridblock gb = gb1d(n, BLOCKSIZE);
  _double_merged_axpy<<<gb.g, gb.b>>>(n, x0->val, x1->val, y->val, alpha_0, alpha_1, shift);

  POP_RANGE
}

void preconditionApply(handles *h, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, vector<vtype> *rhs, vector<vtype> *x){
  PUSH_RANGE(__func__,4)
    
  _MPI_ENV;

  vectorCollection<vtype> *RHS = FCG::context.RHS_buffer;
  vectorCollection<vtype> *Xtent_local = FCG::context.Xtent_buffer_local;
  vectorCollection<vtype> *Xtent_2_local = FCG::context.Xtent_buffer_2_local;


  if(bootamg_data->solver_type == 0){
    for(int k=0; k<boot_amg->n_hrc; k++){

        if(DETAILED_TIMING && ISMASTER){
            TIME::start();
        }
        FCG::setHrrchBufferSize(boot_amg->H_array[k]);

        Vector::copyTo(RHS->val[0], rhs);
        itype n, off;
        if(nprocs>1) {
            n=boot_amg->H_array[k]->A_array[0]->n; 
            off=boot_amg->H_array[k]->A_array[0]->row_shift;
            
            // -----------------------------------------------------
            Vector::copyTo(Xtent_local->val[0], x);
            // -----------------------------------------------------
                
        } else {
            // -----------------------------------------------------
            Vector::copyTo(Xtent_local->val[0], x);
            // -----------------------------------------------------
        }
        if(DETAILED_TIMING && ISMASTER){
            TOTAL_RESTPRE_TIME += TIME::stop();
        } 
        
        
        // -------------------------------------------------------------------------------------------------
        GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, RHS, Xtent_local, Xtent_2_local, 1);
        // -------------------------------------------------------------------------------------------------

            
        if(DETAILED_TIMING && ISMASTER){
            TIME::start();
        }

        
        CSR *A = boot_amg->H_array[k]->A_array[0];
        if(nprocs>1) {
            if(A->halo.to_receive->val[0]<off) {
                off=A->halo.to_receive->val[0];
            }
            if((A->halo.to_receive->val[(A->halo.to_receive_n)-1]-A->halo.to_receive->val[0])>(A->n)) {
                n=(1+A->halo.to_receive->val[(A->halo.to_receive_n)-1]-A->halo.to_receive->val[0]);
            } else {
                n=A->n+(1+A->halo.to_receive->val[(A->halo.to_receive_n)-1]-A->halo.to_receive->val[0]);
            }
            
            // -----------------------------------------------------
            Vector::copyTo(x, Xtent_local->val[0]);
            // -----------------------------------------------------
                
        } else {
            // -----------------------------------------------------
            Vector::copyTo(x, Xtent_local->val[0]);
            // -----------------------------------------------------
        }


        if(DETAILED_TIMING && ISMASTER){
            TOTAL_RESTPRE_TIME += TIME::stop();
        } 

    }
  }else{
    assert(false);
  }
  
  POP_RANGE
}


#include <cuda_profiler_api.h>
vtype flexibileConjugateGradients_v3(CSR* A, handles *h, vector<vtype> *x, vector<vtype> *rhs, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, int precon, int max_iter, double rtol, int *num_iter, bool precondition_flag){
  PUSH_RANGE(__func__,3)
    
  _MPI_ENV;
  precon = 1;
  itype n = A->n;

  Vectorinit_CNT
  vector<vtype> *v = Vector::init<vtype>(A->n, true, true);
    
  Vector::fillWithValue(v, 0.);
  vector<vtype> *w = NULL;
  vector<vtype> *r = NULL;
  vector<vtype> *d = NULL;
  vector<vtype> *q = NULL;

  r = Vector::clone(rhs);

  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
      TIME::start();
  }

  
  if(nprocs > 1)
    halo_sync(A->halo, A, x, true);
        
  w = CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, A, x, NULL, 1., 0.);
//   printf("---- w ----\n"); Vector::print<vtype>(w, -1, stdout);
  
  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
      TOTAL_CSRVECTOR_TIME += TIME::stop();
  }
 
  if(DETAILED_TIMING && ISMASTER){
      TIME::start();
  }
  // w.local r.local
  Vector::axpy(h->cublas_h, w, r, -1.);

  // aggregate norm
  vtype delta0 = Vector::norm_MPI(h->cublas_h, r);
  vtype rhs_norm = Vector::norm_MPI(h->cublas_h, rhs);
  if(delta0 <= DBL_EPSILON * rhs_norm){
    *num_iter = 0;
    exit(1);
  }
  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
      TOTAL_NORMMPI_TIME += TIME::stop();
  }

  if(precon){
    /* apply preconditioner to w */
    // r.local v.full
    if(DETAILED_TIMING && ISMASTER){
      TIME::start();
    }

    if (precondition_flag) {
      preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, v);
    } else {
      Vector::copyTo(v, r);
    }
    if(nprocs > 1) {
      for (int k=0; k<boot_amg->n_hrc; k++)
        halo_sync(boot_amg->H_array[k]->A_array[0]->halo, boot_amg->H_array[k]->A_array[0], v, true);
    }

    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
      TOTAL_PRECONDAPPLY_TIME += TIME::stop();
    } 

  }
  if(DETAILED_TIMING && ISMASTER){
      TIME::start();
  }
  // sync by smoother
  pico_info.update(__FILE__, __LINE__+1);
  CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, A, v, w, 1., 0.);
    
//   PICO_PRINT( 
//     fprintf(fp, "---- v ----\n"); Vector::print<vtype>(v, -1, fp);
//     fprintf(fp, "---- w ----\n"); Vector::print<vtype>(w, -1, fp);
//   )

  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
      TOTAL_CSRVECTOR_TIME += TIME::stop();
  } 

  if(DETAILED_TIMING && ISMASTER){
      TIME::start();
  }
  // get a local v
  vtype alpha_local = Vector::dot(h->cublas_h, r, v);
  vtype beta_local = Vector::dot(h->cublas_h, w, v);
  

  vtype alpha = 0., beta = 0.;
  CHECK_MPI( MPI_Allreduce(
    &alpha_local,
    &alpha,
    1,
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD
  ) );

  CHECK_MPI( MPI_Allreduce(
    &beta_local,
    &beta,
    1,
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD
  ) );

  vtype delta = beta;
  vtype theta = alpha / delta;
  vtype gamma = 0.;

  
  Vector::axpy(h->cublas_h, v, x, theta);
    
  Vector::axpy(h->cublas_h, w, r, -theta);
  
  vtype l2_norm = Vector::norm_MPI(h->cublas_h, r);
  if (l2_norm <= rtol * delta0){
      *num_iter = 1;
  }

  int iter = 0;

  d = Vector::clone(v);
  q = Vector::clone(w);
  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
      TOTAL_AXPY2_TIME += TIME::stop();
  } 

  do{

    int idx = iter % 2;

    if(idx == 0){
      Vector::fillWithValue(v, 0.);

      
      if(precon){
        if(DETAILED_TIMING && ISMASTER){
            TIME::start();
        }
        
        if (precondition_flag) {
          preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, v);
        } else {
          Vector::copyTo(v, r);
        }
        if(nprocs > 1) {
          for (int k=0; k<boot_amg->n_hrc; k++)
            halo_sync(boot_amg->H_array[k]->A_array[0]->halo, boot_amg->H_array[k]->A_array[0], v, true);
        }

        if(DETAILED_TIMING && ISMASTER){
            cudaDeviceSynchronize();
            TOTAL_PRECONDAPPLY_TIME += TIME::stop();
        }
      }

      // A.local * v.full = w.local
      if(DETAILED_TIMING && ISMASTER){
        TIME::start();
      }
      
      
      pico_info.update(__FILE__, __LINE__+1);
      CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, A, v, w, 1., 0.);
      
      
      if(DETAILED_TIMING && ISMASTER){
         cudaDeviceSynchronize();
         TOTAL_CSRVECTOR_TIME += TIME::stop();
      }
      // r.local w.local q.local v.full
      if(DETAILED_TIMING && ISMASTER){
    	 TIME::start();
      }
      
      triple_innerproduct(r, w, q, v, &alpha, &beta, &gamma, 0);
      
      if(DETAILED_TIMING && ISMASTER){
          cudaDeviceSynchronize();
          TOTAL_TRIPLEPROD_TIME += TIME::stop();
      } 
    }else{
      Vector::fillWithValue(d, 0.);
        
      if(precon){
        if(DETAILED_TIMING && ISMASTER){
     	  TIME::start();
        }
        
        if (precondition_flag) {
          preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, d);
        } else {
          Vector::copyTo(d, r);
        }
        if(nprocs > 1) {
          for (int k=0; k<boot_amg->n_hrc; k++)
            halo_sync(boot_amg->H_array[k]->A_array[0]->halo, boot_amg->H_array[k]->A_array[0], d, true);
        }

        if(DETAILED_TIMING && ISMASTER){
          cudaDeviceSynchronize();
          TOTAL_PRECONDAPPLY_TIME += TIME::stop();
        } 
      }

      // A.local * d.full = q.local
      if(DETAILED_TIMING && ISMASTER){
     	  TIME::start();
      }
      
      pico_info.update(__FILE__, __LINE__+1);
      CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, A, d, q, 1., 0.);
      
      
      if(DETAILED_TIMING && ISMASTER){
         cudaDeviceSynchronize();
     	 TOTAL_CSRVECTOR_TIME += TIME::stop();
      }

      if(DETAILED_TIMING && ISMASTER){
     	 TIME::start();
      }
      
      triple_innerproduct(r, q, w, d, &alpha, &beta, &gamma, 0);
      
      if(DETAILED_TIMING && ISMASTER){
          cudaDeviceSynchronize();
          TOTAL_TRIPLEPROD_TIME += TIME::stop();
      } 
    }

    theta = gamma / delta;

    delta = beta - pow(gamma, 2) / delta;
    vtype theta_2 = alpha / delta;

    if(DETAILED_TIMING && ISMASTER){
     	 TIME::start();
    }
    if(idx == 0){
      double_merged_axpy(d, v, x, -theta, theta_2, d->n, 0);

      double_merged_axpy(q, w, r, -theta, -theta_2, r->n, 0);
    }else{
      double_merged_axpy(v, d, x, -theta, theta_2, v->n, 0);

      double_merged_axpy(w, q, r, -theta, -theta_2, r->n, 0);
    }
    if(DETAILED_TIMING && ISMASTER){
          cudaDeviceSynchronize();
          TOTAL_DOUBLEMERGED_TIME += TIME::stop();
    } 

    if(DETAILED_TIMING && ISMASTER){
     	 TIME::start();
    }
    l2_norm = Vector::norm_MPI(h->cublas_h, r);
    if(DETAILED_TIMING && ISMASTER){
          cudaDeviceSynchronize();
          TOTAL_NORMPI2_TIME += TIME::stop();
    } 

    if(VERBOSE > 0)
      std::cout << "bootpcg iteration: " << iter << "  residual: " << l2_norm << " relative residual: " << l2_norm / delta0 << "\n";

    iter++;

  }while(l2_norm > rtol * delta0 && iter < max_iter);
  
  
  assert( std::isfinite(l2_norm) );

  *num_iter = iter + 1;

  if(precon && precondition_flag){
    FCG::freePreconditionContext();
  }

  Vector::free(w);
  free(v);
  Vector::free(d);
  Vector::free(q);
  Vector::free(r);
  if(0 && ISMASTER) {
    Eval::printMetaData("agg;csrvector_time", TOTAL_CSRVECTOR_TIME/1000,1);
    Eval::printMetaData("agg;normpi_time", TOTAL_NORMMPI_TIME/1000,1);
    Eval::printMetaData("agg;precondapply_time", TOTAL_PRECONDAPPLY_TIME/1000,1);
    Eval::printMetaData("agg;axpy2_time", TOTAL_AXPY2_TIME/1000,1);
    Eval::printMetaData("agg;tripleprod_time", TOTAL_TRIPLEPROD_TIME/1000,1);
    Eval::printMetaData("agg;doublemerged_time", TOTAL_DOUBLEMERGED_TIME/1000,1);
    Eval::printMetaData("agg;normpi2_time", TOTAL_NORMPI2_TIME/1000,1);
  }
  
  POP_RANGE
  return l2_norm;
}

