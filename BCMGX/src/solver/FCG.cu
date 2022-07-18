#pragma once

#include "FCG.h"

float TOTAL_CSRVECTOR_TIME=0.;
float TOTAL_NORMMPI_TIME=0.;
float TOTAL_PRECONDAPPLY_TIME=0.;
float TOTAL_AXPY2_TIME=0.;
float TOTAL_TRIPLEPROD_TIME=0.;
float TOTAL_DOUBLEMERGED_TIME=0.;
float TOTAL_NORMPI2_TIME=0.;
float TOTAL_RESTPRE_TIME=0.;

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

  assert(r->n == w->n &&  w->n == q->n);

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
  
  gridblock gb = gb1d(n, BLOCKSIZE);
  _double_merged_axpy<<<gb.g, gb.b>>>(n, x0->val, x1->val, y->val, alpha_0, alpha_1, shift);

}


void preconditionApply(handles *h, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, vector<vtype> *rhs, vector<vtype> *x){
  _MPI_ENV;

  vectorCollection<vtype> *RHS = FCG::context.RHS_buffer;
  vectorCollection<vtype> *Xtent = FCG::context.Xtent_buffer;

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
      Vector::copyToWithOff(Xtent->val[0], x, n, off);
  } else {
      Vector::copyTo(Xtent->val[0], x);
  }
  if(DETAILED_TIMING && ISMASTER){
      TOTAL_RESTPRE_TIME += TIME::stop();
  } 

  GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, RHS, Xtent, 1);
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
      Vector::copyToWithOff(x, Xtent->val[0],n,off);
      } else {
      	  Vector::copyTo(x, Xtent->val[0]);
      }

      if(nprocs > 1){
          sync_solution(A->halo, A, x);
      }
      if(DETAILED_TIMING && ISMASTER){
          TOTAL_RESTPRE_TIME += TIME::stop();
      } 

    }
  }else{
    assert(false);
  }
}

#include <cuda_profiler_api.h>
vtype flexibileConjugateGradients_v3(CSR* A, handles *h, vector<vtype> *x, vector<vtype> *rhs, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, int precon, int max_iter, double rtol, int *num_iter){
  _MPI_ENV;
  precon = 1;
  itype n = A->n;
  vector<vtype> *v = Vector::init<vtype>(A->full_n, true, true);

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

  w = CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, x, NULL, 1., 0.);
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
    preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, v);
    if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
      TOTAL_PRECONDAPPLY_TIME += TIME::stop();
    } 

  }
  if(DETAILED_TIMING && ISMASTER){
      TIME::start();
  }
  // sync by smoother
  CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, v, w, 1., 0.);
  if(DETAILED_TIMING && ISMASTER){
      cudaDeviceSynchronize();
      TOTAL_CSRVECTOR_TIME += TIME::stop();
  } 
  if(DETAILED_TIMING && ISMASTER){
      TIME::start();
  }
  // get a local v
  vector<vtype> *v_local = Vector::init<vtype>(n, false, true);
  v_local->val = v->val + A->row_shift;

  vtype alpha_local = Vector::dot(h->cublas_h, r, v_local);
  vtype beta_local = Vector::dot(h->cublas_h, w, v_local);

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
  free(v_local);

  vtype delta = beta;
  vtype theta = alpha / delta;
  vtype gamma = 0.;

  Vector::axpyWithOff(h->cublas_h, v, x, theta, A->n, A->row_shift);

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
      Vector::fillWithValueWithOff(v, 0.,A->n, A->row_shift);

      if(precon){
  	if(DETAILED_TIMING && ISMASTER){
     	 TIME::start();
 	}
        preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, v);
        if(DETAILED_TIMING && ISMASTER){
          cudaDeviceSynchronize();
          TOTAL_PRECONDAPPLY_TIME += TIME::stop();
        } 
      }

      // A.local * v.full = w.local
      if(DETAILED_TIMING && ISMASTER){
     	 TIME::start();
      }
      CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, v, w, 1., 0.);
      if(DETAILED_TIMING && ISMASTER){
         cudaDeviceSynchronize();
     	 TOTAL_CSRVECTOR_TIME += TIME::stop();
      } 

      // r.local w.local q.local v.full
      if(DETAILED_TIMING && ISMASTER){
    	 TIME::start();
      }
      triple_innerproduct(r, w, q, v, &alpha, &beta, &gamma, A->row_shift);
      if(DETAILED_TIMING && ISMASTER){
          cudaDeviceSynchronize();
          TOTAL_TRIPLEPROD_TIME += TIME::stop();
      } 
    }else{
      Vector::fillWithValueWithOff(d, 0., A->n, A->row_shift);

      if(precon){
        if(DETAILED_TIMING && ISMASTER){
     	  TIME::start();
        }
        preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, d);
        if(DETAILED_TIMING && ISMASTER){
          cudaDeviceSynchronize();
          TOTAL_PRECONDAPPLY_TIME += TIME::stop();
        } 
      }

      // A.local * d.full = q.local
      if(DETAILED_TIMING && ISMASTER){
     	  TIME::start();
      }
      CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, d, q, 1., 0.);
      if(DETAILED_TIMING && ISMASTER){
         cudaDeviceSynchronize();
     	 TOTAL_CSRVECTOR_TIME += TIME::stop();
      } 

      if(DETAILED_TIMING && ISMASTER){
     	 TIME::start();
      }
      triple_innerproduct(r, q, w, d, &alpha, &beta, &gamma, A->row_shift);
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
      double_merged_axpy(d, v, x, -theta, theta_2, A->n, A->row_shift);
      double_merged_axpy(q, w, r, -theta, -theta_2, r->n, 0);
    }else{
      double_merged_axpy(v, d, x, -theta, theta_2, A->n, A->row_shift);
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

  if(precon){
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

  return l2_norm;
}

