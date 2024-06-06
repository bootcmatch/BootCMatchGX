#include "triple_inner_product.h"

#include "utility/cudamacro.h"
#include "utility/function_cnt.h"
#include "utility/metrics.h"
#include "utility/timing.h"

__global__ void _triple_innerproduct(itype n, vtype* r, vtype* w, vtype* q, vtype* v, vtype* alpha_beta_gamma, itype shift)
{
    __shared__ vtype alpha_shared[FULL_WARP];
    __shared__ vtype beta_shared[FULL_WARP];
    __shared__ vtype gamma_shared[FULL_WARP];

    itype tid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp = threadIdx.x / FULL_WARP;
    int lane = tid % FULL_WARP;
    int i = tid;

    if (threadIdx.x < FULL_WARP) {
        alpha_shared[threadIdx.x] = 0.;
        beta_shared[threadIdx.x] = 0.;
        gamma_shared[threadIdx.x] = 0.;
    }
    __syncthreads();
    if (i >= n) {
        return;
    }

    vtype v_i = v[i + shift];
    vtype alpha_i = r[i] * v_i;
    vtype beta_i = w[i] * v_i;
    vtype gamma_i = q[i] * v_i;

#pragma unroll
    for (int k = FULL_WARP >> 1; k > 0; k = k >> 1) {
        alpha_i += __shfl_down_sync(FULL_MASK, alpha_i, k);
        beta_i += __shfl_down_sync(FULL_MASK, beta_i, k);
        gamma_i += __shfl_down_sync(FULL_MASK, gamma_i, k);
    }

    if (lane == 0) {
        alpha_shared[warp] = alpha_i;
        beta_shared[warp] = beta_i;
        gamma_shared[warp] = gamma_i;
    }

    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int k = FULL_WARP >> 1; k > 0; k = k >> 1) {
            alpha_shared[lane] += __shfl_down_sync(FULL_MASK, alpha_shared[lane], k);
            beta_shared[lane] += __shfl_down_sync(FULL_MASK, beta_shared[lane], k);
            gamma_shared[lane] += __shfl_down_sync(FULL_MASK, gamma_shared[lane], k);
        }

        if (lane == 0) {
            atomicAdd(&alpha_beta_gamma[0], alpha_shared[0]);
            atomicAdd(&alpha_beta_gamma[1], beta_shared[0]);
            atomicAdd(&alpha_beta_gamma[2], gamma_shared[0]);
        }
    }
}

void triple_innerproduct(vector<vtype>* r, vector<vtype>* w, vector<vtype>* q, vector<vtype>* v, vtype* alpha, vtype* beta, vtype* gamma, itype shift)
{
    PUSH_RANGE(__func__, 4)

    _MPI_ENV;

    assert(r->n == w->n && w->n == q->n);

#if DETAILED_TIMING
    if (ISMASTER) {
        TIME::start();
    }
#endif

    Vectorinit_CNT
        vector<vtype>* alpha_beta_gamma
        = Vector::init<vtype>(3, true, true);
    Vector::fillWithValue(alpha_beta_gamma, 0.);

    GridBlock gb = gb1d(r->n, BLOCKSIZE);

    _triple_innerproduct<<<gb.g, gb.b>>>(r->n, r->val, w->val, q->val, v->val, alpha_beta_gamma->val, shift);

    vector<vtype>* alpha_beta_gamma_host = Vector::copyToHost(alpha_beta_gamma);

#if DETAILED_TIMING
    if (ISMASTER) {
        cudaDeviceSynchronize();
        TOTAL_TRIPLEPROD_TIME += TIME::stop();
    }
#endif

    vtype abg[3];

#if DETAILED_TIMING
    if (ISMASTER) {
        TIME::start();
    }
#endif

    CHECK_MPI(MPI_Allreduce(
        alpha_beta_gamma_host->val,
        abg,
        3,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD));

#if DETAILED_TIMING
    if (ISMASTER) {
        cudaDeviceSynchronize();
        TOTAL_ALLREDUCE_TIME += TIME::stop();
    }
#endif

    *alpha = abg[0];
    *beta = abg[1];
    *gamma = abg[2];

    Vector::free(alpha_beta_gamma);
    POP_RANGE
}
