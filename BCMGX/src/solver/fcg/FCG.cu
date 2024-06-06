#include "solver/fcg/FCG.h"

#include "basic_kernel/halo_communication/extern.h"
#include "basic_kernel/halo_communication/halo_communication.h"
#include "op/basic.h"
#include "op/double_merged_axpy.h"
#include "op/triple_inner_product.h"
#include "preconditioner/bcmg/BcmgPreconditionContext.h"
#include "preconditioner/prec_apply.h"
#include "utility/function_cnt.h"
#include "utility/metrics.h"
#include "utility/timing.h"

#include <cuda_profiler_api.h>

vtype flexibileConjugateGradients_v3(CSR* A, handles* h, vector<vtype>* x, vector<vtype>* rhs, cgsprec* pr, const params& p, SolverOut* out)
{
    PUSH_RANGE(__func__, 3)

    _MPI_ENV;

    if (myid == 0) {
        cudaProfilerStart();
    }

    Vectorinit_CNT
        vector<vtype>* v
        = Vector::init<vtype>(A->n, true, true);

    Vector::fillWithValue(v, 0.);
    vector<vtype>* w = NULL;
    vector<vtype>* r = NULL;
    vector<vtype>* d = NULL;
    vector<vtype>* q = NULL;

    r = Vector::clone(rhs);

#if DETAILED_TIMING
    if (ISMASTER) {
        cudaDeviceSynchronize();
        TIME::start();
    }
#endif

    w = CSRm::CSRVector_product_adaptive_miniwarp_witho(A, x, NULL, 1., 0.);

#if DETAILED_TIMING
    if (ISMASTER) {
        cudaDeviceSynchronize();
        TOTAL_CSRVECTOR_TIME += TIME::stop();
    }
#endif

    // w.local r.local
    my_axpby(w->val, w->n, r->val, -1., 1.);

    // aggregate norm
    vtype delta0 = Vector::norm_MPI(h->cublas_h, r);
    vtype rhs_norm = Vector::norm_MPI(h->cublas_h, rhs);

    if (delta0 <= DBL_EPSILON * rhs_norm) {
        out->niter = 0;
        exit(1);
    }

    if (ISMASTER) {
        TIME::start();
    }

    if (p.sprec != PreconditionerType::NONE) {
        prec_apply(h, A, r, v, pr, p, &out->precOut);
    } else {
        Vector::copyTo(v, r);
    }

#if DETAILED_TIMING
    if (ISMASTER) {
        TIME::start();
    }
#endif

    // sync by smoother
    pico_info.update(__FILE__, __LINE__ + 1);

    CSRm::CSRVector_product_adaptive_miniwarp_witho(A, v, w, 1., 0.);

#if DETAILED_TIMING
    if (ISMASTER) {
        cudaDeviceSynchronize();
        TOTAL_CSRVECTOR_TIME += TIME::stop();
    }
#endif

    // get a local v
    vtype alpha_local = Vector::dot(h->cublas_h, r, v);
    vtype beta_local = Vector::dot(h->cublas_h, w, v);

    vtype alpha = 0., beta = 0.;
    CHECK_MPI(MPI_Allreduce(
        &alpha_local,
        &alpha,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD));

    CHECK_MPI(MPI_Allreduce(
        &beta_local,
        &beta,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD));

    vtype delta = beta;
    vtype theta = alpha / delta;
    vtype gamma = 0.;

    my_axpby(v->val, v->n, x->val, theta, 1.);
    my_axpby(w->val, w->n, r->val, -theta, 1.);

    vtype l2_norm = Vector::norm_MPI(h->cublas_h, r);
    if (l2_norm <= p.rtol * delta0) {
        out->niter = 1;
    }

    int iter = 0;

    d = Vector::clone(v);
    q = Vector::clone(w);

    do {

        int idx = iter % 2;

        if (idx == 0) {

            Vector::fillWithValue(v, 0.);

            if (p.sprec != PreconditionerType::NONE) {
                prec_apply(h, A, r, v, pr, p, &out->precOut);
            } else {
                Vector::copyTo(v, r);
            }

#if DETAILED_TIMING
            if (ISMASTER) {
                TIME::start();
            }
#endif

            pico_info.update(__FILE__, __LINE__ + 1);

            CSRm::CSRVector_product_adaptive_miniwarp_witho(A, v, w, 1., 0.);

#if DETAILED_TIMING
            if (ISMASTER) {
                cudaDeviceSynchronize();
                TOTAL_CSRVECTOR_TIME += TIME::stop();
            }
#endif

            triple_innerproduct(r, w, q, v, &alpha, &beta, &gamma, 0);

        } else {

            Vector::fillWithValue(d, 0.);

            if (p.sprec != PreconditionerType::NONE) {
                prec_apply(h, A, r, d, pr, p, &out->precOut);
            } else {
                Vector::copyTo(d, r);
            }

#if DETAILED_TIMING
            if (ISMASTER) {
                TIME::start();
            }
#endif
            pico_info.update(__FILE__, __LINE__ + 1);

            CSRm::CSRVector_product_adaptive_miniwarp_witho(A, d, q, 1., 0.);

#if DETAILED_TIMING
            if (ISMASTER) {
                cudaDeviceSynchronize();
                TOTAL_CSRVECTOR_TIME += TIME::stop();
            }
#endif

            triple_innerproduct(r, q, w, d, &alpha, &beta, &gamma, 0);
        }

        theta = gamma / delta;

        delta = beta - pow(gamma, 2) / delta;
        vtype theta_2 = alpha / delta;

#if DETAILED_TIMING
        if (ISMASTER) {
            TIME::start();
        }
#endif

        if (idx == 0) {
            double_merged_axpy(d, v, x, -theta, theta_2, d->n, 0);

            double_merged_axpy(q, w, r, -theta, -theta_2, r->n, 0);
        } else {
            double_merged_axpy(v, d, x, -theta, theta_2, v->n, 0);

            double_merged_axpy(w, q, r, -theta, -theta_2, r->n, 0);
        }

#if DETAILED_TIMING
        if (ISMASTER) {
            cudaDeviceSynchronize();
            TOTAL_DOUBLEMERGED_TIME += TIME::stop();
        }
#endif

#if DETAILED_TIMING
        if (ISMASTER) {
            TIME::start();
        }
#endif

        l2_norm = Vector::norm_MPI(h->cublas_h, r);

#if DETAILED_TIMING
        if (ISMASTER) {
            cudaDeviceSynchronize();
            TOTAL_NORM_TIME += TIME::stop();
        }
#endif

        if (ISMASTER) {
            if (p.dispnorm) {
                printf("%d) r norm: %.10lf\n", iter, l2_norm);
            }
        }

        iter++;

    } while (l2_norm > p.rtol * delta0 && iter < p.itnlim);

    if (ISMASTER) {
        printf("New solver timer: %f (in seconds)\n", TIME::stop() / 1000);
    }

    assert(std::isfinite(l2_norm));

    out->niter = iter + 1;

    if (myid == 0) {
        cudaProfilerStop();
    }

    Vector::free(w);
    free(v);
    Vector::free(d);
    Vector::free(q);
    Vector::free(r);

    POP_RANGE
    return l2_norm;
}

vector<vtype>* solve_fcg(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x0, cgsprec* pr, const params& p, SolverOut* out)
{
    PUSH_RANGE(__func__, 2)

    _MPI_ENV;

    vector<vtype>* Sol = Vector::init<vtype>(Alocal->n, true, true);
    Vector::fillWithValue(Sol, 0.);

    if (ISMASTER) {
        TIME::start();
    }

    vtype residual = 0.;

    out->local_n = Alocal->n;
    out->full_n = Alocal->full_n;

    residual = flexibileConjugateGradients_v3(
        Alocal, // pr.H->A_array[0],
        h,
        Sol,
        rhs,
        pr,
        p,
        out);

    out->exitRes = residual;

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    if (ISMASTER) {
        cudaDeviceSynchronize();
        out->solTime = TIME::stop();
    }

    POP_RANGE
    return Sol;
}
