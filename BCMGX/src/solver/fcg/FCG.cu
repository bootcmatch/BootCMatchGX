#include "solver/fcg/FCG.h"

#include "halo_communication/halo_communication.h"
#include "op/basic.h"
#include "op/double_merged_axpy.h"
#include "op/triple_inner_product.h"
#include "preconditioner/bcmg/BcmgPreconditionContext.h"
#include "preconditioner/prec_apply.h"
#include "utility/profiling.h"

/**
 * @brief Performs Flexible Conjugate Gradient method (v3) to solve a linear system.
 * 
 * This function solves the system \(A x = b\) using the Flexible Conjugate Gradient method,
 * with the option to use a preconditioner. It applies the method iteratively, checking for convergence
 * based on the relative residual norm and a given tolerance.
 * 
 * @param A The matrix representing the linear system (CSR format).
 * @param h Handles for CUDA streams and various resources.
 * @param x The initial guess for the solution.
 * @param rhs The right-hand side vector.
 * @param pr Actual preconditioner data (such as preconditioner type and internal structures).
 * @param p User-defined solver and preconditioner settings (like iteration limits, stopping criteria, etc.).
 * @param out Output structure to store results, including convergence history and other statistics.
 * @return int Return value indicating the exit status: 0 for success, -1 if iteration limit is reached.
 */
int flexibileConjugateGradients_v3(CSR* A, handles* h, vector<vtype>* x, vector<vtype>* rhs, cgsprec* pr, const params& p, SolverOut* out)
{
    _MPI_ENV;

    // This should contain info about the "exit status"... so far set to -1 only if iter > p.itnlim
    int retval = 0;

    vector<vtype>* v = Vector::init<vtype>(A->n, true, true);
    vector<vtype>* alpha_beta_gamma = Vector::init<vtype>(3, true, true);

    Vector::fillWithValue(v, 0.);

    vector<vtype>* w = NULL;
    vector<vtype>* r = NULL;
    vector<vtype>* d = NULL;
    vector<vtype>* q = NULL;

    r = Vector::clone(rhs);

    w = CSRm::CSRVector_product_adaptive_miniwarp_witho(A, x, NULL, 1., 0.);

    my_axpby(w->val, w->n, r->val, -1., 1.);

    vtype delta0 = Vector::norm_MPI(h->cublas_h, r);
    vtype rhs_norm = Vector::norm_MPI(h->cublas_h, rhs);

    if (p.sprec != PreconditionerType::NONE) {
        prec_apply(h, A, r, v, pr, p, &out->precOut);
    } else {
        Vector::copyTo(v, r, (nprocs > 1) ? *(A->os.streams->comm_stream) : 0);
    }

    CSRm::CSRVector_product_adaptive_miniwarp_witho(A, v, w, 1., 0.);

    vtype alpha_local = Vector::dot(h->cublas_h, r, v);
    vtype beta_local = Vector::dot(h->cublas_h, w, v);

    vtype alpha = 0., beta = 0.;

    BEGIN_PROF("MPI_Allreduce");
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
    END_PROF("MPI_Allreduce");

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
                // Vector::copyTo(v, r);
                Vector::copyTo(v, r, (nprocs > 1) ? *(A->os.streams->comm_stream) : 0);
            }

            CSRm::CSRVector_product_adaptive_miniwarp_witho(A, v, w, 1., 0.);

            triple_innerproduct(r, w, q, v, alpha_beta_gamma, &alpha, &beta, &gamma, 0);
        } else {
            Vector::fillWithValue(d, 0.);

            if (p.sprec != PreconditionerType::NONE) {
                prec_apply(h, A, r, d, pr, p, &out->precOut);
            } else {
                Vector::copyTo(d, r, (nprocs > 1) ? *(A->os.streams->comm_stream) : 0);
            }

            CSRm::CSRVector_product_adaptive_miniwarp_witho(A, d, q, 1., 0.);

            triple_innerproduct(r, q, w, d, alpha_beta_gamma, &alpha, &beta, &gamma, 0);
        }

        theta = gamma / delta;

        // delta = beta - pow(gamma, 2) / delta;
        delta = beta - ((gamma * gamma) / delta);
        vtype theta_2 = alpha / delta;

        if (idx == 0) {
            double_merged_axpy(d, v, x, -theta, theta_2, d->n, 0);
            double_merged_axpy(q, w, r, -theta, -theta_2, r->n, 0);
        } else {
            double_merged_axpy(v, d, x, -theta, theta_2, v->n, 0);
            double_merged_axpy(w, q, r, -theta, -theta_2, r->n, 0);
        }

        l2_norm = Vector::norm_MPI(h->cublas_h, r);

        if (ISMASTER) {
            if (p.dispnorm) {
                printf("%d) r norm: %.10lf\n", iter, l2_norm);
            }
        }

        iter++;

    } while (l2_norm > p.rtol * delta0 && iter < p.itnlim);

    assert(std::isfinite(l2_norm));

    if (iter >= p.itnlim) {
        retval = -1;
    }

    out->niter = iter + 1;
    out->exitRes = l2_norm;

    Vector::free(alpha_beta_gamma);
    Vector::free(w);
    Vector::free(d);
    Vector::free(q);
    Vector::free(r);
    Vector::free(v);

    return retval;
}

/**
 * @brief Solves a linear system using the Flexible Conjugate Gradient method (v3).
 * 
 * This function wraps the flexible conjugate gradient method, calling it and managing
 * the solution process. It stores the solution in `x0` and provides a copy in the return value.
 * 
 * @param h Handles for CUDA streams and various resources.
 * @param Alocal The matrix representing the linear system (CSR format).
 * @param rhs The right-hand side vector.
 * @param x0 The initial guess for the solution (input and output).
 * @param pr Actual preconditioner data (such as preconditioner type and internal structures).
 * @param p User-defined solver and preconditioner settings (like iteration limits, stopping criteria, etc.).
 * @param out Output structure to store results, including convergence history and other statistics.
 * @return vector<vtype>* The solution vector.
 */
vector<vtype>* solve_fcg(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x0, cgsprec* pr, const params& p, SolverOut* out)
{
    _MPI_ENV;

    out->local_n = Alocal->n;
    out->full_n = Alocal->full_n;

    out->retv = flexibileConjugateGradients_v3(
        Alocal,
        h,
        x0,
        rhs,
        pr,
        p,
        out);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    // To be removed, should be used the solution returned in x0. Instead, should be returned out->retv (or void then check out->retv in the calling function)
    vector<vtype>* Sol = Vector::init<vtype>(Alocal->n, true, true);
    Vector::copyTo(Sol, x0);

    return Sol;
}
