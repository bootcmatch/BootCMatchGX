#include "solver/cghs/CG_HS.h"

#include "halo_communication/halo_communication.h"
#include "op/basic.h"
#include "preconditioner/prec_apply.h"
#include "utility/handles.h"
#include "utility/memory.h"
#include "utility/profiling.h"
#include "utility/utils.h"

#define VEC_DIM 3

/**
 * @brief Solves a linear system using the Conjugate Gradient (CG) method.
 *
 * @param h Handles for CUDA streams and various resources.
 * @param Alocal Pointer to the local sparse matrix in CSR format.
 * @param rhs_loc Right-hand side vector.
 * @param x0_loc Initial solution guess (solution is updated in this variable).
 * @param p User-defined solver and preconditioner settings (like iteration limits, stopping criteria, etc.).
 * @param pr Actual preconditioner data (such as preconditioner type and internal structures).
 * @param out Solver output structure for residuals and convergence history.
 * @return Status code: 0 for success, nonzero for failure.
 */
int cg_hs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, const params& p, cgsprec* pr, SolverOut* out)
{
    _MPI_ENV;

    int excnlim = -1;
    int retval = -1;
    int iter = 0;

    stype ln = Alocal->n;
    gstype fn = Alocal->full_n;

    out->resHist = MALLOC(vtype, p.itnlim, true);

    vector<vtype>* s_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* p2_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* r2_loc = Vector::init<vtype>(ln, true, true);

    Vector::fillWithValue(s_loc, 0.);

    vector<vtype>* u2_loc = NULL;
    if (pr->ptype != PreconditionerType::NONE) {
        u2_loc = Vector::init<vtype>(ln, true, true);
    }

    Vector::copyTo(r2_loc, rhs_loc); // r2_loc = rhs_loc

    vector<vtype>* w_loc = CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x0_loc, NULL, 1., 0.);

    my_axpby(w_loc->val, ln, r2_loc->val, -1.0, 1.0);

    if (pr->ptype != PreconditionerType::NONE) {
        Vector::fillWithValue(u2_loc, 0.);
        // u2_loc = prec * r2_loc
        prec_apply(h, Alocal, r2_loc, u2_loc, pr, p, &out->precOut);

        Vector::copyTo(p2_loc, u2_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0); // p2_loc = u2_loc
    } else {
        Vector::copyTo(p2_loc, r2_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0); // p2_loc = r2_loc
    }

    vector<vtype>* urpsm = Vector::init<vtype>(VEC_DIM, true, false);
    vector<vtype>* urps = Vector::init<vtype>(VEC_DIM, true, false);
    vector<vtype>* ab = Vector::init<vtype>(VEC_DIM - 1, true, false);

    vtype f, delta0, l2_norm;
    delta0 = 1.0;

    if (p.stop_criterion == 1) {
        delta0 = Vector::norm_MPI(h->cublas_h, r2_loc);
    }

    if (ISMASTER) {
        out->resHist[0] = delta0;
    }

    for (iter = 0; iter < p.itnlim; iter++) {

        if (iter > 0) {
            if (pr->ptype != PreconditionerType::NONE) {
                Vector::fillWithValue(u2_loc, 0.);

                prec_apply(h, Alocal, r2_loc, u2_loc, pr, p, &out->precOut);

                urpsm->val[2] = Vector::dot(h->cublas_h, r2_loc, u2_loc);
            } else {
                urpsm->val[2] = Vector::dot(h->cublas_h, r2_loc, r2_loc);
            }

            BEGIN_PROF("MPI_Allreduce");
            CHECK_MPI(MPI_Allreduce(urpsm->val, urps->val, VEC_DIM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
            END_PROF("MPI_Allreduce");

            ab->val[1] = urps->val[2] / urps->val[0]; // beta
            f = ab->val[1];

            if (pr->ptype != PreconditionerType::NONE) {
                my_axpby(u2_loc->val, ln, p2_loc->val, 1.0, f);
            } else {
                my_axpby(r2_loc->val, ln, p2_loc->val, 1.0, f);
            }
        }

        CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, p2_loc, s_loc, 1., 0.);

        if (pr->ptype != PreconditionerType::NONE) {
            // ur = u2_loc . r2_loc
            urpsm->val[0] = Vector::dot(h->cublas_h, u2_loc, r2_loc);
        } else {
            // ur = r2_loc . r2_loc
            urpsm->val[0] = Vector::dot(h->cublas_h, r2_loc, r2_loc);
        }

        // ps = p2_loc . s_loc
        urpsm->val[1] = Vector::dot(h->cublas_h, p2_loc, s_loc);

        BEGIN_PROF("MPI_Allreduce");
        CHECK_MPI(MPI_Allreduce(urpsm->val, urps->val, VEC_DIM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        END_PROF("MPI_Allreduce");

        ab->val[0] = urps->val[0] / urps->val[1]; // alpha
        f = ab->val[0];

        my_axpby(p2_loc->val, ln, x0_loc->val, f, 1.0); // x2 = x2 + alpha * p2
        my_axpby(s_loc->val, ln, r2_loc->val, -f, 1.0); // r2 = r2 - alpha * s

        l2_norm = Vector::norm_MPI(h->cublas_h, r2_loc);

        if (ISMASTER) {
            if (p.dispnorm) {
                printf("%d) r norm: %.10lf\n", iter, l2_norm);
            }
            out->resHist[iter + 1] = l2_norm;
        }

        if (l2_norm < p.rtol * delta0) {
            excnlim = 0;
            retval = 0;
            break;
        }
    } // end iter

    // Even if max_iter exceeded return the current status
    if (ISMASTER) {
        out->del0 = delta0;
        out->full_n = fn;
        out->local_n = ln;
        out->niter = iter + 1 + excnlim;
        out->exitRes = l2_norm;
    }

    Vector::free(w_loc);
    Vector::free(s_loc);
    Vector::free(r2_loc);
    Vector::free(p2_loc);
    Vector::free(urpsm);
    Vector::free(urps);
    Vector::free(ab);
    if (pr->ptype != PreconditionerType::NONE) {
        Vector::free(u2_loc);
    }

    return retval;
}

/**
 * @brief Solves a linear system using CG-HS and returns the solution.
 *
 * @param h Handles for CUDA streams and various resources.
 * @param Alocal Pointer to the sparse matrix in CSR format.
 * @param rhs_loc Right-hand side vector.
 * @param x0_loc Initial solution guess.
 * @param p User-defined solver and preconditioner settings (like iteration limits, stopping criteria, etc.).
 * @param pr Actual preconditioner data (such as preconditioner type and internal structures).
 * @param out Solver output structure.
 * @return The computed solution vector.
 */
vector<vtype>* solve_cg_hs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, cgsprec* pr, const params& p, SolverOut* out)
{
    _MPI_ENV;

    out->retv = cg_hs(h, Alocal, rhs_loc, x0_loc, p, pr, out);

    // To be removed, should be used the solution returned in x0_loc. Instead, should be returned out->retv (or void then check out->retv in the calling function)
    out->sol_local = Vector::init<vtype>(x0_loc->n, true, true);
    Vector::copyTo(out->sol_local, x0_loc);

    return out->sol_local;
}
