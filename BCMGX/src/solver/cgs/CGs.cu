#include "solver/cgs/CGs.h"

#include "halo_communication/halo_communication.h"
#include "op/LBfunctions.h"
#include "op/basic.h"
#include "op/scalarWorkMO.h"
#include "preconditioner/prec_apply.h"
#include "solver/cgs_cublas/CGscublas.h"
#include "solver/krylov_base/computevm.h"
#include "solver/krylov_base/mpk.h"
#include "solver/pipelined_cgs/pipelinedCGs.h"
#include "utility/handles.h"
#include "utility/profiling.h"

/**
 * @brief Performs the Conjugate Gradient (CG) s-step method for solving linear systems.
 * 
 * This function implements the CG s-step method as described in the papers by Chronopoulos and Gear. It performs
 * iterative steps to solve the linear system \( A x = b \), using multiple stages (s-steps) of matrix-vector
 * products and preconditioning. The method alternates between using preconditioning and performing matrix-vector
 * products based on the solver's configuration.
 * 
 * The function iterates until a convergence criterion is met (based on the residual norm) or the maximum number of
 * iterations is reached. The residuals are recomputed periodically depending on the provided settings.
 * 
 * @param h Handle to CUDA and other resources.
 * @param Alocal The local CSR matrix used in the matrix-vector products.
 * @param rhs_loc The local right-hand side vector.
 * @param x_loc The local initial guess vector.
 * @param p The parameters that control the behavior of the solver, including solver settings and limits.
 * @param pr Preconditioner storage.
 * @param out The output structure that will store the results, including residual history and final solution.
 * 
 * @return The return code representing the status of the solver (0 for success, non-zero for failure).
 * 
 * @note This function performs the iterative s-step CG method, which alternates between matrix-vector products and
 *       preconditioning, depending on the current iteration step and the solver configuration.
 *       The residuals are recalculated at specified intervals based on the provided parameters.
 * 
 * @details The function follows the s-step approach, where the residual is computed and updated periodically,
 *          alternating between using the preconditioner and performing matrix-vector multiplications. The result
 *          is stored in the provided output structure `out`, and the solver iterates until convergence or the
 *          maximum number of iterations is reached.
 */
int cgsstep(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x_loc, const params& p, cgsprec* pr, SolverOut* out)
{
    _MPI_ENV;

    int i, iter, info;

    int excnlim = -1;
    int retval = -1;

    stype ln = Alocal->n;
    gstype fn = Alocal->full_n;

    int s = p.sstep;

    out->resHist = MALLOC(vtype, p.itnlim, true);
    // out->solTime = 0.;

    vector<vtype>* u1_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* r_loc = Vector::init<vtype>(ln, true, true);

    vector<vtype>* u_loc = NULL;
    if (pr->ptype != PreconditionerType::NONE) {
        u_loc = Vector::init<vtype>(ln, true, true);
    }

    Vector::copyTo(r_loc, rhs_loc); // r_loc = rhs_loc
    vector<vtype>* w_loc = CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, NULL, 1., 0.);

    my_axpby(w_loc->val, ln, r_loc->val, -1., 1.); // r_loc = r_loc - w_loc

    vector<vtype>* sP1 = Vector::init<vtype>(ln * 2 * s, true, true);
    vector<vtype>* sP2 = Vector::init<vtype>(ln * 2 * s, true, true);

    Vector::fillWithValue(sP1, 0.);

    vector<vtype>* sCP1 = NULL;
    vector<vtype>* sCP2 = NULL;
    if (pr->ptype == PreconditionerType::NONE && p.ru_res) {
        sCP1 = Vector::init<vtype>(ln * s, true, true);
        sCP2 = Vector::init<vtype>(ln * s, true, true);

        Vector::fillWithValue(sCP1, 0.);
    }

    vectordh<vtype>* vm = Vector::initdh<vtype>(2 * s);
    vectordh<vtype>* alpha = Vector::initdh<vtype>(s);
    vectordh<vtype>* beta = Vector::initdh<vtype>(s * s);

    for (i = 0; i < s; i++) {
        alpha->val_[i] = 0.0;
    }

    for (i = 0; i < s * s; i++) {
        beta->val_[i] = 0.0;
    }

    vector<vtype>* W = Vector::init<vtype>(s * s, true, false);
    vector<vtype>* Wcopy = Vector::init<vtype>(s * s, true, false);

    for (i = 0; i < s * s; i++) {
        W->val[i] = 0.0;
    }

    for (i = 0; i < s * s; i++) {
        Wcopy->val[i] = 0.0;
    }

    vtype sone, sminusone, delta0, l2_norm;
    sone = 1.0;
    sminusone = -1.0;
    delta0 = 1.0;

    if (p.stop_criterion == 1) {
        delta0 = Vector::norm_MPI(h->cublas_h, r_loc);
    }

    if (ISMASTER) {
        out->resHist[0] = delta0;
    }

    for (iter = 0; iter < p.itnlim; iter++) {
        if (iter > 0 && info == 0 && p.ru_res && p.rec_res_int > 0 && iter % p.rec_res_int == 0) {
            if (p.dispnorm) {
                printf("recompute residual\n");
            }

            CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, w_loc, 1., 0.);

            Vector::copyTo(r_loc, rhs_loc);

            my_axpby(w_loc->val, ln, r_loc->val, -1., 1.);
        }

        if (info == 0 && iter % 2 == 0) { // even
            mpk(h, Alocal, r_loc, s, sP2, pr, u1_loc, p, out);

            computevm(sP2, s, r_loc, vm, pr->ptype != PreconditionerType::NONE);

            info = scalarWorkMO(vm, W, alpha, beta, s, iter);

            if (info != 0) {
                retval = info;
                break;
            }

            Vector::copydhToD<vtype>(alpha);
            Vector::copydhToD<vtype>(beta);

            if (p.ru_res && pr->ptype == PreconditionerType::NONE) {
                CHECK_DEVICE(cudaMemcpy(sCP2->val, sP2->val + ln, ln * s * sizeof(vtype), cudaMemcpyDeviceToDevice));
            }

            // sP2[:,1:s]=sP2[:,1:s]+sP1[:,1:s]*beta, x_loc=x_loc+sP2[:,1:s]*alpha
            mydmmv(sP1->val, ln, s, beta->val, s, s, sP2->val, alpha->val, x_loc->val, sone);

            if (p.ru_res) {
                if (pr->ptype != PreconditionerType::NONE) {
                    // sP2[:,s+1:2s]=sP2[:,s+1:2s]+sP1[:,s+1:2s]*beta, r_loc=r_loc-sP2[:,s+1:2s]*alpha
                    mydmmv(sP1->val + (s)*ln, ln, s, beta->val, s, s, sP2->val + (s)*ln, alpha->val, r_loc->val, sminusone);
                } else {
                    // sCP2[:,2:s+1]=sCP2[:,2:s+1]+sCP1[:,2:s+1]*beta, r_loc=r_loc-sCP2[:,2:s+1]*alpha
                    mydmmv(sCP1->val, ln, s, beta->val, s, s, sCP2->val, alpha->val, r_loc->val, sminusone);
                }
            } else {
                CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, w_loc, 1., 0.);
                Vector::copyTo(r_loc, rhs_loc);
                my_axpby(w_loc->val, ln, r_loc->val, -1., 1.);
            }
        } // end even

        if (info == 0 && iter % 2 == 1) { // odd
            mpk(h, Alocal, r_loc, s, sP1, pr, u1_loc, p, out);
            computevm(sP1, s, r_loc, vm, pr->ptype != PreconditionerType::NONE);
            info = scalarWorkMO(vm, W, alpha, beta, s, iter);

            if (info != 0) {
                retval = info;
                break;
            }

            Vector::copydhToD<vtype>(alpha);
            Vector::copydhToD<vtype>(beta);

            if (p.ru_res && pr->ptype == PreconditionerType::NONE) {
                CHECK_DEVICE(cudaMemcpy(sCP1->val, sP1->val + ln, ln * s * sizeof(vtype), cudaMemcpyDeviceToDevice));
            }

            // sP1[:,1:s]=sP1[:,1:s]+sP2[:,1:s]*beta, x_loc=x_loc+sP1[:,1:s]*alpha
            mydmmv(sP2->val, ln, s, beta->val, s, s, sP1->val, alpha->val, x_loc->val, sone);

            if (p.ru_res) {
                if (pr->ptype != PreconditionerType::NONE) {
                    // sP1[:,s+1:2s]=sP1[:,s+1:2s]+sP2[:,s+1:2s]*beta, r_loc=r_loc-sP1[:,s+1:2s]*alpha
                    mydmmv(sP2->val + (s)*ln, ln, s, beta->val, s, s, sP1->val + (s)*ln, alpha->val, r_loc->val, sminusone);
                } else {
                    // sCP1[:,1:s]=sCP1[:,1:s]+sCP2[:,1:s]*beta, r_loc=r_loc-sCP1[:,1:s]*alpha
                    mydmmv(sCP2->val, ln, s, beta->val, s, s, sCP1->val, alpha->val, r_loc->val, sminusone);
                }
            } else {
                CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, w_loc, 1., 0.);
                Vector::copyTo(r_loc, rhs_loc);
                my_axpby(w_loc->val, ln, r_loc->val, -1., 1.);
            }
        } // end odd

        l2_norm = Vector::norm_MPI(h->cublas_h, r_loc);

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

    Vector::free(r_loc);
    Vector::free(w_loc);
    Vector::free(sP1);
    Vector::free(sP2);
    Vector::free(W);
    Vector::free(Wcopy);
    Vector::freedh(vm);
    Vector::freedh(alpha);
    Vector::freedh(beta);
    if (pr->ptype == PreconditionerType::NONE && p.ru_res) {
        Vector::free(sCP1);
        Vector::free(sCP2);
    }
    Vector::free(u1_loc);
    if (pr->ptype != PreconditionerType::NONE) {
        Vector::free(u_loc);
    }

    return retval;
}

/**
 * @brief Solves the linear system \( A x = b \) using the Conjugate Gradient (CG) s-step method.
 * 
 * This function provides an interface to solve a linear system using the CG s-step method. It calls the `cgsstep`
 * function to perform the iterative solver steps and returns the final solution vector.
 * 
 * @param h Handle to CUDA and other resources.
 * @param Alocal The local CSR matrix used in the matrix-vector products.
 * @param rhs_loc The local right-hand side vector.
 * @param x_loc The local initial guess vector.
 * @param p The parameters that control the behavior of the solver, including solver settings and limits.
 * @param pr Preconditioner storage.
 * @param out The output structure that will store the results, including residual history and final solution.
 * 
 * @return The solution vector after performing the CG s-step method, stored in `out->sol_local`.
 * 
 * @note The final solution is returned in the output structure `out->sol_local`, and the result of the CG s-step
 *       method is captured in `out->retv`, which indicates the status of the solver.
 * 
 * @details The function acts as a wrapper to invoke the `cgsstep` function, providing an interface for the user
 *          to solve the system. The initial guess `x0_loc` is updated in-place with the final solution, and the
 *          output structure contains additional information, such as the residual history and exit conditions.
 */
vector<vtype>* solve_cgs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, cgsprec* pr, const params& p, SolverOut* out)
{
    out->retv = cgsstep(h, Alocal, rhs_loc, x0_loc, p, pr, out);

    // To be removed, should be used the solution returned in x0. Instead, should be returned out->retv (or void then check out->retv in the calling function)
    out->sol_local = Vector::init<vtype>(x0_loc->n, true, true);
    Vector::copyTo(out->sol_local, x0_loc);

    return out->sol_local;
}
