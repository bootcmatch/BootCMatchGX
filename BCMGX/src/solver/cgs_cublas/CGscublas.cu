// This code implements the CG s-step method introduced in the following papers:
// Chronopoulos, A., Gear, C., s-step Iterative Methods for Symmetric Linear Systems, 
// J. Comput. Appl. Math., Vol. 25, N. 2, 1989. pages 153--168.
// Chronopoulos, A., Gear, C., On the Efficient Implementation of Preconditioned s-step Conjugate Gradient Methods on Multiprocessors with Memory Hierarchy, 
// Parallel Computing, Vol. 11, N. 1, 1989, pages 37--53.
// Here we use cublas for basic operations on small dense matrices. 

#include "CGscublas.h"

#include "halo_communication/halo_communication.h"
#include "op/LBfunctions.h"
#include "op/basic.h"
#include "op/scalarWorkMO.h"
#include "solver/krylov_base/mpk.h"
#include "utility/memory.h"
#include "utility/mpi.h"
#include "utility/profiling.h"

void computevm_cublas(handles* h, vector<vtype>* sP, int s, vector<vtype>* r_loc, vectordh<vtype>* vm, bool use_prec)
{
    _MPI_ENV;

    stype ln = r_loc->n;

    vectordh<vtype>* svm = Vector::initdh<vtype>(2 * s);

    vtype alpha = 1.0;
    vtype beta = 0.0;

    if (use_prec == 0) {
        cublasDgemv(h->cublas_h, CUBLAS_OP_T, ln, s + 1, &alpha, sP->val, ln, r_loc->val, 1, &beta, svm->val, 1);
        cublasDgemv(h->cublas_h, CUBLAS_OP_T, ln, s - 1, &alpha, sP->val + ln, ln, sP->val + ln * s, 1, &beta, svm->val + s + 1, 1);
    } else {
        cublasDgemv(h->cublas_h, CUBLAS_OP_T, ln, s, &alpha, sP->val, ln, r_loc->val, 1, &beta, svm->val, 1);
        cublasDgemv(h->cublas_h, CUBLAS_OP_T, ln, 1, &alpha, sP->val, ln, sP->val + ln * (2 * s - 1), 1, &beta, svm->val + s, 1);
        cublasDgemv(h->cublas_h, CUBLAS_OP_T, ln, s - 1, &alpha, sP->val + ln, ln, sP->val + ln * (2 * s - 1), 1, &beta, svm->val + s + 1, 1);
    }

    Vector::copydhToH<vtype>(svm);
    cudaDeviceSynchronize();

    BEGIN_PROF("ALLREDUCE");
    CHECK_MPI(MPI_Allreduce(svm->val_, vm->val_, 2 * s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    END_PROF("ALLREDUCE");

    Vector::freedh(svm);
}

int cgsstep_cublas(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, const params& p, cgsprec* pr, SolverOut* out)
{
    _MPI_ENV;

    int i, s, info;

    int excnlim = -1;
    int retval = -1;

    stype ln = Alocal->n;
    gstype fn = Alocal->full_n;

    s = p.sstep;

    out->resHist = MALLOC(vtype, p.itnlim, true);

    vector<vtype>* u1_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* u_loc = NULL;
    if (pr->ptype != PreconditionerType::NONE) {
        u_loc = Vector::init<vtype>(ln, true, true);
    }
    vector<vtype>* x_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* r_loc = Vector::init<vtype>(ln, true, true);

    Vector::copyTo(x_loc, x0_loc); // x_loc = x0_loc
    Vector::copyTo(r_loc, rhs_loc); // r_loc = rhs_loc

    vector<vtype>* w_loc = CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, NULL, 1., 0.);

    my_axpby(w_loc->val, ln, r_loc->val, -1., 1.); // r_loc = r_loc - w_loc

    vector<vtype>* sP1 = Vector::init<vtype>(ln * 2 * s, true, true);
    vector<vtype>* sP2 = Vector::init<vtype>(ln * 2 * s, true, true);

    vector<vtype>* sCP1 = NULL;
    vector<vtype>* sCP2 = NULL;
    if (pr->ptype == PreconditionerType::NONE && p.ru_res) {
        sCP1 = Vector::init<vtype>(ln * s, true, true);
        sCP2 = Vector::init<vtype>(ln * s, true, true);
    }

    vectordh<vtype>* vm = Vector::initdh<vtype>(2 * s);
    vectordh<vtype>* alpha = Vector::initdh<vtype>(s);
    vectordh<vtype>* beta = Vector::initdh<vtype>(s * s);

    vector<vtype>* W = Vector::init<vtype>(s * s, true, false);
    vector<vtype>* Wcopy = Vector::init<vtype>(s * s, true, false);

    for (i = 0; i < s; i++) {
        alpha->val_[i] = 0.0;
    }

    for (i = 0; i < s * s; i++) {
        beta->val_[i] = 0.0;
    }

    for (i = 0; i < s * s; i++) {
        W->val[i] = 0.0;
    }

    for (i = 0; i < s * s; i++) {
        Wcopy->val[i] = 0.0;
    }

    vtype l2_norm;
    vtype sone = 1.0;
    vtype sminusone = -1.0;
    vtype delta0 = 1.0;

    if (p.stop_criterion == 1) {
        delta0 = Vector::norm_MPI(h->cublas_h, r_loc);
    }

    if (ISMASTER) {
        out->resHist[0] = delta0;
    }

    int iter = 0;
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

            computevm_cublas(h, sP2, s, r_loc, vm, pr->ptype != PreconditionerType::NONE);

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
            cublasDgemm(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, s, &sone, sP1->val, ln, beta->val, s, &sone, sP2->val, ln);
            cublasDgemv(h->cublas_h, CUBLAS_OP_N, ln, s, &sone, sP2->val, ln, alpha->val, 1, &sone, x_loc->val, 1);

            if (p.ru_res) {
                if (pr->ptype != PreconditionerType::NONE) {
                    // sP2[:,s+1:2s]=sP2[:,s+1:2s]+sP1[:,s+1:2s]*beta, r_loc=r_loc-sP2[:,s+1:2s]*alpha
                    cublasDgemm(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, s, &sone, sP1->val + s * ln, ln, beta->val, s, &sone, sP2->val + s * ln, ln);
                    cublasDgemv(h->cublas_h, CUBLAS_OP_N, ln, s, &sminusone, sP2->val + s * ln, ln, alpha->val, 1, &sone, r_loc->val, 1);
                } else {
                    // sCP2[:,2:s+1]=sCP2[:,2:s+1]+sCP1[:,2:s+1]*beta, r_loc=r_loc-sCP2[:,2:s+1]*alpha
                    cublasDgemm(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, s, &sone, sCP1->val, ln, beta->val, s, &sone, sCP2->val, ln);
                    cublasDgemv(h->cublas_h, CUBLAS_OP_N, ln, s, &sminusone, sCP2->val, ln, alpha->val, 1, &sone, r_loc->val, 1);
                }
            } else {
                CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, w_loc, 1., 0.);
                // cudaDeviceSynchronize();

                Vector::copyTo(r_loc, rhs_loc);

                my_axpby(w_loc->val, ln, r_loc->val, -1., 1.);
            }

        } // end even

        if (info == 0 && iter % 2 == 1) { // odd
            mpk(h, Alocal, r_loc, s, sP1, pr, u1_loc, p, out);

            computevm_cublas(h, sP1, s, r_loc, vm, pr->ptype != PreconditionerType::NONE);

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
            cublasDgemm(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, s, &sone, sP2->val, ln, beta->val, s, &sone, sP1->val, ln);
            cublasDgemv(h->cublas_h, CUBLAS_OP_N, ln, s, &sone, sP1->val, ln, alpha->val, 1, &sone, x_loc->val, 1);

            if (p.ru_res) {
                if (pr->ptype != PreconditionerType::NONE) {
                    // sP1[:,s+1:2s]=sP1[:,s+1:2s]+sP2[:,s+1:2s]*beta, r_loc=r_loc-sP1[:,s+1:2s]*alpha
                    cublasDgemm(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, s, &sone, sP2->val + s * ln, ln, beta->val, s, &sone, sP1->val + s * ln, ln);
                    cublasDgemv(h->cublas_h, CUBLAS_OP_N, ln, s, &sminusone, sP1->val + s * ln, ln, alpha->val, 1, &sone, r_loc->val, 1);
                } else {
                    // sCP1[:,2:s+1]=sCP1[:,2:s+1]+sCP2[:,2:s+1]*beta, r_loc=r_loc-sCP1[:,2:s+1]*alpha
                    cublasDgemm(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, s, &sone, sCP2->val, ln, beta->val, s, &sone, sCP1->val, ln);
                    cublasDgemv(h->cublas_h, CUBLAS_OP_N, ln, s, &sminusone, sCP1->val, ln, alpha->val, 1, &sone, r_loc->val, 1);
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

    out->sol_local = Vector::init<vtype>(ln, true, true);

    Vector::copyTo(out->sol_local, x_loc);

    Vector::free(r_loc);
    // Vector::free(rhs_loc);
    Vector::free(x_loc);
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

vector<vtype>* solve_cgs_cublas(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, cgsprec* pr, const params& p, SolverOut* out)
{
    out->retv = cgsstep_cublas(h, Alocal, rhs_loc, x0_loc, p, pr, out);
    return out->sol_local;
}
