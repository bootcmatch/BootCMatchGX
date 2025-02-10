#include "pipelinedCGs.h"

#include "halo_communication/halo_communication.h"
#include "op/LBfunctions.h"
#include "op/basic.h"
#include "op/scalarWorkMO.h"
#include "solver/krylov_base/mpk.h"
#include "utility/memory.h"
#include "utility/mpi.h"
#include "utility/profiling.h"
#include "utility/utils.h"

void pipecomputevm(vector<vtype>* sUR, vector<vtype>* sUQ, int s, vector<vtype>* r_loc, vectordh<vtype>* vm, vectordh<vtype>* svm, bool use_prec)
{
    _MPI_ENV;

    stype ln = r_loc->n;

    if (use_prec) {
        mynddot(s, ln, sUR->val, r_loc->val, svm->val);
        mynddot(s, ln, sUR->val, sUQ->val + ln * (s - 1), svm->val + s);
    } else {
        mynddot(s, ln, sUR->val, r_loc->val, svm->val);
        mynddot(1, ln, sUQ->val + ln * (s - 1), r_loc->val, svm->val + s);
        mynddot(s - 1, ln, sUQ->val, sUQ->val + ln * (s - 1), svm->val + s + 1);
    }

    cudaDeviceSynchronize();

    Vector::copydhToH<vtype>(svm);
}

int pipelinedcgsstep(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, const params& p, cgsprec* pr, SolverOut* o)
{
    _MPI_ENV;

    int i, j, iter, info;

    int excnlim = -1;
    int retval = -1;

    stype ln = Alocal->n;
    gstype fn = Alocal->full_n;

    int s = p.sstep;

    o->resHist = MALLOC(vtype, p.itnlim, true);

    vector<vtype>* u_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* u1_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* x_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* r_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* sP2 = Vector::init<vtype>(ln * (2 * s), true, true);
    vector<vtype>* sT = Vector::init<vtype>(ln * (2 * s), true, true);
    vector<vtype>* W = Vector::init<vtype>(s * s, true, false);
    vector<vtype>* Wcopy = Vector::init<vtype>(s * s, true, false);
    vector<vtype>* sUR1m = Vector::init<vtype>(ln * s * (s + 1), true, true);
    vector<vtype>* sUR2m = Vector::init<vtype>(ln * s * (s + 1), true, true);
    vector<vtype>* sUQ1m = Vector::init<vtype>(ln * s * (s + 1), true, true);
    vector<vtype>* sUQ2m = Vector::init<vtype>(ln * s * (s + 1), true, true);
    vector<vtype>* sUR1 = Vector::init<vtype>(ln * s, true, true);
    vector<vtype>* sUR2 = Vector::init<vtype>(ln * s, true, true);
    vector<vtype>* sUQ1 = Vector::init<vtype>(ln * s, true, true);
    vector<vtype>* sUQ2 = Vector::init<vtype>(ln * s, true, true);
    vector<vtype>* AMs_loc = Vector::init<vtype>(ln, true, true);

    Vector::copyTo(x_loc, x0_loc); // x_loc = x0_loc
    Vector::copyTo(r_loc, rhs_loc); // r_loc = rhs_loc

    vector<vtype>* w_loc = CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, NULL, 1., 0.);

    my_axpby(w_loc->val, ln, r_loc->val, -1.0, 1.0);

    vectordh<vtype>* vm = Vector::initdh<vtype>(2 * s);
    vectordh<vtype>* alpha = Vector::initdh<vtype>(s);
    vectordh<vtype>* beta = Vector::initdh<vtype>(s * s);
    vectordh<vtype>* svm = Vector::initdh<vtype>(2 * s);

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

    Vector::fillWithValue(sUR1m, 0.);
    Vector::fillWithValue(sUR2m, 0.);
    Vector::fillWithValue(sUQ1m, 0.);
    Vector::fillWithValue(sUQ2m, 0.);
    Vector::fillWithValue(sUR1, 0.);
    Vector::fillWithValue(sUR2, 0.);
    Vector::fillWithValue(sUQ1, 0.);
    Vector::fillWithValue(sUQ2, 0.);

    vtype* temp;

    mpk(h, Alocal, r_loc, s, sP2, pr, u1_loc, p, o);

    CHECK_DEVICE(cudaMemcpy(sUR2m->val, sP2->val, s * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
    if (pr->ptype != PreconditionerType::NONE) {
        CHECK_DEVICE(cudaMemcpy(sUQ2m->val, sP2->val + s * ln, s * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
    } else {
        CHECK_DEVICE(cudaMemcpy(sUQ2m->val, sP2->val + ln, s * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
    }

    vtype sone, sminusone, delta0, l2_norm;
    sone = 1.0;
    sminusone = -1.0;
    delta0 = 1.0;

    MPI_Request request;

    if (p.stop_criterion == 1) {
        delta0 = Vector::norm_MPI(h->cublas_h, r_loc);
    }

    if (ISMASTER) {
        o->resHist[0] = delta0;
    }

    for (iter = 0; iter < p.itnlim; iter++) {
        if (iter % 2 == 0) { // even
            if (iter > 0) {
                for (j = 1; j < s + 1; j++) {
                    mydgemm(sUR2m->val + j * s * ln, ln, s, beta->val, s, s, sUR1m->val + j * s * ln);
                    mydgemm(sUQ2m->val + j * s * ln, ln, s, beta->val, s, s, sUQ1m->val + j * s * ln);
                }

                for (j = 0; j < s; j++) {
                    // UR2m[j] = UR1 - UR1m[j] * alpha
                    mydgemv2v(sUR1m->val + (j + 1) * s * ln, ln, s, alpha->val, sUR1->val + j * ln, sminusone, sUR2m->val + j * ln);
                    // UQ2m[j] = UQ1 - UQ1m[j] * alpha
                    mydgemv2v(sUQ1m->val + (j + 1) * s * ln, ln, s, alpha->val, sUQ1->val + j * ln, sminusone, sUQ2m->val + j * ln);
                }
            }

            pipecomputevm(sUR2m, sUQ2m, s, r_loc, vm, svm, pr->ptype != PreconditionerType::NONE);

            BEGIN_PROF("ALLREDUCE");
            CHECK_MPI(MPI_Iallreduce(svm->val_, vm->val_, 2 * s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request));
            END_PROF("ALLREDUCE");

            temp = AMs_loc->val;
            AMs_loc->val = sUQ2m->val + ln * (s - 1);

            mpk(h, Alocal, AMs_loc, s, sT, pr, u1_loc, p, o);

            AMs_loc->val = temp;

            BEGIN_PROF("MPIWAIT");
            CHECK_MPI(MPI_Wait(&request, MPI_STATUS_IGNORE));
            END_PROF("MPIWAIT");

            info = scalarWorkMO(vm, W, alpha, beta, s, iter);

            if (info != 0) {
                retval = info;
                break;
            }

            Vector::copydhToD<vtype>(alpha);
            Vector::copydhToD<vtype>(beta);

            for (j = 1; j < s + 1; j++) {
                CHECK_DEVICE(cudaMemcpy(sUR2m->val + j * (s * ln), sUR2m->val + j * ln, (s - j) * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
                CHECK_DEVICE(cudaMemcpy(sUR2m->val + j * (s * ln) + (s - j) * ln, sT->val, j * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
                CHECK_DEVICE(cudaMemcpy(sUQ2m->val + j * (s * ln), sUQ2m->val + j * ln, (s - j) * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
                if (pr->ptype != PreconditionerType::NONE) {
                    CHECK_DEVICE(cudaMemcpy(sUQ2m->val + j * (s * ln) + (s - j) * ln, sT->val + s * ln, j * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
                } else {
                    CHECK_DEVICE(cudaMemcpy(sUQ2m->val + j * (s * ln) + (s - j) * ln, sT->val + ln, j * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
                }
            }

            // save sURm[0], sUQm[0] in sUR, sUQ
            CHECK_DEVICE(cudaMemcpy(sUR2->val, sUR2m->val, s * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
            CHECK_DEVICE(cudaMemcpy(sUQ2->val, sUQ2m->val, s * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));

            // UR2m[0] = UR2m[0] + UR1m[0] * beta, x_loc = x_loc + UR2m[0] * alpha
            mydmmv(sUR1m->val, ln, s, beta->val, s, s, sUR2m->val, alpha->val, x_loc->val, sone);
            // UQ2m[0] = UQ2m[0] + UQ1m[0] * beta, r_loc = r_loc - UQ2m[0] * alpha
            mydmmv(sUQ1m->val, ln, s, beta->val, s, s, sUQ2m->val, alpha->val, r_loc->val, sminusone);
        } // end even

        if (iter % 2 == 1) {
            for (j = 1; j < s + 1; j++) {
                // UR2m[j] = UR2m[j] + UR1m[j] * beta
                mydgemm(sUR1m->val + j * s * ln, ln, s, beta->val, s, s, sUR2m->val + j * s * ln);
                // UQ2m[j] = UQ2m[j] + UQ1m[j] * beta
                mydgemm(sUQ1m->val + j * s * ln, ln, s, beta->val, s, s, sUQ2m->val + j * s * ln);
            }

            for (j = 0; j < s; j++) {
                mydgemv2v(sUR2m->val + (j + 1) * s * ln, ln, s, alpha->val, sUR2->val + j * ln, sminusone, sUR1m->val + j * ln);
                mydgemv2v(sUQ2m->val + (j + 1) * s * ln, ln, s, alpha->val, sUQ2->val + j * ln, sminusone, sUQ1m->val + j * ln);
            }

            pipecomputevm(sUR1m, sUQ1m, s, r_loc, vm, svm, pr->ptype != PreconditionerType::NONE);

            BEGIN_PROF("ALLREDUCE");
            CHECK_MPI(MPI_Iallreduce(svm->val_, vm->val_, 2 * s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request));
            END_PROF("ALLREDUCE");

            temp = AMs_loc->val;
            AMs_loc->val = sUQ1m->val + ln * (s - 1);

            mpk(h, Alocal, AMs_loc, s, sT, pr, u1_loc, p, o);

            AMs_loc->val = temp;

            BEGIN_PROF("MPIWAIT");
            CHECK_MPI(MPI_Wait(&request, MPI_STATUS_IGNORE));
            END_PROF("MPIWAIT");

            info = scalarWorkMO(vm, W, alpha, beta, s, iter);

            if (info != 0) {
                retval = info;
                break;
            }

            Vector::copydhToD<vtype>(alpha);
            Vector::copydhToD<vtype>(beta);

            for (j = 1; j < s + 1; j++) {
                CHECK_DEVICE(cudaMemcpy(sUR1m->val + j * (s * ln), sUR1m->val + j * ln, (s - j) * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
                CHECK_DEVICE(cudaMemcpy(sUR1m->val + j * (s * ln) + (s - j) * ln, sT->val, j * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));

                CHECK_DEVICE(cudaMemcpy(sUQ1m->val + j * (s * ln), sUQ1m->val + j * ln, (s - j) * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
                if (pr->ptype != PreconditionerType::NONE) {
                    CHECK_DEVICE(cudaMemcpy(sUQ1m->val + j * (s * ln) + (s - j) * ln, sT->val + s * ln, j * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
                } else {
                    CHECK_DEVICE(cudaMemcpy(sUQ1m->val + j * (s * ln) + (s - j) * ln, sT->val + ln, j * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
                }
            }

            CHECK_DEVICE(cudaMemcpy(sUR1->val, sUR1m->val, s * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));
            CHECK_DEVICE(cudaMemcpy(sUQ1->val, sUQ1m->val, s * ln * sizeof(vtype), cudaMemcpyDeviceToDevice));

            mydmmv(sUR2m->val, ln, s, beta->val, s, s, sUR1m->val, alpha->val, x_loc->val, sone);
            mydmmv(sUQ2m->val, ln, s, beta->val, s, s, sUQ1m->val, alpha->val, r_loc->val, sminusone);
        } // end odd

        l2_norm = Vector::norm_MPI(h->cublas_h, r_loc);

        if (ISMASTER) {
            if (p.dispnorm) {
                printf("%d) r norm: %.10lf\n", iter, l2_norm);
            }
            o->resHist[iter + 1] = l2_norm;
        }

        if (l2_norm < p.rtol * delta0) {
            excnlim = 0;
            retval = 0;
            break;
        }

    } // end iter

    // Even if max_iter exceeded return the current status
    if (ISMASTER) {
        o->del0 = delta0;
        o->full_n = fn;
        o->local_n = ln;
        o->niter = iter + 1 + excnlim;
        o->exitRes = l2_norm;
    }

    o->sol_local = Vector::init<vtype>(ln, true, true);

    Vector::copyTo(o->sol_local, x_loc);

    Vector::free(r_loc);
    // Vector::free(rhs_loc);
    Vector::free(w_loc);
    Vector::free(x_loc);
    Vector::free(sP2);
    Vector::free(sT);
    Vector::free(sUR1);
    Vector::free(sUR2);
    Vector::free(sUR1m);
    Vector::free(sUR2m);
    Vector::free(sUQ1);
    Vector::free(sUQ2);
    Vector::free(sUQ1m);
    Vector::free(sUQ2m);
    Vector::free(W);
    Vector::free(Wcopy);
    Vector::free(u1_loc);
    if (pr->ptype != PreconditionerType::NONE) {
        Vector::free(u_loc);
    }

    Vector::freedh(vm);
    Vector::freedh(svm);
    Vector::freedh(alpha);
    Vector::freedh(beta);

    return retval;
}

vector<vtype>* solve_pipelined_cgs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, cgsprec* pr, const params& p, SolverOut* out)
{
#if USECUDAPROFILER
    if (myid == 0) {
        cudaProfilerStart();
    }
#endif

    out->retv = pipelinedcgsstep(h, Alocal, rhs_loc, x0_loc, p, pr, out);

#if USECUDAPROFILER
    if (myid == 0) {
        cudaProfilerStop();
    }
#endif

    return out->sol_local;
}
