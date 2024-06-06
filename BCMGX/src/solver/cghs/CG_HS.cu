#include "solver/cghs/CG_HS.h"

#include "basic_kernel/halo_communication/halo_communication.h"
#include "op/basic.h"
#include "preconditioner/prec_apply.h"
#include "utility/handles.h"
#include "utility/timing.h"
#include "utility/utils.h"

#define CHECKSOLUTION 1
#define VEC_DIM 3
#define USECUDAPROFILER 0

#if USECUDAPROFILER
#include <cuda_profiler_api.h>
#endif

int cg_hs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, const params& p, cgsprec* pr, SolverOut* out)
{
    _MPI_ENV;

    int excnlim = -1;
    int retval = -1;

    stype ln = Alocal->n;
    gstype fn = Alocal->full_n;

    out->resHist = (vtype*)Malloc(p.itnlim * sizeof(vtype));
    out->solTime = 0.;

    vector<vtype>* s_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* x2_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* p2_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* r2_loc = Vector::init<vtype>(ln, true, true);

    vector<vtype>* u2_loc = NULL;
    if (pr->ptype != PreconditionerType::NONE) {
        u2_loc = Vector::init<vtype>(ln, true, true);
    }

    // x2_loc = x0_loc
    Vector::copyTo(x2_loc, x0_loc);
    // r2_loc = rhs_loc
    Vector::copyTo(r2_loc, rhs_loc);

    vector<vtype>* w_loc = CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x2_loc, NULL, 1., 0.);
    cudaDeviceSynchronize();
	my_axpby(w_loc->val, ln, r2_loc->val, -1.0, 1.0);

    if (pr->ptype != PreconditionerType::NONE) {
        // u2_loc=prec*r2_loc
        Vector::fillWithValue(u2_loc, 0.);
        prec_apply(h, Alocal, r2_loc, u2_loc, pr, p, &out->precOut);
        // p2_loc=u2_loc
        Vector::copyTo(p2_loc, u2_loc);
    } else {
        // p2_loc=r2_loc
        Vector::copyTo(p2_loc, r2_loc);
    }

    vectordh<vtype>* urpsm = NULL;
    urpsm = Vector::initdh<vtype>(VEC_DIM);
    vectordh<vtype>* urps = NULL;
    urps = Vector::initdh<vtype>(VEC_DIM);
    vectordh<vtype>* ab = NULL;
    ab = Vector::initdh<vtype>(2); // alpha, beta

    vtype f, delta0, l2_norm;
    delta0 = 1.0;

    if (p.stop_criterion == 1) {
        delta0 = Vector::norm_MPI(h->cublas_h, r2_loc);
    }

    if (ISMASTER) {
        out->resHist[0] = delta0;
    }

    if (ISMASTER) {
        TIME::start();
    }

    int iter = 0;
    for (iter = 0; iter < p.itnlim; iter++) {

		if (iter > 0) {
			if (pr->ptype != PreconditionerType::NONE) {
				Vector::fillWithValue(u2_loc, 0.);
				prec_apply(h, Alocal, r2_loc, u2_loc, pr, p, &out->precOut);
				myddot(ln, u2_loc->val, r2_loc->val, urpsm->val + 2);
			} else {
				myddot(ln, r2_loc->val, r2_loc->val, urpsm->val + 2);
			}
			Vector::copydhToH<vtype>(urpsm);
			CHECK_MPI(MPI_Allreduce(urpsm->val_, urps->val_, VEC_DIM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
			ab->val_[1] = urps->val_[2] / urps->val_[0];
			f = ab->val_[1];
			if (pr->ptype != PreconditionerType::NONE) {
				my_axpby(u2_loc->val, ln, p2_loc->val, 1.0, f);
			} else {
				my_axpby(r2_loc->val, ln, p2_loc->val, 1.0, f);
			}
		}

		CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, p2_loc, s_loc, 1., 0.);
		cudaDeviceSynchronize();

		if (pr->ptype != PreconditionerType::NONE) {
			// ur=u2_loc.r2_loc
			myddot(ln, u2_loc->val, r2_loc->val, urpsm->val);
		} else {
			// ur=r2_loc.r2_loc
			myddot(ln, r2_loc->val, r2_loc->val, urpsm->val);
		}

		// ps=p2_loc.s_loc
		myddot(ln, p2_loc->val, s_loc->val, urpsm->val + 1);
		Vector::copydhToH<vtype>(urpsm);
		CHECK_MPI(MPI_Allreduce(urpsm->val_, urps->val_, VEC_DIM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
		ab->val_[0] = urps->val_[0] / urps->val_[1]; // alpha

		f = ab->val_[0];
		// x2=x2+alpha*p2
		my_axpby(p2_loc->val, ln, x2_loc->val, f, 1.0);
		// r2=r2-alpha*s
		my_axpby(s_loc->val, ln, r2_loc->val, -f, 1.0);

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
        out->solTime = TIME::stop();
        out->del0 = delta0;
        out->full_n = fn;
        out->local_n = ln;
        out->niter = iter + 1 + excnlim;
        out->exitRes = l2_norm;
    }
    out->sol_local = Vector::init<vtype>(ln, true, true);

    Vector::copyTo(out->sol_local, x2_loc);


#if CHECKSOLUTION
    w_loc = CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, out->sol_local, NULL, 1., 0.);
    cudaDeviceSynchronize();
    Vector::copyTo(r2_loc, rhs_loc);
	my_axpby(w_loc->val, ln, r2_loc->val, -1.0, 1.0);
    l2_norm = Vector::norm_MPI(h->cublas_h, r2_loc);
    if (ISMASTER) {
        printf("CHECK - residual norm computed as Ax_loc-b: %.10lf\n", l2_norm);
    }
#endif

    TRACE_REACHED_LINE();

    Vector::free(rhs_loc);
    Vector::free(w_loc);
    Vector::free(s_loc);
    Vector::free(r2_loc);
    Vector::free(x2_loc);
    if (pr->ptype != PreconditionerType::NONE) {
        Vector::free(u2_loc);
    }
    return retval;
}

vector<vtype>* solve_cg_hs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, cgsprec* pr, const params& p, SolverOut* out)
{
    _MPI_ENV;

#if USECUDAPROFILER
    if (myid == 0) {
        cudaProfilerStart();
    }
#endif

    int retv = cg_hs(h, Alocal, rhs_loc, x0_loc, p, pr, out);

#if USECUDAPROFILER
    if (myid == 0) {
        cudaProfilerStop();
    }
#endif

    return out->sol_local;
}
