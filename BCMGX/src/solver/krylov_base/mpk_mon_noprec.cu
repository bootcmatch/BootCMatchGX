#include "mpk_mon_noprec.h"

#include "halo_communication/halo_communication.h"
#include "op/CSRVector_product_adaptive_miniwarp_splitted.h"
#include "preconditioner/prec_apply.h"
#include "utility/profiling.h"

void mpk_mon_noprec(handles* h, CSR* Alocal, vector<vtype>* x_loc, int s, vector<vtype>* sP, Preconditioner* pr, vector<vtype>* y_loc, const InputParameters& ip, const CurrentParameters& cp, SolverOut* out)
{
// To be used for LSGS or CGSN when the monomial basis is used with preconditioner = NONE

    _MPI_ENV;
    BEGIN_PROF(__FUNCTION__);

    int i, pin, pout, ptemp;
    stype n = Alocal->n;

    vtype inv_l2_norm;

    vtype* xtemp = x_loc->val;
    vtype* ytemp = y_loc->val;

    if (cp.solver_type == SolverType::LSGS) {
        Vector::copyTo(sP, x_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);
        pin = 0;
        pout = s;
        for (i = 0; i < s-1; i++) {
            x_loc->val = sP->val + pin * n;
            y_loc->val = sP->val + pout * n;
            CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, y_loc, 1., 0.);
            ptemp = pin;
            pin = pout;
            pout = ptemp + 1;
            x_loc->val = sP->val + pin * n;
            y_loc->val = sP->val + pout * n;
            Vector::copyTo(y_loc, x_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);
            ptemp = pin;
            pin = pout;
            pout = ptemp + 1;
        }
        x_loc->val = sP->val + pin * n;
        y_loc->val = sP->val + pout * n;
        CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, y_loc, 1., 0.);

    }
    if (cp.solver_type == SolverType::CGSN) {
        Vector::copyTo(sP, x_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);
        inv_l2_norm = 1. / Vector::norm_MPI(h->cublas_h, x_loc);
        Vector::scale(h->cublas_h, sP, inv_l2_norm, 1);
        pin = 0;
        pout = s;
        for (i = 0; i < s-1; i++) {
            x_loc->val = sP->val + pin * n;
            y_loc->val = sP->val + pout * n;
            CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, y_loc, 1., 0.);
            ptemp = pin;
            pin = pout;
            pout = ptemp + 1;
            x_loc->val = sP->val + pin * n;
            y_loc->val = sP->val + pout * n;
            Vector::copyTo(y_loc, x_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);
            //Dividi y_loc per norm(y_loc)
            inv_l2_norm = 1. / Vector::norm_MPI(h->cublas_h, y_loc);
            Vector::scale(h->cublas_h, y_loc, inv_l2_norm, 1);

            ptemp = pin;
            pin = pout;
            pout = ptemp + 1;
        }
        x_loc->val = sP->val + pin * n;
        y_loc->val = sP->val + pout * n;
        CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, y_loc, 1., 0.);
    }

    x_loc->val = xtemp;
    y_loc->val = ytemp;

    END_PROF(__FUNCTION__);

}
