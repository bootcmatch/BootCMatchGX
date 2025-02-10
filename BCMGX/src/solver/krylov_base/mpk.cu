#include "mpk.h"

#include "halo_communication/halo_communication.h"
#include "op/CSRVector_product_adaptive_miniwarp_splitted.h"
#include "preconditioner/prec_apply.h"
#include "utility/profiling.h"

#define USE_SPLITTED_PRODUCT 0

void mpk(handles* h, CSR* Alocal, vector<vtype>* x_loc, int s, vector<vtype>* sP, cgsprec* pr, vector<vtype>* y_loc, const params& p, SolverOut* out)
{
    _MPI_ENV;
    BEGIN_PROF(__FUNCTION__);

    int i, pin, pout, ptemp;
    stype n = Alocal->n;

    vtype* xtemp = x_loc->val;
    vtype* ytemp = y_loc->val;

    if (pr->ptype == PreconditionerType::NONE) {
        Vector::copyTo(sP, x_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);

        for (i = 0; i < s; i++) {
            y_loc->val = sP->val + (i + 1) * n;
            CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, y_loc, 1., 0.);
            x_loc->val = y_loc->val;
        }
    } else {
        y_loc->val = sP->val;

        Vector::fillWithValue(y_loc, 0.);

        cudaDeviceSynchronize();
        prec_apply(h, Alocal, x_loc, y_loc, pr, p, &out->precOut);
        // cudaDeviceSynchronize();

        pin = 0;
        pout = s;
        for (i = 0; i < 2 * s - 1; i++) {
            x_loc->val = sP->val + pin * n;
            y_loc->val = sP->val + pout * n;
            if (i % 2 == 0) {
                // y=A x, sP[pout] = A * sP[pin]

                BEGIN_PROF("spmv_new_or_splitted");
#if USE_SPLITTED_PRODUCT
                if (nprocs > 1) {
                    if (pr->ptype != PreconditionerType::BCMG) {
                        halo_sync(Alocal->halo, Alocal, x_loc, true);
                    } else {
                        for (int k = 0; k < pr->bcmg.boot_amg->n_hrc; k++) {
                            halo_sync(pr->bcmg.boot_amg->H_array[k]->A_array[0]->halo, pr->bcmg.boot_amg->H_array[k]->A_array[0], x_loc, true);
                        }
                    }
                }
                if (s == 1) {
                    CSRm::CSRVector_product_adaptive_miniwarp_new(Alocal, x_loc, y_loc, 1., 0.);
                } else {
                    CSRVector_product_adaptive_miniwarp_splitted(Alocal, x_loc, y_loc, i / 2, s, 1., 0.);
                }
                cudaDeviceSynchronize();
#else
                CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, y_loc, 1., 0.);
#endif
                END_PROF("spmv_new_or_splitted");
            } else {
                Vector::fillWithValue(y_loc, 0.);
                cudaDeviceSynchronize();
                prec_apply(h, Alocal, x_loc, y_loc, pr, p, &out->precOut);
                // cudaDeviceSynchronize();
            }
            ptemp = pin;
            pin = pout;
            pout = ptemp + 1;
        }
    }

    x_loc->val = xtemp;
    y_loc->val = ytemp;

    END_PROF(__FUNCTION__);
}
