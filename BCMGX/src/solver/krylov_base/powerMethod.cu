#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "halo_communication/halo_communication.h"
#include "op/basic.h"
#include "preconditioner/prec_apply.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"
#include "utility/profiling.h"

vtype power_method(handles* h, CSR* Alocal, const InputParameters& ip, Preconditioner* pr, const CurrentParameters& cp)
{
    // This function returns just the maximum eigval of prec*A.

    _MPI_ENV;
    BEGIN_PROF(__FUNCTION__);

//    if (pr->type == PreconditionerType::NONE) {
//        DIE("*** Preconditioner required!\n");
//    }

    int i;
    stype ln = Alocal->n;
    vtype invynorm, mx, mx1, mx1_loc, err;

    vector<vtype>* y_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* t_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* w_loc = Vector::init<vtype>(ln, true, true);

    Vector::fillWithValue(y_loc, 1.0);
    Vector::fillWithValue(t_loc, 0.0); // necessario...

    err = 1.0;
    mx = 0.0;
    mx1 = 0.0;
    mx1_loc = 0.0;
    i = 0;
    while (i < ip.power_method_itnlim && err > ip.power_method_tol * fabs(mx)) {

        invynorm = 1. / Vector::norm_MPI(h->cublas_h, y_loc);
        cudaDeviceSynchronize();

        my_axpby(y_loc->val, ln, t_loc->val, invynorm, 0.);

        CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, t_loc, w_loc, 1., 0.);
        cudaDeviceSynchronize();
        Vector::fillWithValue(y_loc, 0.);
        cudaDeviceSynchronize();

        //prec_apply(h, Alocal, w_loc, y_loc, pr, ip);
        if (pr->type == PreconditionerType::NONE) {
            Vector::copyTo(y_loc, w_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);
        } else {
        	prec_apply(h, Alocal, w_loc, y_loc, pr, ip);
        }

        cudaDeviceSynchronize();
        mx1_loc = Vector::dot(h->cublas_h, t_loc, y_loc);
        cudaDeviceSynchronize();

        CHECK_MPI(MPI_Allreduce(&mx1_loc, &mx1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

        err = fabs(mx1 - mx);
        mx = mx1;

        i++;
        if (ISMASTER && i % 100 == 0) {
            printf("iter %d - error %.10lf  mx %.10lf\n", i, err, mx);
        }
    }
    if (ISMASTER) {
        printf("iter %d - error %.10lf  mx %.10lf\n", i, err, mx);
    }

    Vector::free(y_loc);
    Vector::free(t_loc);
    Vector::free(w_loc);

    END_PROF(__FUNCTION__);

    return mx;
}
