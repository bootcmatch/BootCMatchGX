#include "preconditioner/prec_apply.h"
#include "utility/metrics.h"
#include "utility/mpi.h"
#include "utility/timing.h"

void prec_apply(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, cgsprec* pr, const params& p, PrecOut* out)
{
    _MPI_ENV;
    
#if DETAILED_TIMING
    if (ISMASTER) {
        TIME::start();
    }
#endif

    switch (pr->ptype) {
    case PreconditionerType::NONE:
        break;

    case PreconditionerType::BCMG:
        bcmg_apply(h, Alocal, r_loc, u_loc, pr, p, out);
        break;

    case PreconditionerType::L1_JACOBI:
        l1jacobi_apply(h, Alocal, r_loc, u_loc, pr, p, out);
        break;

    default:
        printf("Unhandled value for enum class PreconditionerType in %s().\n", __FUNCTION__);
        exit(0);
    }

#if DETAILED_TIMING
    if (ISMASTER) {
        cudaDeviceSynchronize();
        TOTAL_PRECAPPLY_TIME += TIME::stop();
    }
#endif
}
