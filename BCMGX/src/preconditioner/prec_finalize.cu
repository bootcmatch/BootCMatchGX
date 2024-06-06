#include "preconditioner/bcmg/bcmg.h"
#include "preconditioner/l1jacobi/l1jacobi.h"
#include "preconditioner/prec_finalize.h"

void prec_finalize(handles* h, CSR* Alocal, cgsprec* pr, const params& p)
{
    switch (pr->ptype) {
    case PreconditionerType::NONE:
        break;

    case PreconditionerType::L1_JACOBI:
        l1jacobi_finalize(h, Alocal, pr, p);
        break;

    case PreconditionerType::BCMG:
        bcmg_finalize(h, Alocal, pr, p);
        break;

    default:
        printf("Unhandled value for enum class PreconditionerType in %s().\n", __FUNCTION__);
        exit(1);
    }
}
