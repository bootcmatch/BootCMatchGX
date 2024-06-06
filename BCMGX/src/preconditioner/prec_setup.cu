#include "preconditioner/prec_setup.h"

void prec_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p)
{
    switch (pr->ptype) {
    case PreconditionerType::NONE:
        break;

    case PreconditionerType::L1_JACOBI:
        l1jacobi_setup(h, Alocal, pr, p);
        break;

    case PreconditionerType::BCMG:
        bcmg_setup(h, Alocal, pr, p);
        break;

    default:
        printf("Unhandled value for enum class PreconditionerType in %s().\n", __FUNCTION__);
        exit(1);
    }
}
