#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/bcmg/bcmg.h"
#include "preconditioner/l1jacobi/l1jacobi.h"
#include "utility/handles.h"

struct cgsprec {
    PreconditionerType ptype; // if > 0 use ptype prec

    // l1-Jacobi
    L1JacobiData l1jacobi;

    // BCMG
    BcmgData bcmg;
};

void prec_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p);
