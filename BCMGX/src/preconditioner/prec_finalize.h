#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "preconditioner/prec_setup.h"
#include "utility/handles.h"

void prec_finalize(handles* h, CSR* Alocal, cgsprec* pr, const params& p);
