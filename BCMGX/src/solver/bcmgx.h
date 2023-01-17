#pragma once

#include "basic_kernel/matrix/CSR.h"
#include "prec_setup/matchingAggregation.h"

vector<vtype>* bcmgx(CSR *Alocal, vector<vtype> *rhs, const params p, bool precondition_flag);
