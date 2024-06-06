#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/prec_apply.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"

vector<vtype>* solve_fcg(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x0, cgsprec* pr, const params& p, SolverOut* out);
