#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/prec_apply.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"

vector<vtype>* solve_cg_hs(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, cgsprec* pr, const params& p, SolverOut* out);
