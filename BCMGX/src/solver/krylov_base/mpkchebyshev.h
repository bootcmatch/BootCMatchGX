#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/prec_setup.h"
#include "solver/SolverOut.h"
#include "utility/handles.h"

void mpkc(handles* h, CSR* Alocal, vector<vtype>* x_loc, int s, vector<vtype>* sP, Preconditioner* pr, vector<vtype>* y_loc,
    vectordh<vtype>* abg, vector<vtype>* zp, const InputParameters& ip, const CurrentParameters& cp, SolverOut* out);

void copyToCM(vtype maxval);
