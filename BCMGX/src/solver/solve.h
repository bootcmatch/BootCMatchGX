#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "solver/SolverOut.h"

vector<vtype>* solve(CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x0, const params& p, SolverOut* out);
