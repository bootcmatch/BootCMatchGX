#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/PrecOut.h"
#include "utility/handles.h"

struct cgsprec;

struct L1JacobiData {
    vector<vtype>* pl1j = NULL;
    vector<vtype>* w_loc = NULL;
    vector<vtype>* rcopy_loc = NULL;
};

void l1jacobi_iter(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, vector<vtype>* pl1j, vector<vtype>* rcopy_loc, vector<vtype>* w_loc);

void l1jacobi_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p);
void l1jacobi_apply(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, cgsprec* pr, const params& p, PrecOut* out);
void l1jacobi_finalize(handles* h, CSR* Alocal, cgsprec* pr, const params& p);
