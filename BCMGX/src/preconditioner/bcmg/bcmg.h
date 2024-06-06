#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/PrecOut.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

struct cgsprec;

struct BcmgData {
    bootBuildData* bootamg_data = NULL;
    boot* boot_amg = NULL;
    applyData* amg_cycle = NULL;
    hierarchy* H = NULL;
};

void bcmg_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p);
void bcmg_apply(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x, cgsprec* pr, const params& p, PrecOut* out);
void bcmg_finalize(handles* h, CSR* Alocal, cgsprec* pr, const params& p);
