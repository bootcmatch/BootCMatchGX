#pragma once

#include "config/Params.h"
#include "datastruct/vector.h"
#include "preconditioner/PrecOut.h"

// struct out_cgs {
struct SolverOut {
    int retv = 0;
    vector<vtype>* sol_local;
    vtype* resHist;
    vtype exitRes;
    vtype del0;
    int niter;
    float solTime;
    // int nproc;
    gstype full_n;
    stype local_n;

    PrecOut precOut {};
};

void dump(const char* filename, const params& p, SolverOut* out);
