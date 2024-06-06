#pragma once

#include "utility/setting.h"
#include <stdio.h>

struct CSR;

class ProcessSelector {
private:
    bool use_row_shift;
    stype* rows_per_process;
    gstype* row_shift_per_process;
    gsstype* last_row_index_per_process;
    FILE* debug;
    int nprocs;

public:
    gstype row_shift;

    ProcessSelector(CSR* dlA, FILE* debug);
    ~ProcessSelector();
    int getProcessByRow(itype row);
    void setUseRowShift(bool use_row_shift);
};
