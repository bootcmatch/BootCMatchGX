#pragma once

#include "datastruct/vector.h"
#include "preconditioner/bcmg/AMG.h"

struct BcmgPreconditionContext {
    vectorCollection<vtype>* RHS_buffer;

    vectorCollection<vtype>* Xtent_buffer_local;
    vectorCollection<vtype>* Xtent_buffer_2_local;

    hierarchy* hrrch;
    int max_level_nums;
    itype* max_coarse_size;
};

namespace Bcmg {

extern BcmgPreconditionContext context;

void initPreconditionContext(hierarchy* hrrch);

void setHrrchBufferSize(hierarchy* hrrch);

void freePreconditionContext();
}
