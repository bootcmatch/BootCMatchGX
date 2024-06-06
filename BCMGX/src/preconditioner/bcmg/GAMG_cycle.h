#pragma once

#define RTOL 0.25

#include "datastruct/vector.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"
#include "utility/setting.h"

namespace GAMGcycle {
extern vector<vtype>* Res_buffer;

void initContext(int n);

__inline__ void setBufferSize(itype n);

void freeContext();
}

void GAMG_cycle(handles* h, int k, bootBuildData* bootamg_data, boot* boot_amg, applyData* amg_cycle, vectorCollection<vtype>* Rhs, vectorCollection<vtype>* Xtent, vectorCollection<vtype>* Xtent_2, int l /*, int coarsesolver_type*/);
