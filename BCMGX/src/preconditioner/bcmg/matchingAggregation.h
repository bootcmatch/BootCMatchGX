#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

hierarchy* adaptiveCoarsening(handles* h, buildData* amg_data, const params& p);

void relaxPrepare(handles* h, int level, CSR* A, hierarchy* hrrch, buildData* amg_data);
