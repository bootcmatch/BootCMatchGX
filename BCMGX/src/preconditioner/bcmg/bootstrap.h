#pragma once

#include "config/Params.h"
#include "preconditioner/bcmg/AMG.h"
#include "utility/handles.h"

namespace Bootstrap {

void innerIterations(handles* h, bootBuildData* bootamg_data, boot* boot_amg, applyData* amg_cycle);

boot* bootstrap(handles* h, bootBuildData* bootamg_data, applyData* apply_data, const params& p);

}
