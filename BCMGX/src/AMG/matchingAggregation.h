#pragma once

#include <mpi.h>
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "cub/device/device_reduce.cuh"

#include "utility/setting.h"
#include "spmspmMGPU/spspmpi.h"
#include "AMG.h"
#include "matrix/scalar.h"
#include "matrix/CSR.h"
#include "matrix/vector.h"
#include "solver/solutionAggregator.h"
#include "utility/utils.h"

hierarchy* adaptiveCoarsening(handles *h, buildData *amg_data, const params p);

void relaxPrepare(handles *h, int level, CSR *A, hierarchy *hrrch, buildData *amg_data, int force_relax_type);
