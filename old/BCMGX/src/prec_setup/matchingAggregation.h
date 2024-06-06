#pragma once

#include <mpi.h>
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "cub/device/device_reduce.cuh"

#include "utility/setting.h"
#include "prec_setup/spmspmMGPU/spspmpi.h"
#include "AMG.h"
#include "basic_kernel/matrix/scalar.h"
#include "basic_kernel/matrix/CSR.h"
#include "basic_kernel/matrix/vector.h"
#include "basic_kernel/halo_communication/halo_communication.h"
#include "utility/utils.h"

hierarchy* adaptiveCoarsening(handles *h, buildData *amg_data, const params p, bool precondition_flag);

void relaxPrepare(handles *h, int level, CSR *A, hierarchy *hrrch, buildData *amg_data, int force_relax_type);
