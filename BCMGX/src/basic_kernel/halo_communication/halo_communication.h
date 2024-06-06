#pragma once

#include "basic_kernel/halo_communication/local_permutation.h"
#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include <cub/cub.cuh>

__global__ void _getToSend_new(itype n, vtype* x, vtype* what_to_send, itype* to_send, itype shift);

__global__ void setReceivedWithMask(itype n, vtype* x, vtype* received, gstype* receive_map, itype shift);

halo_info haloSetup(CSR* A, CSR* R);
void halo_sync(halo_info hi, CSR* A, vector<vtype>* x, bool local_flag = false);
