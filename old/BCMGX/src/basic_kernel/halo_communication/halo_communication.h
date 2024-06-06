#pragma once

#include <cub/cub.cuh>
#include "basic_kernel/matrix/scalar.h"
#include "basic_kernel/matrix/CSR.h"
#include "basic_kernel/matrix/vector.h"
#include "basic_kernel/halo_communication/local_permutation.h"


__global__
void _getToSend(itype n, vtype *x, vtype *what_to_send, itype *to_send);

__global__
void _getToSend_new(itype n, vtype *x, vtype *what_to_send, itype *to_send, itype shift);


__global__
void setReceivedWithMask(itype n, vtype *x, vtype *received, gstype *receive_map, itype shift);

__global__
void setReceivedWithMask_new(itype n, vtype *x, vtype *received, gstype *receive_map, itype shift);

halo_info haloSetup(CSR *A, CSR *R);
void halo_sync(halo_info hi, CSR *A, vector<vtype> *x, bool local_flag = false);
