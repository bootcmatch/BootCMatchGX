#pragma once

#include <cub/cub.cuh>
#include "matrix/scalar.h"
#include "matrix/CSR.h"
#include "matrix/vector.h"

__global__
void _getToSend(itype n, vtype *x, vtype *what_to_send, itype *to_send);
__global__
void setReceivedWithMask(itype n, vtype *x, vtype *received, itype *receive_map, itype shift);

halo_info setupAggregation(CSR *A, CSR *R);
void sync_solution(halo_info hi, CSR *A, vector<vtype> *x);
