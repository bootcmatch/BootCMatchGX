/**
 * @file
 */
#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "utility/cudamacro.h"
#include "utility/mpi.h"
#include <cuda_runtime.h>

vector<vtype>* aggregate_vector_all(vector<vtype>* u_local, itype full_n = 0);

CSR* split_matrix_mpi_host(CSR* A);
CSR* split_matrix_mpi(CSR* A);

CSR* join_matrix_mpi(CSR* Alocal);
