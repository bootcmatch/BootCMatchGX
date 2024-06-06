#pragma once

#include "utility/cudamacro.h"
#include "utility/mpi.h"
#include <cuda_runtime.h>

vector<vtype>* aggregate_vector(vector<vtype>* u_local, itype full_n, vector<vtype>* u);
vector<vtype>* aggregate_vector(vector<vtype>* u_local, itype full_n);

CSR* split_matrix_mpi(CSR* A);

CSR* join_matrix_mpi(CSR* Alocal);

CSR* join_matrix_mpi_all(CSR* Alocal);
