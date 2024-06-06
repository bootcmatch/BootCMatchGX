#pragma once


#include "utility/myMPI.h"
#include <cuda_runtime.h>
#include "utility/cudamacro.h"

vector<vtype>* aggregateVector(vector<vtype> *u_local, itype full_n, vector<vtype> *u);


CSR* split_MatrixMPI(CSR *A);

CSR* join_MatrixMPI(CSR *Alocal);

CSR* join_MatrixMPI_all(CSR *Alocal);
