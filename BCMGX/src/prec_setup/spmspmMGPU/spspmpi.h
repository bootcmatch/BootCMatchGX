#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "utility/utils.h"
#include "utility/handles.h"
#include "nsparse.h"
#include "basic_kernel/matrix/CSR.h"
#include <cuda_profiler_api.h>
#include "prec_setup/spmspmMGPU/csrseg.h"


itype merge(itype a[], itype b[], itype c[], itype n1, itype n2);

CSR* nsparseMGPU(CSR *Alocal, CSR *Pfull, csrlocinfo *Plocal, bool used_by_solver = true);

void sym_mtx(CSR *Alocal);

vector<int> *get_missing_col( CSR *Alocal, CSR *Plocal );

void compute_rows_to_rcv_CPU( CSR *Alocal, CSR *Plocal, vector<int> *_bitcol );

CSR* nsparseMGPU_noCommu(handles *h, CSR *Alocal, CSR *Plocal);

CSR* nsparseMGPU_commu(handles *h, CSR *Alocal, CSR *Plocal);

// ----------------------- PICO -------------------------------

CSR* nsparseMGPU_noCommu_new(handles *h, CSR *Alocal, CSR *Plocal, bool used_by_solver = true);

CSR* nsparseMGPU_commu_new(handles *h, CSR *Alocal, CSR *Plocal, bool used_by_solver = true);

vector<int> *get_dev_missing_col( CSR *Alocal, CSR *Plocal );
