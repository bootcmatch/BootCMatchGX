#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "utility/utils.h"
#include "utility/handles.h"
#include "nsparse.h"
#include "matrix/CSR.h"
#include <cuda_profiler_api.h>
#include "spmspmMGPU/csrseg.h"


itype merge(itype a[], itype b[], itype c[], itype n1, itype n2);

CSR* nsparseMGPU(CSR *Alocal, CSR *Pfull, csrlocinfo *Plocal);

void sym_mtx(CSR *Alocal);

vector<int> *get_missing_col( CSR *Alocal, CSR *Plocal );

void compute_rows_to_rcv_CPU( CSR *Alocal, CSR *Plocal, vector<int> *_bitcol );

CSR* nsparseMGPU_noCommu(handles *h, CSR *Alocal, CSR *Plocal);

CSR* nsparseMGPU_commu(handles *h, CSR *Alocal, CSR *Plocal);

