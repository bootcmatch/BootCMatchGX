//#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_profiler_api.h>

#include "utils.h"
#include "handles.h"


#ifdef NSP2_NSPARSE
#include "nsp.h"
#else
#include "nsparse.h"
#endif

#include "CSR.h"

#define USESHRINKEDMATRIX //art

void *Malloc(size_t sz);

itype merge(itype a[], itype b[], itype c[], itype n1, itype n2);

// version0: A is local, P is FULL
CSR* nsparseMGPU_version0(CSR *Alocal, CSR *Pfull);

CSR* nsparseMGPU_version1(handles *h, CSR *Alocal, CSR *Plocal);

CSR* nsparseMGPU_version1(handles *h, CSR *Alocal, CSR *Plocal, double &loc_time);
