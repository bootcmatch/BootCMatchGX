#pragma once

#include "matrix/CSR.h"
#include "AMG/AMG.h"
#include "solutionAggregator.h"
#include "AMG/matchingAggregation.h"
#include "utility/utils.h"

#define OVERLAPPED_SMO

struct overlappedSmoother{
  vector<itype> *loc_rows;
  itype loc_n;

  vector<itype> *needy_rows;
  itype needy_n;
};

struct overlappedSmootherList{
  cudaStream_t *local_stream;
  cudaStream_t *comm_stream;

  overlappedSmoother *oss;
  int nm;
};

#define SMOOTHER jacobi_adaptive_miniwarp_overlapped

vector<vtype>* solve(CSR *Alocal, vector<vtype> *rhs, const params p);
