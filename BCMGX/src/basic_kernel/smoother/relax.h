#ifndef OVERLAPPED_SMO
#define OVERLAPPED_SMO

#include "basic_kernel/matrix/CSR.h"
#include "utility/function_cnt.h"
#include "utility/handles.h"

#include "basic_kernel/matrix/scalar.h"

extern halo_info *H_halo_info;

#include "prec_setup/AMG.h"  //BUG
#include "basic_kernel/halo_communication/halo_communication.h"

struct relaxContext{
  vector<vtype> *temp_buffer;
  itype n;
};

namespace Relax{
  extern relaxContext context;

  void initContext(itype n);

  void set_n_context(itype n);

  void freeContext();
}

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

extern overlappedSmootherList *osl;

overlappedSmootherList* init_overlappedSmootherList(int nm);

void setupOverlappedSmoother(CSR *A, overlappedSmoother *os);

void setupOverlappedSmoother_cpu(CSR *A, overlappedSmoother *os);

overlappedSmootherList* init_overlappedSmootherList(int nm);

void relaxCoarsest(handles *h, int k, CSR* A, vector<vtype>* D, vector<vtype>* M, vector<vtype> *f, int relax_type, vtype relax_weight, vector<vtype> *u, vector<vtype> **u_, itype nlocal, bool forward=true);

void relax(handles *h, int k, int level, CSR* A, vector<vtype>* D, vector<vtype>* M, vector<vtype> *f, int relax_type, vtype relax_weight, vector<vtype> *u, vector<vtype> **u_, bool forward=true);
#endif
