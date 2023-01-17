#pragma once

#include "basic_kernel/halo_communication/halo_communication.h"
#include "basic_kernel/halo_communication/local_permutation.h"

#include "utility/function_cnt.h"
#include "utility/cudamacro.h"
#include "utility/handles.h"

// NOTE:: Presenti sia buffer locali che buffer globali !!!!

struct FCGPreconditionContext{
  vectorCollection<vtype> *RHS_buffer;

  vectorCollection<vtype> *Xtent_buffer_local;
  vectorCollection<vtype> *Xtent_buffer_2_local;

  
  hierarchy *hrrch;
  int max_level_nums;
  itype *max_coarse_size;
};

namespace FCG{

  extern FCGPreconditionContext context;

  void initPreconditionContext(hierarchy *hrrch);

  void setHrrchBufferSize(hierarchy *hrrch);

  void freePreconditionContext();
}

void preconditionApply(handles *h, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, vector<vtype> *rhs, vector<vtype> *x);

vtype flexibileConjugateGradients_v3(CSR* A, handles *h, vector<vtype> *x, vector<vtype> *rhs, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, int precon, int max_iter, double rtol, int *num_iter, bool precondition_flag);
