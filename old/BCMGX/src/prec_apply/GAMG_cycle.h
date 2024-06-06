#pragma once

#define RTOL 0.25
#include "utility/setting.h"
#include "utility/handles.h"
#include "basic_kernel/matrix/vector.h"

#include "prec_setup/AMG.h"

extern int DETAILED_TIMING;
extern float TOTAL_SOLRELAX_TIME;
extern float TOTAL_RESTGAMG_TIME;

namespace GAMGcycle{
  extern vector<vtype> *Res_buffer;

  void initContext(int n);

  __inline__
  void setBufferSize(itype n);

  void freeContext();
}

void GAMG_cycle(handles *h, int k, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, vectorCollection<vtype> *Rhs, vectorCollection<vtype> *Xtent, vectorCollection<vtype> *Xtent_2, int l, int coarsesolver_type);
//void GAMG_cycle(handles *h, int k, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, vectorCollection<vtype> *Rhs, vectorCollection<vtype> *Xtent, vectorCollection<vtype> *Xtent_2, int l);
