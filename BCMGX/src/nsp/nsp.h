#pragma once
#include "nsparse.h"  // provides sfCSR (now guarded with #pragma once)
void nsp_spgemm_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c);
