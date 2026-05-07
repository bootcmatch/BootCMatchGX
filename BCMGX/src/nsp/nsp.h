#pragma once
// #include "nsparse.h"  // provides sfCSR (now guarded with #pragma once)

#ifdef FLOAT
typedef float real;

#elif defined DOUBLE
typedef double real;

#else
typedef double real;
#endif

typedef struct
{
    int *rpt;
    int *col;
    real *val;
    int *d_rpt;
    int *d_col;
    real *d_val;
    int M;
    int N;
    int nnz;
    int nnz_max;
    char *matrix_name;
} sfCSR;

#define div_round_up(a, b) ((a % b == 0)? a / b : a / b + 1)

void nsp_spgemm_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c);
