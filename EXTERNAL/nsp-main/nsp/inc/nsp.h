#pragma once

#include <cuda.h>        // to use cuda
// #include <cusparse_v2.h> // to use cusparse

#include <helper_cuda.h>
// #include "csrseg.h"

#include <cub/cub.cuh>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>  

#ifdef FLOAT
typedef float real;

#elif defined DOUBLE
typedef double real;

#else
typedef double real;
#endif


#define SHTB 49152
#define WARPSIZE 32 
#define MASKFULL 0xffffffff
#define MAXINT 0x7fffffff

#define HASH_UPD

#define CC 70

#undef BLKSIZE_MxM
#define BLKSIZE_MxM 1024


#define MAX_SH_DENSE 5120     // maximum array size that fit the dense bin
#define MIN_DENSITY 10.       // minimum density of a row allowed to use the dense bin
#define MIN_DENSITY_chunk 30. // minimum density of a row allowed to use the dense_chunk kernel

// disjoint number to hash table
#define HASH_SCAL 107
// number of groups
#define BIN_NUM 11
// maximum row size to use the PWARP/ROW-based kernel during the symbolic computation
#define IMB_PWMIN WARPSIZE
// maxiumum row size to use the PWARP/ROW-based kernel during the computation
#define B_PWMIN WARPSIZE/2
// number of threads for each row to use in PWARP/ROW-based kernel
#define PWARP WARPSIZE/8
// macro to compute GPU resources
#define div_round_up(a, b) ((a % b == 0)? a / b : a / b + 1)


/* Structure of Formats*/
typedef struct
{
    int *rpt;
    int *col;
    real *val;
    int *d_rpt;
    int *d_col;
    real *d_val;
    unsigned int M;
    unsigned int N;
    int nnz;
    int nnz_max;
    char *matrix_name;
} sfCSR;


// data structure for nsparse
typedef struct {
   // stream for each group
   cudaStream_t* stream = NULL;
   // number of nnz of each row of C
   int *d_row_nz;
   // maximum nnz value
   int  h_max = 0;
   int *d_max = NULL;
   // size (number of rows) of each group
   int *h_bin_size;
   int *d_bin_size;
   // offset of each group
   int *h_bin_offset;
   int *d_bin_offset;
   // permutation array (based on d_row_z)
   int *d_row_perm;
   // maximum SHared memory per Thread-Block for the symbolic computation (setting)
   int SHTB_set_max;
   // maximum SHared memory per Thread-Block for the computation 
   int SHTB_cmp_max;
   // maxiumum row size of group 1 to use the TB/ROW-based kernel during the symbolic computation
   int IMB_MIN;
   // maxiumum row size of group 1 to use the TB/ROW-based kernel during the computation
   int B_MIN;
} sfBIN;


//////////////////////////////////////////////////////////////////////////////////////////

__global__ void nsp_calc_val_sort_rows(int nSigBits, int *row_perm,int bin_offset, int *iat, int *ja, double *coef);

void nsp_spgemm_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c);

