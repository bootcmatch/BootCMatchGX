#pragma once

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <float.h>
#include <mpi.h>

#define DEFAULT_STREAM 0
#define WARP_SIZE 32
#define FULL_WARP 32
#define FULL_MASK 0xFFFFFFFF

#define MINI_WARP_THRESHOLD_2 3
#define MINI_WARP_THRESHOLD_4 6
#define MINI_WARP_THRESHOLD_8 12
#define MINI_WARP_THRESHOLD_16 24

void *Malloc(size_t sz);

void *Realloc(void *ptr, size_t sz);

// cuSPARSE API errors
const char* cusparseGetStatusString(cusparseStatus_t error);
const char* cublasGetStatusString(cublasStatus_t status);
void CHECK_CUBLAS(cublasStatus_t err);

namespace Eval{
  void printMetaData(const char* name, double value, int type);
}


#define CHECK_CUSPARSE(X) \
    { \
    cusparseStatus_t status = X;\
    if ( status != CUSPARSE_STATUS_SUCCESS ) { \
      const char *err_str = cusparseGetStatusString(status);\
      fprintf( stderr, "[ERROR CUSPARSE] :\n\t%s; LINE: %d; FILE: %s\n", err_str, __LINE__, __FILE__);\
	exit(1); \
    } \
    }

#define CHECK_DEVICE(X) \
    { \
    cudaError_t status = X;\
    if ( status != cudaSuccess ) { \
      const char *err_str = cudaGetErrorString(status);\
      fprintf( stderr, "[ERROR DEVICE] :\n\t%s; LINE: %d; FILE: %s\n", err_str, __LINE__, __FILE__);\
	exit(1); \
    } \
    }

#define CHECK_HOST(X) \
    { \
    if ( X == NULL ) { \
      fprintf( stderr, "[ERROR HOST] :\n\t LINE: %d; FILE: %s\n", __LINE__, __FILE__);\
	exit(1); \
    } \
    }



namespace TIME{

  void init();

  void addTimer();

  void start();

  float stop();

  void free();
}

__device__
__inline__
unsigned int getMaskByWarpID(unsigned int m_size, unsigned int m_id){
  if(m_size == 32)
    return FULL_WARP;
  unsigned int m = ( 1 << (m_size) ) - 1;
  return ( m << (m_size*m_id) );
}

struct gridblock{
  dim3 g;
  dim3 b;
};

gridblock gb1d(const unsigned n, const unsigned block_size, const bool is_warp_agg=false, int MINI_WARP_SIZE=32);
