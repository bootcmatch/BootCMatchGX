#include "utility/utils.h"

void *Realloc(void *pptr, size_t sz) {
	void *ptr;
	if (!sz) {
	    printf("Allocating zero bytes...\n");
	    exit(EXIT_FAILURE);
	}
	ptr = (void *)realloc(pptr, sz);
	if (!ptr) {
		fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	return ptr;
}

void *Malloc(size_t sz) {
	void *ptr;
	if (!sz) {
	    printf("Allocating zero bytes...\n");
	    exit(EXIT_FAILURE);
	}
	ptr = (void *)malloc(sz);
	if (!ptr) {
	    fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
	    exit(EXIT_FAILURE);
	}
	memset(ptr, 0, sz);
	return ptr;
}

namespace Eval{
  void printMetaData(const char* name, double value, int type){
    printf("#META %s ", name);
    if(type == 0){
      int value_int = (int) value;
      printf("int %d", value_int);
    }
    else if(type == 1)
      printf("float %le", value);
    printf("\n");
  }
}

gridblock gb1d(const unsigned n, const unsigned block_size, const bool is_warp_agg, int MINI_WARP_SIZE){
  gridblock gb;

  int n_ = n;
  if(n==0) {
          gb.b = 0;
          gb.g = 0;
          return gb;
  }
  if(is_warp_agg)
    n_ *= MINI_WARP_SIZE;

  dim3 block (block_size);
  dim3 grid (  (n_+(block.x-1)) / block.x);

  gb.b = block;
  gb.g = grid;
  return gb;
}


const char* cusparseGetStatusString(cusparseStatus_t error){
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:                  return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:          return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:             return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:            return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:            return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR:            return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED:         return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:           return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    return "<unknown>";
}



const char* cublasGetStatusString(cublasStatus_t status) {
  switch(status) {
    case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:     return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:     return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN_ERROR";
}

void CHECK_CUBLAS(cublasStatus_t err){
  const char *err_str = cublasGetStatusString(err);
  if(err != CUBLAS_STATUS_SUCCESS){
    printf("[ERROR CUBLAS] :\n\t%s\n", err_str);
    exit(1);
  }
}


#include <ctime>
#ifndef TIMING_H
#define TIMING_H
#include <sys/time.h>

#define GETTOD   gettimeofday(&temp_1, (struct timezone*)0)

#define TIME_ELAPSED ((temp_1.tv_sec)+((temp_1 .tv_usec)/(1.e6)))
static struct timeval temp_1;
#endif

namespace TIME{

  int timer_index;
  int n;
  double *starts, *stops;

  void init(){
    TIME::timer_index = 0;
    TIME::n = 0;
    TIME::starts = NULL;
    TIME::stops = NULL;
  }

  void addTimer(){
    TIME::starts = (double*) realloc(TIME::starts, sizeof(double) * TIME::n);
    CHECK_HOST(TIME::starts);
    TIME::stops = (double*) realloc(TIME::stops, sizeof(double) * TIME::n);
    CHECK_HOST(TIME::stops);
    TIME::starts[TIME::n-1] = 0.;
    TIME::stops[TIME::n-1] = 0.;
  }

  void start(){
    if(TIME::timer_index == TIME::n){
      TIME::n++;
      TIME::addTimer();
    }
    GETTOD;
    TIME::starts[TIME::timer_index] = TIME_ELAPSED;
    TIME::timer_index++;
  }

  float stop(){
    double milliseconds = 0.;
    double start_ = TIME::starts[TIME::timer_index-1];
    GETTOD;
    double stop_ = TIME_ELAPSED;
    milliseconds = stop_ - start_;
    milliseconds *=1000.0;
    TIME::timer_index--;
    return (float)milliseconds;
  }

  void free(){
    std::free( TIME::starts);
    std::free( TIME::stops);
  }
}
