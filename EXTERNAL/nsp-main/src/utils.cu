#include "utils.h"


gridblock gb1d(const unsigned n, const unsigned block_size, const bool is_warp_agg, int MINI_WARP_SIZE){
  gridblock gb;

  int n_ = n;

  if(is_warp_agg)
    n_ *= MINI_WARP_SIZE;

  dim3 block (block_size);
  dim3 grid ( ceil( (double) n_ / (double) block.x));

  gb.b = block;
  gb.g = grid;

  //printf("%d %d\n\n", gb.g.x, gb.b.x);

  return gb;
}


// cuSPARSE API errors
const char* cusparseGetStatusString(cusparseStatus_t error){
    switch (error)
    {
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:   return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
        case CUSPARSE_STATUS_NOT_SUPPORTED:            return "CUSPARSE_STATUS_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT:               return "CUSPARSE_STATUS_ZERO_PIVOT";
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

//##############################################################################

namespace TIME{

  int timer_index;
  int n;
  cudaEvent_t *starts, *stops;

  void init(){
    TIME::timer_index = 0;
    TIME::n = 0;
    TIME::starts = NULL;
    TIME::stops = NULL;
  }

  void addTimer(){
    TIME::starts = (cudaEvent_t*) realloc(TIME::starts, sizeof(cudaEvent_t) * TIME::n);
    CHECK_HOST(TIME::starts);
    TIME::stops = (cudaEvent_t*) realloc(TIME::stops, sizeof(cudaEvent_t) * TIME::n);
    CHECK_HOST(TIME::stops);
    cudaEventCreate(&TIME::starts[TIME::n-1]);
    cudaEventCreate(&TIME::stops[TIME::n-1]);
  }

  void start(){
    if(TIME::timer_index == TIME::n){
      TIME::n++;
      TIME::addTimer();
    }
    cudaEventRecord(TIME::starts[TIME::timer_index]);
    TIME::timer_index++;
  }

  float stop(){
    CHECK_DEVICE( cudaDeviceSynchronize() );
    float milliseconds = 0.;
    cudaEvent_t start_ = TIME::starts[TIME::timer_index-1];
    cudaEvent_t stop_ = TIME::stops[TIME::timer_index-1];

    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&milliseconds, start_, stop_);
    TIME::timer_index--;
    return milliseconds;
  }

  void free(){
    for(int i=0; i<TIME::n; i++){
      cudaEventDestroy( TIME::starts[i]);
      cudaEventDestroy( TIME::stops[i]);
    }
    std::free( TIME::starts);
    std::free( TIME::stops);
  }
}

/*
#include <ctime>

namespace TIME{

  int timer_index;
  int n;
  float *starts, *stops;

  void init(){
    TIME::timer_index = 0;
    TIME::n = 0;
    TIME::starts = NULL;
    TIME::stops = NULL;
  }

  void addTimer(){
    TIME::starts = (float*) realloc(TIME::starts, sizeof(float) * TIME::n);
    CHECK_HOST(TIME::starts);
    TIME::stops = (float*) realloc(TIME::stops, sizeof(float) * TIME::n);
    CHECK_HOST(TIME::stops);
    TIME::starts[TIME::n-1] = 0.;
    TIME::stops[TIME::n-1] = 0.;
  }

  void start(){
    if(TIME::timer_index == TIME::n){
      TIME::n++;
      TIME::addTimer();
    }
    TIME::starts[TIME::timer_index] = (float) clock() /  (float) CLOCKS_PER_SEC;
    TIME::timer_index++;
  }

  float stop(){
    CHECK_DEVICE( cudaDeviceSynchronize() );
    float milliseconds = 0.;
    float start_ = TIME::starts[TIME::timer_index-1];
    float stop_ = (float) clock() /  (float) CLOCKS_PER_SEC;

    milliseconds = stop_ - start_;
    TIME::timer_index--;
    return milliseconds;
  }

  void free(){
    std::free( TIME::starts);
    std::free( TIME::stops);
  }
}
*/
