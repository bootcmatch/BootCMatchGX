#pragma once

#include "utility/utils.h"

struct handles{
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStream_t stream3;
  cudaStream_t stream4;

  cusparseHandle_t cusparse_h0, cusparse_h1;
  cublasHandle_t cublas_h;
};

namespace Handles{
  handles* init();

  void free(handles *h);
}
