#include "utility/handles.h"

  handles* Handles::init(){

    handles *h = (handles*) malloc(sizeof(handles));
    CHECK_HOST(h);

    CHECK_CUSPARSE( cusparseCreate(&(h->cusparse_h0)) );
    CHECK_CUSPARSE( cusparseCreate(&(h->cusparse_h1)) );

    CHECK_CUBLAS( cublasCreate(&(h->cublas_h)) );

    CHECK_DEVICE( cudaStreamCreate(&(h->stream1)) );
    CHECK_DEVICE( cudaStreamCreate(&(h->stream2)) );
    CHECK_DEVICE( cudaStreamCreate(&(h->stream3)) );
    CHECK_DEVICE( cudaStreamCreate(&(h->stream4)) );

    CHECK_CUSPARSE( cusparseSetStream(h->cusparse_h1, h->stream1) );

    return h;
  }

  void Handles::free(handles *h){
    CHECK_CUSPARSE( cusparseDestroy(h->cusparse_h0) );
    CHECK_CUSPARSE( cusparseDestroy(h->cusparse_h1) );

    CHECK_CUBLAS( cublasDestroy(h->cublas_h) );

    CHECK_DEVICE( cudaStreamDestroy(h->stream1) );
    CHECK_DEVICE( cudaStreamDestroy(h->stream2) );
    CHECK_DEVICE( cudaStreamDestroy(h->stream3) );
    CHECK_DEVICE( cudaStreamDestroy(h->stream4) );
    std::free(h);
  }
