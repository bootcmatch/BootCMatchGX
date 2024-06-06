#include "utility/handles.h"

handles* Handles::init()
{

    handles* h = (handles*)malloc(sizeof(handles));
    CHECK_HOST(h);

    CHECK_CUBLAS(cublasCreate(&(h->cublas_h)));

    CHECK_DEVICE(cudaStreamCreate(&(h->stream1)));
    CHECK_DEVICE(cudaStreamCreate(&(h->stream2)));
    CHECK_DEVICE(cudaStreamCreate(&(h->stream3)));
    CHECK_DEVICE(cudaStreamCreate(&(h->stream4)));
    CHECK_DEVICE(cudaStreamCreate(&(h->stream_free)));

    return h;
}

void Handles::free(handles* h)
{
    CHECK_CUBLAS(cublasDestroy(h->cublas_h));

    CHECK_DEVICE(cudaStreamDestroy(h->stream1));
    CHECK_DEVICE(cudaStreamDestroy(h->stream2));
    CHECK_DEVICE(cudaStreamDestroy(h->stream3));
    CHECK_DEVICE(cudaStreamDestroy(h->stream4));
    CHECK_DEVICE(cudaStreamDestroy(h->stream_free));

    std::free(h);
}
