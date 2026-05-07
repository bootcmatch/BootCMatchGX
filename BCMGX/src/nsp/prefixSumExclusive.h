#pragma once

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

// Out-of-place exclusive prefix sum: out[i] = sum(in[0..i-1])
// Scans nelems elements; out must have room for nelems elements.
template<typename T>
inline void prefixSumExclusive(const T *in, int nelems, T *out, int initVal) {
    thrust::device_ptr<const T> d_in  = thrust::device_pointer_cast(in);
    thrust::device_ptr<T>       d_out = thrust::device_pointer_cast(out);
    thrust::exclusive_scan(d_in, d_in + nelems, d_out, (T)initVal);
}

// In-place exclusive prefix sum on a CUDA stream: inout[i] = sum(inout[0..i-1])
// Scans nelems elements in place.
template<typename T>
inline void prefixSumExclusive(T *inout, int nelems, int initVal, cudaStream_t stream) {
    thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(inout);
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           d_ptr, d_ptr + nelems, d_ptr, (T)initVal);
}
