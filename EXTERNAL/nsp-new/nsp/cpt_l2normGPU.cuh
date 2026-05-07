#pragma once
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>

//----------------------------------------------------------------------------------------

template <typename T>
struct operat
{
    __device__
        T operator()(const T& x) const {
            return x * x;
        }
};

//----------------------------------------------------------------------------------------

rExt cpt_l2normGPU(const iReg nterms, const rExt * __restrict__ vector){

    operat<rExt> unary_op;
    thrust::plus<rExt> binary_op;
    rExt init = 0;

    // copy memory to a new device_vector (which automatically allocates memory)
    thrust::device_vector<rExt> d_x(vector, vector + nterms); // sizeof vector is (nterms)

    rExt red = std::sqrt( (thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op)) );

    return red;
}

//----------------------------------------------------------------------------------------

__global__ void square(const iReg nterms, rExt * __restrict__ vector){
   int i = threadIdx.x;
   while(i < nterms){
      vector[i] = vector[i]*vector[i];
      i += blockDim.x;
   }
}

//----------------------------------------------------------------------------------------

rExt cpt_l2normGPU_cub(const iReg nterms, rExt * __restrict__ vector){

   // Initilize error flag
   cudaError_t cudaError = cudaSuccess;

   // Alloc scratches
   rExt *d_out;
   cudaError = cudaMalloc((void **)&(d_out), sizeof(rExt));
   CheckCudaError(ERROR_INFO,"allocating d_out failed", cudaError);

   // Determine temporary device storage requirements
   void *d_temp_storage = NULL;
   size_t temp_storage_bytes = 0;

   // Square vector
   LaunchCudaKernel(square<<<1,1024>>>(nterms,vector));
   cudaError = cudaDeviceSynchronize();
   CheckCudaError(ERROR_INFO,"runtime error", cudaError);

   // Alloc CUB temporary storage
   cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vector, d_out, nterms);
   cudaError = cudaMalloc(&d_temp_storage, temp_storage_bytes);
   CheckCudaError(ERROR_INFO,"allocating d_temp_storage failed", cudaError);

   // CUB reduction
   cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vector, d_out, nterms);

   // copy result D2H
   rExt red;
   cudaError = cudaMemcpy(&red, d_out, sizeof(rExt), cudaMemcpyDeviceToHost);
   CheckCudaError(ERROR_INFO,"copying result D2H failed", cudaError);

   return std::sqrt(red);
}

//----------------------------------------------------------------------------------------
