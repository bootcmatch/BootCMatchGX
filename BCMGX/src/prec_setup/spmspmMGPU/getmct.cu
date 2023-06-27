#include <stdlib.h>
using namespace std;

#include "cuCompactor.cuh"
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include "utility/setting.h"
#include "utility/utils.h"

#include "utility/cudamacro.h"
#include "basic_kernel/custom_cudamalloc/custom_cudamalloc.h"

#include "utility/function_cnt.h"

#define NUM_THR 1024

extern functionCall_cnt function_cnt;

struct int_predicate
{
	__host__ __device__
	bool operator()(const int x, const int f, const int l)
	{
		return ((x<f)||(x>l));
	}
};

struct int_equal_to
{
    __host__ __device__
    bool operator()(int a, int b)
    {
        return a == b;
    }
};

int *getmct(int *Col, int nnz, int f, int l, int *uvs, int **bitcol, int *bitcolsize, int num_thr) {

    int *d_output, d_size;
    if ((*bitcol)!=NULL) {
        d_output = *bitcol;
        d_size = bitcolsize[0];
    }else{
        if (nnz>sizeof_buffer_4_getmct) {
            if (sizeof_buffer_4_getmct>0) {
                MY_CUDA_CHECK( cudaFree(buffer_4_getmct) );
            }
            sizeof_buffer_4_getmct = nnz;
            cudaMalloc_CNT
            CHECK_DEVICE( cudaMalloc(&buffer_4_getmct,sizeof(int)*sizeof_buffer_4_getmct) );
        }
        d_output = buffer_4_getmct;
        d_size=cuCompactor::compact<int>(Col,d_output,nnz,int_predicate(),num_thr,0,l-f);
        cudaDeviceSynchronize();
    }
    thrust::device_ptr<int> dev_ptr( d_output );
    if ((*bitcol) == NULL)
        thrust::sort( dev_ptr, dev_ptr + d_size );
    thrust::device_vector<int> uv(dev_ptr, dev_ptr+d_size);
    if ((*bitcol) == NULL)
        uv.erase(thrust::unique(uv.begin(), uv.end(), int_equal_to()),uv.end());
    // -------------------------------------------------------
    
    
    int * dv_ptr = thrust::raw_pointer_cast(uv.data());
	uvs[0]=uv.size(); 
    int *h_ptr=(int *)malloc(uvs[0]*sizeof(int));
	if(h_ptr==NULL) {
		fprintf(stderr,"Could not get memory in getmct\n");
		exit(1);
	}
    CHECK_DEVICE( cudaMemcpy(h_ptr,dv_ptr,uvs[0]*sizeof(int),cudaMemcpyDeviceToHost) );

    if ((*bitcol) == NULL) {
        // ------------- custom cudaMalloc -------------
//         cudaMalloc_CNT
//         cudaMalloc(bitcol, sizeof(int)*(uvs[0]));
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        *bitcol = CustomCudaMalloc::alloc_itype(uv.size(), 1);
        // ---------------------------------------------
        CHECK_DEVICE( cudaMemcpy(*bitcol,dv_ptr,uv.size()*sizeof(int),cudaMemcpyDeviceToDevice) );
        *bitcolsize = *uvs;
//         MY_CUDA_CHECK( cudaFree(d_output) );
    }
    return h_ptr;
}


__global__
void primo_kernel (int* local, int* uv, int*  all, int uvs, int locals, int f, unsigned int* idx) {
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int lowerhalo=0;
    if(tid<uvs) {
        lowerhalo = uv[tid] < f;
        if(lowerhalo) {
            all[tid] = uv[tid];
        } else {
            all[tid + locals] = uv[tid];
        }
    }
    
    lowerhalo = lowerhalo ? tid +1 : 0;
    if(lowerhalo>0) {
        unsigned int compared, result;
        while(true) {
            compared = idx[0];
            if (lowerhalo > compared) {
                result = atomicCAS(idx, compared, lowerhalo);
                if (result == lowerhalo)
                    break;
            }else{
                break;
            }
        }
    }
}

__global__
void secondo_kernel (int* local, int locals, int* all, unsigned int *idx) {
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < locals) {
        all[idx[0]+tid] = local[tid];
    }
}

extern unsigned int * idx_4shrink;
extern bool alloced_idx;

int *getmct_4shrink(int *Col, int nnz, int f, int l, int first_or_last, int *uvs, int **bitcol, int *bitcolsize, int* post_local, int num_thr) {

    int *d_output, d_size;
    if ((*bitcol)!=NULL) {
        d_output = *bitcol;
        d_size = bitcolsize[0];
    }else{
        if (nnz>sizeof_buffer_4_getmct) {
            if (sizeof_buffer_4_getmct>0) {
                MY_CUDA_CHECK( cudaFree(buffer_4_getmct) );
            }
            sizeof_buffer_4_getmct = nnz;
            cudaMalloc_CNT
            CHECK_DEVICE( cudaMalloc(&buffer_4_getmct,sizeof(int)*sizeof_buffer_4_getmct) );
        }
        d_output = buffer_4_getmct;
        d_size=cuCompactor::compact<int>(Col,d_output,nnz,int_predicate(),num_thr,0,l);
        cudaDeviceSynchronize();
    }
    thrust::device_ptr<int> dev_ptr( d_output );
    if ((*bitcol) == NULL)
        thrust::sort( dev_ptr, dev_ptr + d_size );
    thrust::device_vector<int> uv(dev_ptr, dev_ptr+d_size);
    if ((*bitcol) == NULL)
        uv.erase(thrust::unique(uv.begin(), uv.end(), int_equal_to()),uv.end());
	
	thrust::device_vector<int> result( (l - f +1) + uv.size(), -1);
    
    unsigned int first_post_local;
    if (first_or_last) {
        if (first_or_last <0) {
            thrust::sequence(result.begin(), result.begin() +l+1, 0);
            thrust::copy(uv.begin(), uv.end(), result.begin() + l+1);
            first_post_local = 0;
        }else{
            thrust::copy(uv.begin(), uv.end(), result.begin());
            thrust::sequence(result.begin() + uv.size(), result.end(), 0);
            first_post_local = (unsigned int) uv.size();
        }
    }else{
        thrust::device_vector<int> locals(l - f +1, 1);
        thrust::sequence(locals.begin(), locals.end(), 0);
            
        if (alloced_idx == false) {
            cudaMalloc_CNT
            CHECK_DEVICE( cudaMalloc( (void**)&idx_4shrink, sizeof( unsigned int ) ) );
            alloced_idx = true;
        }
        unsigned int *dev_idx = idx_4shrink;
        
        CHECK_DEVICE( cudaMemset(dev_idx, 0, sizeof(unsigned int)) );
            
        int * dv_uv = thrust::raw_pointer_cast(uv.data());
        int * dv_local = thrust::raw_pointer_cast(locals.data());
        int * dv_result = thrust::raw_pointer_cast(result.data());
            
        gridblock gb = gb1d(uv.size(), NUM_THR);
        primo_kernel<<<gb.g, gb.b>>>(dv_local, dv_uv, dv_result, uv.size(), locals.size(), 0, dev_idx);
        CHECK_DEVICE( cudaMemcpy(&first_post_local, dev_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
            
        gb = gb1d(locals.size(), NUM_THR);
        secondo_kernel<<<gb.g, gb.b>>>(dv_local, locals.size(), dv_result, dev_idx);
    }
    
    
    uvs[0] = result.size();
    int * dv_ptr = thrust::raw_pointer_cast(result.data());
    
    assert ( uvs[0] == ((l - f +1) + uv.size()) );
    assert( uvs[0] >= (l-f) );

    if ((*bitcol) == NULL) {
        int * dv_ptr2 = thrust::raw_pointer_cast(uv.data());
        // ------------- custom cudaMalloc -------------
//         cudaMalloc_CNT
//         cudaMalloc(bitcol, sizeof(int)*(uvs[0]));
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        *bitcol = CustomCudaMalloc::alloc_itype(uv.size(), 1);
        // ---------------------------------------------
        CHECK_DEVICE( cudaMemcpy(*bitcol,dv_ptr2,uv.size()*sizeof(int),cudaMemcpyDeviceToDevice) );
        *bitcolsize = d_size;
//         MY_CUDA_CHECK( cudaFree(d_output) );
    }
    *post_local = (int) first_post_local;
    
    cudaMalloc_CNT
    CHECK_DEVICE( cudaMalloc(&d_output, uvs[0] * sizeof(int)) );
    CHECK_DEVICE( cudaMemcpy(d_output, dv_ptr, uvs[0] * sizeof(int),cudaMemcpyDeviceToDevice) );
	
  	return d_output;
}
