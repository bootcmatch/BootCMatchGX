
/**
 * @file cuCompactor.h
 *
 * @brief Provides compacting and filtering functionality using CUDA. 
 *        This header contains GPU kernels and utility functions for efficiently 
 *        compacting a list based on a predicate and performing other CUDA-related tasks.
 *
 * @date Created on: 21/mag/2015
 * @author knotman
 */

#ifndef CUCOMPACTOR_H_
#define CUCOMPACTOR_H_

#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "cuda_error_check.h"

#include "utility/setting.h"
#include "utility/cudamacro.h"
#include "utility/memory.h"

namespace cuCompactor {

/**
 * @def WARPSIZE
 * 
 * @brief Defines the warp size used in compacting operations.
 * 
 * A warp in CUDA consists of 32 threads, which is the size of a single SIMD (Single Instruction, Multiple Data) execution unit.
 */
#define WARPSIZE (32)

/**
 * @brief Computes the ceiling of the division of x by y.
 *
 * @param x The numerator of the division.
 * @param y The denominator of the division.
 * 
 * @return The smallest integer greater than or equal to the result of x / y.
 */
__host__ __device__ int divupXT(int x, int y) { return x / y + (x % y ? 1 : 0); }

/**
 * @brief Computes the power of 2 raised to the exponent e.
 *
 * @param e The exponent to which 2 is raised.
 * 
 * @return The value of 2 raised to the power e.
 */
__device__ __inline__ int pow2i (int e){
	return 1<<e;
}

/**
 * @brief Kernel function to compute the count of valid elements in each thread block based on a predicate.
 *
 * This kernel will check each element in the input array and compute how many elements satisfy the predicate.
 *
 * @tparam T The type of elements in the input array.
 * @tparam Predicate The type of the predicate function.
 * 
 * @param d_input The device pointer to the input array.
 * @param length The length of the input array.
 * @param d_BlockCounts The device pointer where the number of valid elements in each thread block is stored.
 * @param predicate The predicate function to check elements against.
 * @param f The lower bound used in the predicate.
 * @param l The upper bound used in the predicate.
 */
template <typename T,typename Predicate>
__global__ void computeBlockCounts(T* d_input,int length,int*d_BlockCounts,Predicate predicate,int f, int l){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < length){
		int pred = predicate(d_input[idx],f,l);
		int BC=__syncthreads_count(pred);

		if(threadIdx.x==0){
			d_BlockCounts[blockIdx.x]=BC; // BC will contain the number of valid elements in all threads of this thread block
		}
	}
}

/**
 * @brief Kernel function to compact an array based on a predicate.
 *
 * This kernel performs the actual compaction by copying valid elements to the output array. 
 * It uses warp-level and block-level parallelism to achieve high performance.
 *
 * @tparam T The type of elements in the input and output arrays.
 * @tparam Predicate The type of the predicate function.
 * 
 * @param d_input The device pointer to the input array.
 * @param length The length of the input array.
 * @param d_output The device pointer where the compacted array will be stored.
 * @param d_BlocksOffset The device pointer to store the offsets for each thread block.
 * @param predicate The predicate function to check elements against.
 * @param f The lower bound used in the predicate.
 * @param l The upper bound used in the predicate.
 */
template <typename T,typename Predicate>
__global__ void compactK(T* d_input,int length, T* d_output,int* d_BlocksOffset,Predicate predicate, int f, int l){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	extern __shared__ int warpTotals[];
	if(idx < length){
		int pred = predicate(d_input[idx],f,l);
		int w_i = threadIdx.x/WARPSIZE; //warp index
		int w_l = idx % WARPSIZE;//thread index within a warp

		// compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
		int t_m = FULL_MASK >> (WARPSIZE-w_l); //thread mask
		#if (CUDART_VERSION < 9000)
		int b   = __ballot(pred) & t_m; //ballot result = number whose ith bit is one if the ith's thread pred is true masked up to the current index in warp
		#else
		int b	= __ballot_sync(FULL_MASK,pred) & t_m;
		#endif
		int t_u	= __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

		// last thread in warp computes total valid counts for the warp
		if(w_l==WARPSIZE-1){
			warpTotals[w_i]=t_u+pred;
		}

		// need all warps in thread block to fill in warpTotals before proceeding
		__syncthreads();

		// first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
		int numWarps = blockDim.x/WARPSIZE;
		unsigned int numWarpsMask = FULL_MASK >> (WARPSIZE-numWarps);
		if(w_i==0 && w_l<numWarps){
			int w_i_u=0;
			for(int j=0;j<=5;j++){ // must include j=5 in loop in case any elements of warpTotals are identically equal to 32
				#if (CUDART_VERSION < 9000)
		                int b_j =__ballot( warpTotals[w_l] & pow2i(j) ); //# of the ones in the j'th digit of the warp offsets
				#else
				int b_j =__ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j) );
				#endif
				w_i_u += (__popc(b_j & t_m)  ) << j;
				//printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
			}
			warpTotals[w_l]=w_i_u;
		}

		// need all warps in thread block to wait until prefix sum is calculated in warpTotals
		__syncthreads(); 

		// if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
		if(pred){
			d_output[t_u+warpTotals[w_i]+d_BlocksOffset[blockIdx.x]]= d_input[idx];
		}


	}
}

/**
 * @brief Prints the elements of an array from the device to the console.
 *
 * This kernel is for debugging purposes. It prints elements of the array with line breaks after 
 * a specified number of elements.
 *
 * @tparam T The type of the array elements.
 * 
 * @param hd_data The device pointer to the array to be printed.
 * @param size The size of the array.
 * @param newline The number of elements to print per line.
 */
template <class T>
__global__  void printArray_GPU(T* hd_data, int size,int newline){
	int w=0;
	for(int i=0;i<size;i++){
		if(i%newline==0) {
			printf("\n%i -> ",w);
			w++;
		}
		printf("%i ",hd_data[i]);
	}
	printf("\n");
}

/**
 * @brief Performs compaction of the input array based on a predicate.
 *
 * This function uses multiple phases to compute the compacted array:
 * 1. Compute the number of valid elements in each thread block.
 * 2. Compute the exclusive prefix sum of the valid block counts.
 * 3. Compact the valid elements by using the computed offsets.
 *
 * @tparam T The type of the array elements.
 * @tparam Predicate The type of the predicate function.
 * 
 * @param d_input The device pointer to the input array.
 * @param d_output The device pointer where the compacted array will be stored.
 * @param length The length of the input array.
 * @param predicate The predicate function to check elements against.
 * @param blockSize The block size to use in kernel launches.
 * @param f The lower bound used in the predicate.
 * @param l The upper bound used in the predicate.
 * 
 * @return The number of elements in the compacted array.
 */
template <typename T,typename Predicate>
int compact(T* d_input,T* d_output,int length, Predicate predicate, int blockSize, int f, int l){
	static int stat_numBlocks = 0;
    int numBlocks = divupXT(length,blockSize);
    if (numBlocks > stat_numBlocks) {
        //if (stat_numBlocks > 0) {
		CUDA_FREE(glob_d_BlocksCount);
		CUDA_FREE(glob_d_BlocksOffset);
        //}
        stat_numBlocks = numBlocks;
		glob_d_BlocksCount = CUDA_MALLOC(int, stat_numBlocks);
        glob_d_BlocksOffset = CUDA_MALLOC(int, stat_numBlocks);
    }
	thrust::device_ptr<int> thrustPrt_bCount(glob_d_BlocksCount);
	thrust::device_ptr<int> thrustPrt_bOffset(glob_d_BlocksOffset);

	//phase 1: count number of valid elements in each thread block
	computeBlockCounts<<<numBlocks,blockSize>>>(d_input,length,glob_d_BlocksCount,predicate, f, l);
	
	//phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
	thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);
	
	//phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
	compactK<<<numBlocks,blockSize,sizeof(int)*(blockSize/WARPSIZE)>>>(d_input,length,d_output,glob_d_BlocksOffset,predicate, f, l);

	// determine number of elements in the compacted list
	int compact_length = thrustPrt_bOffset[numBlocks-1] + thrustPrt_bCount[numBlocks-1];

// 	CUDA_FREE(d_BlocksCount);
// 	CUDA_FREE(d_BlocksOffset);

	return compact_length;
}

} /* namespace cuCompactor */
#endif /* CUCOMPACTOR_H_ */
