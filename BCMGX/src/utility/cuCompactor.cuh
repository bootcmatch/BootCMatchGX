
/**
 * @file cuCompactor.h
 * @brief This header defines a CUDA-based compaction operation (cuCompactor).
 * 
 * It processes an input array, compacting it by removing unwanted elements based on a predicate function.
 * The algorithm is optimized using warp-level synchronization and memory management techniques to maximize performance.
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

#define WARPSIZE (32)

/**
 * @brief Divides two integers and rounds up to the nearest integer.
 * 
 * Computes the ceiling of the division result: `x / y`, i.e., it calculates the smallest integer greater than or
 * equal to the division of x by y.
 * 
 * @param x The numerator.
 * @param y The denominator.
 * @return The ceiling of the division result.
 */
__host__ __device__ __inline__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

/**
 * @brief Computes 2 raised to the power of `e`.
 * 
 * This is done by bit-shifting the value `1` to the left by `e` positions (equivalent to 2^e).
 * 
 * @param e The exponent.
 * @return 2 raised to the power of e.
 */
__device__ __inline__ int pow2i (int e) {
	return 1<<e;
}

namespace cuCompactor {

/**
 * @brief Kernel to count the number of valid elements in each thread block based on a predicate.
 * 
 * This kernel computes the number of valid elements in each block by applying a predicate function. Each thread in a
 * block checks if its corresponding element satisfies the condition defined in the predicate and counts the valid ones.
 * The total number of valid elements in each block is stored in `d_BlockCounts`.
 * 
 * @tparam T The type of elements in the input array.
 * @tparam Predicate The type of the predicate function.
 * @param d_input Input device array.
 * @param length The length of the input array.
 * @param d_BlockCounts Output device array to store the count of valid elements in each block.
 * @param predicate The predicate function used to check each element.
 * @param f Lower bound parameter for the predicate.
 * @param l Upper bound parameter for the predicate.
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
 * @brief Kernel to compact the input array based on the predicate function.
 * 
 * This kernel compacts the input array by moving the valid elements (those that satisfy the predicate) to the
 * correct positions in the output array. It uses warp-level synchronization and exclusive prefix sum techniques to
 * efficiently determine the output offset for each element.
 * 
 * @tparam T The type of elements in the input array.
 * @tparam Predicate The type of the predicate function.
 * @param d_input Input device array.
 * @param length The length of the input array.
 * @param d_output Output device array to store the compacted elements.
 * @param d_BlocksOffset Device array to store the offsets for each block.
 * @param predicate The predicate function used to check each element.
 * @param f Lower bound parameter for the predicate.
 * @param l Upper bound parameter for the predicate.
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
 * @brief Kernel to print the contents of an array in the GPU for debugging.
 * 
 * This kernel prints the contents of the input array to the console in a formatted manner. It is useful for debugging
 * and visualizing the array content.
 * 
 * @tparam T The type of elements in the array.
 * @param hd_data The input device array.
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
 * @brief Performs the compaction of an input array by removing unwanted elements based on the predicate.
 * 
 * This function coordinates the execution of the CUDA kernels to perform the compaction. It first computes the number
 * of valid elements per block, then computes the prefix sum of valid block counts, and finally performs the actual
 * compaction by moving the valid elements to the output array.
 * 
 * @tparam T The type of elements in the input array.
 * @tparam Predicate The type of the predicate function.
 * @param d_input The input device array.
 * @param d_output The output device array to store the compacted elements.
 * @param length The length of the input array.
 * @param predicate The predicate function to check the validity of each element.
 * @param blockSize The number of threads per block for kernel execution.
 * @param f Lower bound parameter for the predicate.
 * @param l Upper bound parameter for the predicate.
 * @return The number of valid (compacted) elements.
 */
template <typename T,typename Predicate>
int compact(T* d_input, T* d_output,int length, Predicate predicate, int blockSize, int f, int l){
	static int stat_numBlocks = 0;
    int numBlocks = divup(length,blockSize);
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
