#ifndef _CUDAMACRO_H
#define _CUDAMACRO_H

#define MY_CUDA_CHECK( call) {                                    \
	    cudaError err = call;                                                    \
	    if( cudaSuccess != err) {                                                \
		            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
					                    __FILE__, __LINE__, cudaGetErrorString( err) );              \
		            exit(EXIT_FAILURE);                                                  \
		        } }

#define MY_CHECK_ERROR(errorMessage) {                                    \
	    cudaError_t err = cudaGetLastError();                                    \
	    if( cudaSuccess != err) {                                                \
		            fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s, nthreads=%d, nblocks=%d, n=%d, number=%d.\n",    \
					                    errorMessage, __FILE__, __LINE__, cudaGetErrorString( err), nThreads, nBlocks, n, number );\
		            exit(EXIT_FAILURE);                                                  \
		        }                                                                        \
	    }

#define MY_CUBLAS_CHECK( call) {                                    \
            cublasStatus_t err = call;                                                    \
            if( CUBLAS_STATUS_SUCCESS != err) {                                                \
                            fprintf(stderr, "Cublas error in file '%s' in line %i : %d.\n",        \
                                                            __FILE__, __LINE__,  err);              \
                            exit(EXIT_FAILURE);                                                  \
                        } }
#define MY_CUSPARSE_CHECK( call) {                                    \
            cusparseStatus_t err = call;                                                    \
            if( CUSPARSE_STATUS_SUCCESS != err) {                                                \
                            fprintf(stderr, "Cusparse error in file '%s' in line %i : %d.\n",        \
                                                            __FILE__, __LINE__,  err);              \
                            exit(EXIT_FAILURE);                                                  \
                        } }



#endif
