/** @file */
#include "op/basic.h"

#include "utility/profiling.h"
#include "utility/setting.h"
#include "utility/utils.h"

// D=C+f*A*B, nrA>>ncA dgemv2
/**
 * @brief Performs the operation D = C + f * A * B in parallel.
 *
 * This kernel computes the matrix-vector product of A and B, scales it by f,
 * and adds it to C. The result is stored in D.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the vector C.
 * @param f The scalar multiplier.
 * @param D Pointer to the output vector D.
 */
__global__ void cdmv2v(double* A, unsigned int nrA, unsigned int ncA, double* B, double* C, double f, double* D)
{
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= (nrA)) {
        return;
    }
    double ro = 0.;
    for (int k = 0; k < ncA; k++) {
        ro += A[(k * nrA) + tid] * B[k];
    }
    D[tid] = C[tid] + f * ro;
}

/**
 * @brief Wrapper function for the cdmv2v kernel.
 *
 * This function launches the cdmv2v kernel to perform the operation D = C + f * A * B.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the vector C.
 * @param f The scalar multiplier.
 * @param D Pointer to the output vector D.
 */
void mydgemv2v(double* A, int nrA, int ncA, double* B, double* C, double f, double* D)
{
    BEGIN_PROF(__FUNCTION__);
    int nthreads = BLOCKSIZE;
    int nblocks = ((nrA) + BLOCKSIZE - 1) / BLOCKSIZE;
    cdmv2v<<<nblocks, nthreads>>>(A, (unsigned int)nrA, (unsigned int)ncA, B, C, f, D);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

// A=B+f*C
/**
 * @brief Performs the operation A = B + f * C in parallel.
 *
 * This kernel computes the operation A = B + f * C for each element.
 *
 * @param A Pointer to the output vector A.
 * @param nrA The number of elements in vector A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the vector C.
 * @param f The scalar multiplier.
 */
__global__ void abfc(double* A, stype nrA, double* B, double* C, double f)
{
    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= (nrA)) {
        return;
    }
    A[tid] = B[tid] + f * C[tid];
}

// A=B+f*C
/**
 * @brief Wrapper function for the abfc kernel.
 *
 * This function launches the abfc kernel to perform the operation A = B + f * C.
 *
 * @param A Pointer to the output vector A.
 * @param nrA The number of elements in vector A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the vector C.
 * @param f The scalar multiplier.
 */
void myabfc(double* A, stype nrA, double* B, double* C, double f)
{
    BEGIN_PROF(__FUNCTION__);
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrA) + BLOCKSIZE - 1) / BLOCKSIZE;
    abfc<<<nblocks, nthreads>>>(A, (stype)nrA, B, C, f);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

// y = a*x + b*y
/**
 * @brief Performs the operation y = a * x + b * y in parallel.
 *
 * This kernel computes the operation for each element in the vectors.
 *
 * @param x Pointer to the input vector x.
 * @param nrx The number of elements in vector x.
 * @param y Pointer to the output vector y.
 * @param a The scalar multiplier for vector x.
 * @param b The scalar multiplier for vector y.
 */
__global__ void axpby_(double* x, stype nrx, double* y, double a, double b)
{
    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= (nrx)) {
        return;
    }
    if (b == 1.0) {
        y[tid] += a * x[tid];
    } else {
        y[tid] = a * x[tid] + b * y[tid];
    }
}

// y = a*x + b*y
/**
 * @brief Wrapper function for the axpby_ kernel.
 *
 * This function launches the axpby_ kernel to perform the operation y = a * x + b * y.
 *
 * @param x Pointer to the input vector x.
 * @param nrx The number of elements in vector x.
 * @param y Pointer to the output vector y.
 * @param a The scalar multiplier for vector x.
 * @param b The scalar multiplier for vector y.
 */
void my_axpby(double* x, stype nrx, double* y, double a, double b)
{
    BEGIN_PROF(__FUNCTION__);
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrx) + BLOCKSIZE - 1) / BLOCKSIZE;
    axpby_<<<nblocks, nthreads>>>(x, (stype)nrx, y, a, b);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

// C = C + A*B, ncA>>nrA dgemm
/**
 * @brief Performs the matrix multiplication C = C + A * B in parallel.
 *
 * This kernel computes the matrix product of A and B and adds it to C.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the matrix B.
 * @param nrB The number of rows in matrix B.
 * @param ncB The number of columns in matrix B.
 * @param C Pointer to the output matrix C.
 */
__global__ void cdmm(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C)
{
    extern __shared__ double shb[];
    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= (nrA * ncA)) {
        return;
    }
    if (threadIdx.x < (nrB * ncB)) {
        shb[threadIdx.x] = B[threadIdx.x];
    }
    __syncthreads();
    stype row, col;
    double ro = 0.;
    row = tid / ncB;
    col = tid % ncB;
#pragma unroll
    for (stype k = 0; k < nrB; k++) {
        ro += A[(k * nrA) + row] * shb[k + ncB * col];
    }
    C[col * nrA + row] += ro;
}

// dgemm
/**
 * @brief Wrapper function for the cdmm kernel.
 *
 * This function launches the cdmm kernel to perform the matrix multiplication C = C + A * B.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the matrix B.
 * @param nrB The number of rows in matrix B.
 * @param ncB The number of columns in matrix B.
 * @param C Pointer to the output matrix C.
 */
void mydgemm(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C)
{
    BEGIN_PROF(__FUNCTION__);
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrA * ncA) + BLOCKSIZE - 1) / BLOCKSIZE;
    cdmm<<<nblocks, nthreads, nrB * ncB * sizeof(double)>>>(A, nrA, ncA, B, nrB, ncB, C);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

__device__ __inline__ double warpReduce(double val)
{
    for (stype offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ double blockReduceSum(double val)
{
    static __shared__ double sharedouble[1024];
    const unsigned int ttid = threadIdx.x;
    sharedouble[ttid] = val;
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (ttid < d) {
            sharedouble[ttid] += sharedouble[ttid + d];
        }
    }
    return (ttid == 0) ? sharedouble[0] : 0.;
}

/**
 * @brief Computes the dot product of two vectors A and B.
 *
 * This kernel computes the dot product and stores the result in C.
 *
 * @param n The number of elements in the vectors.
 * @param A Pointer to the first input vector A.
 * @param B Pointer to the second input vector B.
 * @param C Pointer to the output variable where the result is stored.
 */
__global__ void cdp(stype n, double* A, double* B, double* C)
{
    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    double r = (tid < n) ? A[tid] * B[tid] : 0.;
    r = blockReduceSum(r);
    if (threadIdx.x == 0) {
        atomicAdd(C, r);
    }
}

/**
 * @brief Wrapper function for the cdp kernel.
 *
 * This function launches the cdp kernel to compute the dot product of A and B.
 *
 * @param n The number of elements in the vectors.
 * @param A Pointer to the first input vector A.
 * @param B Pointer to the second input vector B.
 * @param C Pointer to the output variable where the result is stored.
 */
void myddot(stype n, double* A, double* B, double* C)
{
    BEGIN_PROF(__FUNCTION__);
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((n) + BLOCKSIZE - 1) / BLOCKSIZE;
    cudaMemset(C, 0, sizeof(C));
    cdp<<<nblocks, nthreads>>>((stype)n, A, B, C);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

// C=C+f*A*B, nrA>>ncA dgemv2
/**
 * @brief Performs the operation C = C + f * A * B in parallel.
 *
 * This kernel computes the matrix-vector product of A and B, scales it by f,
 * and adds it to C.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the output vector C.
 * @param f The scalar multiplier.
 */
__global__ void cdmv(double* A, stype nrA, stype ncA, double* B, double* C, double f)
{
    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= (nrA)) {
        return;
    }
    double ro = 0.;
    for (stype k = 0; k < ncA; k++) {
        ro += A[(k * nrA) + tid] * B[k];
    }
    C[tid] += f * ro;
}

/**
 * @brief Wrapper function for the cdmv kernel.
 *
 * This function launches the cdmv kernel to perform the operation C = C + f * A * B.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the vector B.
 * @param C Pointer to the output vector C.
 * @param f The scalar multiplier.
 */
void mydgemv(double* A, stype nrA, stype ncA, double* B, double* C, double f)
{
    BEGIN_PROF(__FUNCTION__);
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrA) + BLOCKSIZE - 1) / BLOCKSIZE;
    cdmv<<<nblocks, nthreads>>>(A, (stype)nrA, (stype)ncA, B, C, f);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

/**
 * @brief Computes the dot product of multiple vectors A and B.
 *
 * This kernel computes the dot product for multiple segments and stores the result in C.
 *
 * @param nsp The number of segments.
 * @param n The number of elements in each vector.
 * @param A Pointer to the input matrix A.
 * @param B Pointer to the input vector B.
 * @param C Pointer to the output vector where results are stored ```cpp
 * @param C Pointer to the output vector where results are stored.
 */
__global__ void ncdp(stype nsp, stype n, double* A, double* B, double* C)
{
    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    double rb = (tid < n) ? B[tid] : 0.;
    double r;
    for (stype i = 0; i < nsp; i++) {
        r = (tid < n) ? A[tid + (i * n)] * rb : 0.;
        r = blockReduceSum(r);
        if (threadIdx.x == 0) {
            atomicAdd(C + i, r);
        }
    }
}

/**
 * @brief Wrapper function for the ncdp kernel.
 *
 * This function launches the ncdp kernel to compute the dot product of multiple segments.
 *
 * @param nsp The number of segments.
 * @param n The number of elements in each vector.
 * @param A Pointer to the input matrix A.
 * @param B Pointer to the input vector B.
 * @param C Pointer to the output vector where results are stored.
 */
void mynddot(stype nsp, stype n, double* A, double* B, double* C)
{
    BEGIN_PROF(__FUNCTION__);
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((n) + BLOCKSIZE - 1) / BLOCKSIZE;
    cudaMemset(C, 0, sizeof(*C) * nsp);
    ncdp<<<nblocks, nthreads>>>((stype)nsp, (stype)n, A, B, C);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

// C=C+A*B; E=E+f*C*D
/**
 * @brief Performs the operation C = C + A * B and E = E + f * C * D in parallel.
 *
 * This kernel computes the matrix product of A and B, adds it to C, and scales the result
 * by f before adding it to E.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the matrix B.
 * @param nrB The number of rows in matrix B.
 * @param ncB The number of columns in matrix B.
 * @param C Pointer to the output matrix C.
 * @param D Pointer to the vector D.
 * @param E Pointer to the output vector E.
 * @param f The scalar multiplier.
 */
__global__ void cdmmv(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C, double* D, double* E, double f)
{
    extern __shared__ double shb[];

    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid >= (nrA * ncA)) {
        return;
    }
    if (threadIdx.x < (nrB * ncB)) {
        shb[threadIdx.x] = B[threadIdx.x];
    }
    __syncthreads();
    stype row, col;
    row = tid / ncB;
    col = tid % ncB;
    double ro = C[col * nrA + row];
#pragma unroll
    for (stype k = 0; k < nrB; k++) {
        ro += A[(k * nrA) + row] * shb[k + ncB * col];
    }

    C[col * nrA + row] = ro;

    ro *= D[col];
#if USEONLYATOM
    atomicAdd(E + row, f * ro);
#else
    sro[threadIdx.x] = ro;
    __syncthreads();
    if (col == 0) {
        for (stype i = threadIdx.x + 1; i < (threadIdx.x + ncB); i++) {
            if (i >= BLOCKSIZE) {
                break;
            }
            ro += sro[i];
        }
        atomicAdd(E + row, f * ro);
    } else {
        if (blockIdx.x > 0 && threadIdx.x < ncB && threadIdx.x < col) {
            atomicAdd(E + row, f * ro);
        }
    }
#endif
}

/**
 * @brief Wrapper function for the cdmmv kernel.
 *
 * This function launches the cdmmv kernel to perform the operations C = C + A * B and E = E + f * C * D.
 *
 * @param A Pointer to the matrix A.
 * @param nrA The number of rows in matrix A.
 * @param ncA The number of columns in matrix A.
 * @param B Pointer to the matrix B.
 * @param nrB The number of rows in matrix B.
 * @param ncB The number of columns in matrix B.
 * @param C Pointer to the output matrix C.
 * @param D Pointer to the vector D.
 * @param E Pointer to the output vector E.
 * @param f The scalar multiplier.
 */
void mydmmv(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C, double* D, double* E, double f)
{
    BEGIN_PROF(__FUNCTION__);
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrA * ncA) + BLOCKSIZE - 1) / BLOCKSIZE;
    cdmmv<<<nblocks, nthreads, (BLOCKSIZE + (nrB * ncB)) * sizeof(double)>>>(A, (stype)nrA, (stype)ncA, B, (stype)nrB, (stype)ncB, C, D, E, f);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}
