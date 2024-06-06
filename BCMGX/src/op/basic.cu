#include "op/basic.h"

#include "utility/setting.h"
#include "utility/utils.h"

// D=C+f*A*B, nrA>>ncA dgemv2
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

// D=C+f*A*B, nrA>>ncA dgemv2
void mydgemv2v(double* A, int nrA, int ncA, double* B, double* C, double f, double* D)
{
    int nthreads = BLOCKSIZE;
    int nblocks = ((nrA) + BLOCKSIZE - 1) / BLOCKSIZE;
    cdmv2v<<<nblocks, nthreads>>>(A, (unsigned int)nrA, (unsigned int)ncA, B, C, f, D);
    cudaDeviceSynchronize();
}

// A=B+f*C
__global__ void abfc(double* A, stype nrA, double* B, double* C, double f)
{
    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= (nrA)) {
        return;
    }
    A[tid] = B[tid] + f * C[tid];
}

// A=B+f*C
void myabfc(double* A, stype nrA, double* B, double* C, double f)
{
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrA) + BLOCKSIZE - 1) / BLOCKSIZE;
    abfc<<<nblocks, nthreads>>>(A, (stype)nrA, B, C, f);
    cudaDeviceSynchronize();
}

// y = a*x + b*y
__global__ void axpby_(double* x, stype nrx, double* y, double a, double b)
{
    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= (nrx)) {
        return;
    }
    if (b==1.0) {
    y[tid] += a * x[tid];
    } else {
    y[tid] = a * x[tid] + b * y[tid];
    }
}

// y = a*x + b*y
void my_axpby(double* x, stype nrx, double* y, double a, double b)
{
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrx) + BLOCKSIZE - 1) / BLOCKSIZE;
    axpby_<<<nblocks, nthreads>>>(x, (stype)nrx, y, a, b);
    cudaDeviceSynchronize();
}

// C = C + A*B, ncA>>nrA dgemm
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
void mydgemm(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C)
{
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrA * ncA) + BLOCKSIZE - 1) / BLOCKSIZE;
    cdmm<<<nblocks, nthreads, nrB * ncB * sizeof(double)>>>(A, nrA, ncA, B, nrB, ncB, C);
    cudaDeviceSynchronize();
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

__global__ void cdp(stype n, double* A, double* B, double* C)
{
    stype tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    double r = (tid < n) ? A[tid] * B[tid] : 0.;
    r = blockReduceSum(r);
    if (threadIdx.x == 0) {
        atomicAdd(C, r);
    }
}

void myddot(stype n, double* A, double* B, double* C)
{
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((n) + BLOCKSIZE - 1) / BLOCKSIZE;
    cudaMemset(C, 0, sizeof(C));
    cdp<<<nblocks, nthreads>>>((stype)n, A, B, C);
    cudaDeviceSynchronize();
}

// C=C+f*A*B, nrA>>ncA dgemv2
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

// dgemv2
void mydgemv(double* A, stype nrA, stype ncA, double* B, double* C, double f)
{
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrA) + BLOCKSIZE - 1) / BLOCKSIZE;
    cdmv<<<nblocks, nthreads>>>(A, (stype)nrA, (stype)ncA, B, C, f);
    cudaDeviceSynchronize();
}

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

void mynddot(stype nsp, stype n, double* A, double* B, double* C)
{
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((n) + BLOCKSIZE - 1) / BLOCKSIZE;
    cudaMemset(C, 0, sizeof(*C) * nsp);
    ncdp<<<nblocks, nthreads>>>((stype)nsp, (stype)n, A, B, C);
    cudaDeviceSynchronize();
}

// C=C+A*B; E=E+f*C*D
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
    double* sro = shb + (nrB * ncB);
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

void mydmmv(double* A, stype nrA, stype ncA, double* B, stype nrB, stype ncB, double* C, double* D, double* E, double f)
{
    stype nthreads = BLOCKSIZE;
    stype nblocks = ((nrA * ncA) + BLOCKSIZE - 1) / BLOCKSIZE;
    cdmmv<<<nblocks, nthreads, (BLOCKSIZE + (nrB * ncB)) * sizeof(double)>>>(A, (stype)nrA, (stype)ncA, B, (stype)nrB, (stype)ncB, C, D, E, f);
    cudaDeviceSynchronize();
}
