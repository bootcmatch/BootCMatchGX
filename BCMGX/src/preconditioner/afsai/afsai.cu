#include "choldc.h"
#include "datastruct/matrixItem.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "halo_communication/halo_communication.h"
#include "halo_communication/local_permutation.h"
#include "halo_communication/newoverlap.h"
#include "preconditioner/afsai/afsai.h"
#include "preconditioner/prec_setup.h"
#include "scalar_product_atomicAdd.h"
#include "utility/arrays.h"
#include "utility/globals.h"
#include "utility/handles.h"
#include "utility/memory.h"
#include "utility/precision.h"
#include "utility/profiling.h"
#include "utility/setting.h"

#include <cub/cub.cuh>
#include <cub/device/device_select.cuh>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define USEGPU
#undef USEASYNC
#undef USEPEER2PEER
#define AFSAI_NTHREADS "AFSAI_NTHREADS"
#define AFSAI_NBLOCKS "AFSAI_NBLOCKS"
#define ROWSINPA "ROWSINPA"
#define DROWSINPA "DROWSINPA"
#define MAXNTHREADS 1024
#define MAXNBLOCKS 2147483648
#define WARPSIZE 32
#define DEFAULTNGPU 1
#define MAXNGPU 1
#define NGPUENVV "NGPU_X_AFSAI"
#define MAXKSTEP 63
#define MAXSTEPS 2
#define AFSAISTEP 30
#define AFSAILFIL 1
#define AFSAIEPSILON 0.001

static int nrows, nterms;

typedef struct {
    int nscal; // scaling type
    int nstep; // max iterations [Kiter]
    int stepSize; // #retained entries from gradient [s]
    REALafsai epsilon;
} AFSAIParams;

__global__ void d_gatherFullSysAdapt(int nrows, int rowsinpa, int orow, int mmax, unsigned int wkrsppt, int* d_mrow_A, int* d_mrow_old_A, int* d_ia_A, int* d_ja_A, int* d_IWN, REALafsai* d_coef_A, REALafsai* d_full_A, REALafsai* d_rhs, int* d_done);

int kapGrad(int, int, int, int, int*, int*, REALafsai*, REALafsai*, int*, int*,
    int*, int*, REALafsai*);
int gatherFullsysAdapt(int, int, int, int*, int*, int*, REALafsai*, REALafsai**, REALafsai*);
extern "C" void dpotrf_(char*, int*, REALafsai*, int*, int*);
extern "C" void dcopy_(int*, REALafsai*, int*, REALafsai*, int*);
extern "C" void dpotrs_(char*, int*, int*, REALafsai*, int*, REALafsai*, int*, int*);
extern "C" REALafsai ddot_(int*, REALafsai*, int*, REALafsai*, int*);

__global__ void _shift_array(itype n, itype* v, gsstype shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }
    gsstype scratch = v[i];
    scratch += shift;
    v[i] = scratch;
}

void shift_array(itype n, itype* v, gsstype shift)
{
    GridBlock gb = gb1d(n, BLOCKSIZE);
    _shift_array<<<gb.g, gb.b>>>(n, v, shift);
}

struct RowIndexShifter {
    const itype shift;

    __host__ __device__ __forceinline__ explicit RowIndexShifter(const itype shift)
        : shift(shift)
    {
    }

    __host__ __device__ __forceinline__
        itype
        operator()(itype a) const
    {
        return a + shift;
    }
};

__global__ void f2d(double* t, REALafsai* s, int n)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n) {
        t[tid] = s[tid];
    }
}

__global__ void d2f(REALafsai* t, double* s, int n)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n) {
        t[tid] = s[tid];
    }
}

__global__ void finddiag(int* ia_A, int* ja_A, int* ind, int nrows, gstype row_shift)
{

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int irow = tid / WARPSIZE;
    int lane = tid % WARPSIZE;
    int lind, sind, rclz;
    if (irow < nrows) {
        lind = ia_A[irow + 1];
        sind = ia_A[irow] + lane;
        do {
            rclz = __clz(__ballot_sync(0xFFFFFFFF, (sind < lind) && (ja_A[sind] == (irow))));
            sind += WARPSIZE;
        } while (rclz == WARPSIZE);
        if (lane == 0) {
            ind[irow] = sind - 1 - rclz;
        }
    }
}

__global__ void scalediag(double* coef_A, double* diags, int* d_ind, int nrows)
{

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int irow = tid;
    if (irow < nrows) {
        diags[irow] = v(1.0) / SQRT(coef_A[d_ind[irow]]);
    }
}

__global__ void rescaleA(int* ia_A, int* ja_A, int* d_ind, REALafsai* coef_A, REALafsai* diags, int nrows)
{

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int irow = tid / WARPSIZE;
    int lane = tid % WARPSIZE;
    int rstart, rend, jcol;
    REALafsai fac;
    if (irow < nrows) {
        rstart = ia_A[irow] + lane;
        rend = ia_A[irow + 1];
        fac = diags[irow];
        for (int we = rstart; we < rend; we += WARPSIZE) {
            jcol = ja_A[we];
            coef_A[we] = fac * coef_A[we] * diags[jcol];
        }
    }
}

__global__ void createprc(REALafsai* coef_A, int* ind, int* JWN, int* IWN,
    int* mrow_A, REALafsai* rhs, REALafsai* DKap_A,
    int* ia_G, int* ja_G, REALafsai* coef_G,
    int mmax, unsigned int wkrsppt, int nrows, int offsetstartrow, int offcol)
{

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int irow = tid / WARPSIZE;
    int lane = tid % WARPSIZE;
    int ind_G;
    REALafsai scal_fac;
    if (irow < nrows) {
        irow += offsetstartrow;
        ind_G = JWN[irow] + irow - offsetstartrow;
        scal_fac = coef_A[ind[irow - offsetstartrow] + offcol] - DKap_A[irow];
        if (scal_fac > v(0.0)) {
            scal_fac = v(1.0) / SQRT(scal_fac);
            for (int i = lane; i < mrow_A[irow]; i += WARPSIZE) {
                coef_G[ind_G + i] = scal_fac * ((rhs + irow * mmax)[i]);
                ja_G[ind_G + i] = (IWN + irow * mmax)[i];
            }
        } else {
            printf("*** negative scaling factor: %d %d %f %f\n", tid, irow - offsetstartrow, coef_A[ind[irow - offsetstartrow] + offcol], DKap_A[irow]);
            scal_fac = v(1.0);
        }
        if (lane == 0) {
            coef_G[ind_G + mrow_A[irow]] = scal_fac;
            ja_G[ind_G + mrow_A[irow]] = irow;
            ia_G[irow + 1 - offsetstartrow] = ind_G + mrow_A[irow] + 1;
            if (irow == offsetstartrow) {
                ia_G[0] = 0;
            }
            if (irow == (offsetstartrow + nrows - 1)) {
                JWN[0] = ind_G + mrow_A[irow] + 1;
            }
        }
    }
}

__global__ void reorderA(REALafsai* sA, REALafsai* A,
    int nrows, int mmax, int mrow)
{
    REALafsai scratch = 1.0;
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int s = tid / (mrow * mrow);
    int e = tid % (mrow * mrow);
    int r = e / mrow;
    int c = e % mrow;
    if (c < r) {
        return; // only the upper part of the matrix
    }
    if (tid < (mrow * mrow * nrows)) {
        scratch = sA[(s * mmax * mmax) + r * mmax + c];
        A[(((((c * (c + 1)) / 2)) + r) * nrows) + s] = scratch;
    }
}

__global__ void reorderRHS(REALafsai* srhs, REALafsai* rhs,
    int nrows, int mmax, int mrow)
{
    REALafsai scratch;
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int s = tid / (mrow);
    int e = tid % (mrow);
    if (tid < (mrow * nrows)) {
        scratch = srhs[s * mmax + e];
        rhs[e * nrows + s] = scratch;
    }
}

__global__ void expandRHS(REALafsai* rhs, REALafsai* rhscratch,
    int nrows, int rowsinpa, int orow, int mrow, int done[])
{
    REALafsai scratch;
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int s = tid % rowsinpa;
    int e = tid / rowsinpa;
    if ((e < mrow) && ((s + orow) < nrows) && !done[s]) {
        scratch = rhscratch[e * rowsinpa + s];
        rhs[e * nrows + s + orow] = scratch;
    }
}

__global__ void reorderSRHS(REALafsai* srhs, REALafsai* rhs,
    int nrows, int mmax, int mrow)
{
    REALafsai scratch;
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int s = tid / (mrow);
    int e = tid % (mrow);
    if (tid < (mrow * nrows)) {
        scratch = rhs[e * nrows + s];
        srhs[s * mmax + e] = scratch;
    }
}

__global__ void check_done(REALafsai* scratch, REALafsai* DKap_A, REALafsai* DKap_old_A,
    int* mrow_A, int* mrow_old_A, int* done_A,
    int nrows, REALafsai threshold)
{

    int irow = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (irow < nrows) {
        if (!done_A[irow]) {
            if (mrow_A[irow] > mrow_old_A[irow]) {
                DKap_A[irow] = scratch[irow];
                if ((fabs(DKap_A[irow] - DKap_old_A[irow]) < threshold * DKap_old_A[irow]) || DKap_A[irow] == 0.0) {
                    done_A[irow] = 1;
                }
                DKap_old_A[irow] = DKap_A[irow];
            } else {
                done_A[irow] = 1;
            }
        }
    }
}

__global__ void d_kapGrad_merge(int nrows, int lastrow, int rowsinpa, int orow, int mmax,
    unsigned int wkrsppt, int mrow_A[], int lfil, int* ia, int* ja,
    REALafsai* coef, REALafsai* rhs, int* IWN, int* WI, REALafsai* WR,
    int* done);

void afsai_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p)
{
    bool concatenated_has_been_allocated = false;
    int GPUDevId = 0, i;
    int* ia_G;
    REALafsai* coef_A;
    double* diags;
    size_t freeMem, totMem;

    int rowsinpa = 1;
    int ngpu = 1;

    REALafsai T_prec = 0;
    int *d_iat_A[MAXNGPU], *d_ja_A[MAXNGPU];
    REALafsai* d_coef_A[MAXNGPU];
    _MPI_ENV;
    unsigned int nthreads = 1, nblocks = 1;

    if (getenv(AFSAI_NTHREADS)) {
        nthreads = atoi(getenv(AFSAI_NTHREADS));
    } else {
        nthreads = MAXNTHREADS;
    }

    if (nthreads < 1 || nthreads > MAXNTHREADS) {
        fprintf(stderr, "Invalid number of threads %d, must be > 0 and < %d\n",
            nthreads, MAXNTHREADS);
        exit(1);
    }

    nrows = Alocal->n;
    nterms = Alocal->nnz;

    if (ISMASTER) {
        printf("Scaling matrix ...\n");
    }
    int* d_ind = CUDA_MALLOC(int, nrows);
    nblocks = (nrows * WARPSIZE + nthreads - 1) / nthreads;
    if (nblocks < 1 || nblocks > MAXNBLOCKS) {
        fprintf(stderr, "Invalid number of blocks $d, must be > 0 and < %ld\n",
            MAXNBLOCKS);
        exit(1);
    }

    finddiag<<<nblocks, nthreads>>>(Alocal->row, Alocal->col, d_ind, nrows, Alocal->row_shift);
    diags = CUDA_MALLOC(double, nrows);
    nblocks = (nrows + nthreads - 1) / nthreads;
    scalediag<<<nblocks, nthreads>>>(Alocal->val, diags, d_ind, nrows);

    /* create a vector with the diag */
    vector<double>* v_d_diag = Vector::init<double>(Alocal->n, false, true);
    v_d_diag->val = diags;
    halo_sync(Alocal->halo, Alocal, v_d_diag, true);
    double* d_coef_ADouble;
    d_coef_ADouble = CUDA_MALLOC(double, nterms);
    vector<double>* v_d_coef_ADouble = Vector::init<double>(Alocal->n, false, true);
    v_d_coef_ADouble->val = d_coef_ADouble;
    CSRm::CSRscaleA_0(Alocal, v_d_diag, v_d_coef_ADouble, 1., 0.);
#if !USECOLSHIFT
    CSRm::shift_cols(Alocal, -(Alocal->col_shifted));
    Alocal->col_shifted = 0;
#endif

    itype* d_concatenatedRow;
    itype* d_concatenatedCol;
    vtype* d_concatenatedVal;
    int offsetstartrow = 0;
    int offcol = 0;
    itype* d_rowsToBeRequestedRet = NULL;
    if (nprocs == 1) {

        if (sizeof(REALafsai) == sizeof(double)) {
            coef_A = (REALafsai*)d_coef_ADouble;
        } else {
            coef_A = CUDA_MALLOC(REALafsai, nterms);
            nblocks = (nterms + nthreads - 1) / nthreads;
            d2f<<<nblocks, nthreads>>>(coef_A, d_coef_ADouble, nterms);
            CUDA_FREE(d_coef_ADouble);
        }

    } else {

        size_t missingItemsSize = 0;
        matrixItem_t* h_missingItems = NULL;

        CSR* AtempXexc = CSRm::init(Alocal->n, Alocal->m, Alocal->nnz, false, true, false, Alocal->full_n, Alocal->row_shift);
        AtempXexc->row = Alocal->row;
        AtempXexc->col = Alocal->col;
        AtempXexc->val = d_coef_ADouble;

        MatrixItemColumnIndexLessThanSelector matrixItemSelector(
#if USECOLSHIFT
            0
#else
            AtempXexc->row_shift
#endif
        );
        size_t rowsToBeRequestedSizeRet = 0;

        h_missingItems = CSRm::requestMissingRows(
            AtempXexc,
            log_file,
            &missingItemsSize,
            NULL,
            &rowsToBeRequestedSizeRet,
            &d_rowsToBeRequestedRet,
            matrixItemSelector,
            NnzColumnSelector(),
            USECOLSHIFT);

        // ---------------------------------------------------------------------------

        matrixItem_t* d_missingItems = copyArrayToDevice(h_missingItems, missingItemsSize);
        FREE(h_missingItems);

        if (log_file) {
            debugMatrixItems("d_missingItems", d_missingItems, missingItemsSize, true, log_file);
        }

        d_concatenatedRow = AtempXexc->row;
        d_concatenatedCol = AtempXexc->col;
        d_concatenatedVal = AtempXexc->val;
        itype concatenatedNnz = AtempXexc->nnz;
        itype concatenatedN = AtempXexc->n;
        itype missingItemsN = AtempXexc->row_shift;
        itype* d_missingItemsRow = NULL;
        itype* d_missingItemsCol = NULL;
        vtype* d_missingItemsVal = NULL;

        size_t missingItemsRowUniqueSize;
        itype* d_missingItemsRowUnique = NULL;

        if (missingItemsN > 0) {
            fillCsrFromMatrixItems(
                d_missingItems,
                missingItemsSize,
                missingItemsN,
                0, // row_shift
                &d_missingItemsRow,
                &d_missingItemsCol,
                &d_missingItemsVal,
                false, // Transposed,
                true // Allocate memory
            );

            if (log_file) {
                debugArray("d_missingItemsRow[%d] = %d\n", d_missingItemsRow, missingItemsN + 1, true, log_file);
                debugArray("d_missingItemsCol[%d] = %d\n", d_missingItemsCol, missingItemsSize, true, log_file);
                debugArray("d_missingItemsVal[%d] = %lf\n", d_missingItemsVal, missingItemsSize, true, log_file);
            }

            d_missingItemsRowUnique = deviceUnique(d_missingItemsRow, missingItemsN + 1, &missingItemsRowUniqueSize);

            if (log_file) {
                debugArray("d_missingItemsRowUnique[%d] = %d\n", d_missingItemsRowUnique, missingItemsRowUniqueSize, true, log_file);
            }
        }

        CUDA_FREE(d_missingItems);

        concatenatedNnz = missingItemsSize + AtempXexc->nnz;
        concatenatedN = AtempXexc->n;

        if (missingItemsN > 0) {
            concatenatedN = missingItemsRowUniqueSize - 1 + Alocal->n;

            itype shift = 0;
            CHECK_DEVICE(cudaMemcpy(
                &shift,
                d_missingItemsRowUnique + missingItemsRowUniqueSize - 1,
                sizeof(itype),
                cudaMemcpyDeviceToHost));

            offcol = shift;

            itype* d_shiftedRow = deviceMap<itype, itype, RowIndexShifter>(
                AtempXexc->row,
                AtempXexc->n + 1,
                RowIndexShifter(shift));

            d_concatenatedRow = concatArrays(
                d_missingItemsRowUnique, // First array
                missingItemsRowUniqueSize, // First array: len
                true, // First array: onDevice
                d_shiftedRow + 1, // Second array
                AtempXexc->n, // Second array: len
                true, // Second array: onDevice
                true // Returned array: onDevice
            );
            offsetstartrow += missingItemsRowUniqueSize - 1;

            CUDA_FREE(d_shiftedRow);

            d_concatenatedCol = concatArrays(
                d_missingItemsCol, // First array
                missingItemsSize, // First array: len
                true, // First array: onDevice
                AtempXexc->col, // Second array
                AtempXexc->nnz, // Second array: len
                true, // Second array: onDevice
                true // Returned array: onDevice
            );

            d_concatenatedVal = concatArrays(
                d_missingItemsVal, // First array
                missingItemsSize, // First array: len
                true, // First array: onDevice
                AtempXexc->val, // Second array
                AtempXexc->nnz, // Second array: len
                true, // Second array: onDevice
                true // Returned array: onDevice
            );
            concatenated_has_been_allocated = true;

            if (log_file) {
                debugArray("d_concatenatedRow[%d] = %d\n", d_concatenatedRow, concatenatedN + 1, true, log_file);
                debugArray("d_concatenatedCol[%d] = %d\n", d_concatenatedCol, concatenatedNnz, true, log_file);
                debugArray("d_concatenatedVal[%d] = %lf\n", d_concatenatedVal, concatenatedNnz, true, log_file);
            }

            CUDA_FREE(d_missingItemsRowUnique);
            CUDA_FREE(d_missingItemsRow);
            CUDA_FREE(d_missingItemsCol);
            CUDA_FREE(d_missingItemsVal);
        }

        nrows = concatenatedN;
        nterms = concatenatedNnz;
        if (sizeof(REALafsai) == sizeof(double)) {
            coef_A = (REALafsai*)d_concatenatedVal;
        } else {
            coef_A = CUDA_MALLOC(REALafsai, nterms);
            nblocks = (nterms + nthreads - 1) / nthreads;
            d2f<<<nblocks, nthreads>>>(coef_A, d_concatenatedVal, nterms);
        }
        if (myid) {
            printf("[Process %d] Shift col of %lu positions\n", myid, (Alocal->row_shift - offsetstartrow));
#if USECOLSHIFT
            shift_array(nterms, d_concatenatedCol, offsetstartrow);
#else
            shift_array(nterms, d_concatenatedCol, -(Alocal->row_shift - offsetstartrow));
#endif
            if (log_file) {
                debugArray("*** d_concatenatedCol[%d] = %d\n", d_concatenatedCol, nterms, true, log_file);
            }
        }
    }

    unsigned int avelen = (nterms + nrows - 1) / nrows;

    if (ISMASTER) {
        printf("Computing the preconditioner...\n");
    }

    int irow, lfil, mmax, maxIter, refine, istep, ind_G, nterms_G;
    int* IWN;
    REALafsai* rhs;

    REALafsai* DKap_A;
    int* mrow_A;

    unsigned int wkrsppt;
    int ntpm;
    cudaStream_t* stream;

    maxIter = AFSAISTEP;
    lfil = AFSAILFIL;
    mmax = maxIter * lfil;
    nterms_G = (mmax + 1) * nrows;
    REALafsai epsilon = AFSAIEPSILON;

    // Franc90
    // Print FSAI configuration
    if (ISMASTER) {
        printf("AFSAI parameters\n");
        printf("%10s: %10i\n", "nstep", maxIter);
        printf("%10s: %10i\n", "stepsize", lfil);
        printf("%10s: %10.3e\n", "eps", epsilon);
    }

    IWN = CUDA_MALLOC_HOST(int, mmax* nrows, true);
    DKap_A = CUDA_MALLOC_HOST(REALafsai, nrows, true);
    mrow_A = CUDA_MALLOC_HOST(int, nrows, true);
    rhs = CUDA_MALLOC_HOST(REALafsai, nrows * mmax, true);

    ia_G = MALLOC(int, nrows + 1, true);

    ind_G = 0;
    ia_G[0] = 0;

    void* d_cub_temp_storage = NULL;
    size_t cub_temp_storage_bytes = 0;

    cudaDeviceProp prop;
    int device;
    int drowsinpa;

    stream = MALLOC(cudaStream_t, ngpu);
    int nrowsxgpu = Alocal->n;

    int *d_iat_Prec, *d_ja_Prec;
    REALafsai* d_coef_Prec;

    REALafsai* d_full_A[MAXNGPU];
    REALafsai *d_rhs[MAXNGPU], *d_srhs[MAXNGPU], *d_rhscratch[MAXNGPU];
    REALafsai *d_DKap_A[MAXNGPU], *d_DKap_old_A[MAXNGPU], *d_scratch[MAXNGPU];
    int *d_IWN[MAXNGPU], *d_KWN[MAXNGPU], *d_WI[MAXNGPU];

    int *d_mrow_A[MAXNGPU], *d_mrow_old_A[MAXNGPU];
    int* d_done[MAXNGPU];
    REALafsai* d_WR[MAXNGPU];

    d_iat_Prec = CUDA_MALLOC(int, nrows + 1, true);
    d_ja_Prec = CUDA_MALLOC(int, nterms_G, true);
    d_coef_Prec = CUDA_MALLOC(REALafsai, nterms_G, true);

    wkrsppt = mmax * avelen;
    printf("AVELEN=%u, WKRSPPT=%d, NROWSXGPU=%d\n", avelen, wkrsppt, nrowsxgpu);
    for (i = 0; i < ngpu; i++) {
        d_rhs[i] = CUDA_MALLOC(REALafsai, nrows * mmax, true);
        d_srhs[i] = CUDA_MALLOC(REALafsai, nrows * mmax, true);
        d_DKap_A[i] = CUDA_MALLOC(REALafsai, nrows, true);
        d_DKap_old_A[i] = CUDA_MALLOC(REALafsai, nrows, true);
        d_scratch[i] = CUDA_MALLOC(REALafsai, nrows, true);
        d_KWN[i] = CUDA_MALLOC(int, nrows + 1, true);
        d_mrow_A[i] = CUDA_MALLOC(int, nrows, true);
        d_mrow_old_A[i] = CUDA_MALLOC(int, nrows, true);
        d_done[i] = CUDA_MALLOC(int, nrows, true);

        d_iat_A[i] = (nprocs == 1) ? Alocal->row : d_concatenatedRow;
        d_ja_A[i] = (nprocs == 1) ? Alocal->col : d_concatenatedCol;
        d_coef_A[0] = coef_A;

        MY_CUDA_CHECK(cudaGetDevice(&device));
        MY_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
#if 0
        printf("GPU %d: Global memory available on device %lu bytes\n", i, prop.totalGlobalMem);
        printf("GPU %d: Shared memory available per block %lu bytes\n", i, prop.sharedMemPerBlock);
#endif
        ntpm = (2 * prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor) / WARPSIZE;
        if (getenv(ROWSINPA)) {
            rowsinpa = atoi(getenv(ROWSINPA));
            // Franc90
            printf("GPU %d: numero di righe del precondizionatore calcolate in parallelo per kapGrad (ROWSINPA)=%d\n", i, rowsinpa);
        } else {
            rowsinpa = ntpm;
            printf("GPU %d: numero di righe del precondizionatore calcolate in parallelo per kapGrad=%d\n", i, rowsinpa);
        }
        if (rowsinpa < 1) {
            fprintf(stderr, "Invalid number of rows in parallel %d, must be > 0\n",
                rowsinpa);
            exit(1);
        }
        rowsinpa = MIN(rowsinpa, nrows);
        if (getenv(DROWSINPA)) {
            drowsinpa = atoi(getenv(DROWSINPA));
            // Franc90
            printf("Numero di righe del precondizionatore calcolate in parallelo per Cholesky (DROWSINPA)=%d\n", drowsinpa);
        } else {
            drowsinpa = nrows;
            printf("Numero di righe del precondizionatore calcolate in parallelo per Cholesky=%d\n",
                drowsinpa);
        }
        if (drowsinpa < 1) {
            fprintf(stderr, "Invalid number of rows in parallel %d, must be > 0\n",
                drowsinpa);
            exit(1);
        }
        d_full_A[i] = CUDA_MALLOC(REALafsai, drowsinpa * (mmax * (mmax + 1) / 2), true);
        d_rhscratch[i] = CUDA_MALLOC(REALafsai, drowsinpa * mmax, true);
        d_WR[i] = CUDA_MALLOC(REALafsai, 2 * rowsinpa * wkrsppt, true);

        MY_CUDA_CHECK(cudaMemGetInfo(&freeMem, &totMem));
        if ((freeMem / mmax) < avelen) {
            fprintf(stderr, "Not enough memory for IWN: required %d, available %zu\n",
                wkrsppt, freeMem);
            exit(1);
        }

        d_WI[i] = CUDA_MALLOC(int, 2 * rowsinpa * wkrsppt, true);
        d_IWN[i] = CUDA_MALLOC(int, mmax* nrows, true);
    }

    int* d_in = d_mrow_A[0];
    int* d_out = d_KWN[0];

    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage, cub_temp_storage_bytes,
        d_in, d_out, ((nrows > MAXNTHREADS) ? nrows : MAXNTHREADS));
    d_cub_temp_storage = CUDA_MALLOC_BYTES(void, cub_temp_storage_bytes);
    for (int g = 0; g < ngpu; g++) {
        int canaccesspeer;
        MY_CUDA_CHECK(cudaStreamCreate(&(stream[g])));
        for (int h = g; h < ngpu; h++) {
            if (h == g) {
                continue;
            }
            cudaDeviceCanAccessPeer(&canaccesspeer, h, 0);
            if (canaccesspeer) {
                MY_CUDA_CHECK(cudaDeviceEnablePeerAccess(h, 0));
            }
        }
    }

    BEGIN_PROF("AFSAI_TPREC");

    istep = 0;
    refine = 1;
    int startrow, lastrow, rowschnk;
    cudaFuncSetCacheConfig(d_kapGrad_merge, cudaFuncCachePreferShared);

    do {
#if defined(USEGPU)
        nblocks = ((rowsinpa * WARPSIZE) + nthreads - 1) / nthreads;
        for (i = 0; i < ngpu; i++) {
            startrow = offsetstartrow;
            lastrow = nrowsxgpu + startrow;
            rowschnk = nrowsxgpu;
            MY_CUDA_CHECK(cudaMemcpy(d_mrow_old_A[i] + startrow, d_mrow_A[i] + startrow,
                rowschnk * sizeof(int), cudaMemcpyDeviceToDevice));
            for (irow = startrow; irow < lastrow; irow += MIN(rowsinpa, rowschnk)) {
                d_kapGrad_merge<<<nblocks, nthreads>>>(nrows, lastrow, MIN(rowsinpa, rowschnk),
                    irow, mmax, wkrsppt, d_mrow_A[i], lfil, d_iat_A[i], d_ja_A[i],
                    d_coef_A[i], d_srhs[i], d_IWN[i], d_WI[i], d_WR[i], d_done[i]);
                cudaError_t err = cudaPeekAtLastError();
                if (cudaSuccess != err) {
                    fprintf(stderr, "Cuda error: in file '%s' in line %i : %s, nthreads=%d, nblocks=%d\n",
                        __FILE__, __LINE__, cudaGetErrorString(err), nthreads, nblocks);
                    exit(EXIT_FAILURE);
                }
            }
        }
        for (i = 0; i < ngpu; i++) {
            startrow = offsetstartrow;
            lastrow = nrowsxgpu + startrow;
            rowschnk = nrowsxgpu;
            for (irow = startrow; irow < lastrow; irow += MIN(drowsinpa, rowschnk)) {
                nblocks = ((MIN(drowsinpa, rowschnk)) + nthreads - 1) / nthreads;
                d_gatherFullSysAdapt<<<nblocks, nthreads>>>(lastrow, MIN(drowsinpa, rowschnk), irow, mmax, wkrsppt, d_mrow_A[i], d_mrow_old_A[i], d_iat_A[i], d_ja_A[i], d_IWN[i], d_coef_A[i], d_full_A[i], d_rhscratch[i], d_done[i]);
                nblocks = ((MIN(drowsinpa, rowschnk) * (istep + lfil)) + nthreads - 1) / nthreads;
                expandRHS<<<nblocks, nthreads>>>(d_srhs[i], d_rhscratch[i],
                    rowschnk, MIN(drowsinpa, rowschnk), irow - startrow, istep + lfil, d_done[i] + irow);
                choldc(MIN(drowsinpa, rowschnk), d_mrow_A[i], d_full_A[i], d_rhscratch[i],
                    d_done[i], irow);
                expandRHS<<<nblocks, nthreads>>>(d_rhs[i], d_rhscratch[i],
                    rowschnk, MIN(drowsinpa, rowschnk), irow - startrow, istep + lfil, d_done[i] + irow);
            }
        }
        for (i = 0; i < ngpu; i++) {
            startrow = offsetstartrow;
            rowschnk = nrowsxgpu;
            MY_CUDA_CHECK(cudaMemset(d_scratch[i], 0, rowschnk * sizeof(REALafsai)));
            nblocks = ((rowschnk * (istep + lfil)) + nthreads - 1) / nthreads;
            multiscalar_product<<<nblocks, nthreads>>>((d_srhs[i]),
                (d_rhs[i]),
                d_scratch[i], rowschnk, istep + lfil);
            MY_CUDA_CHECK(cudaMemset(d_srhs[i], 0, nrows * mmax * sizeof(REALafsai)));
            reorderSRHS<<<nblocks, nthreads>>>(d_srhs[i] + (startrow * mmax),
                d_rhs[i], rowschnk, mmax, istep + lfil);
            nblocks = ((rowschnk + nthreads - 1) / nthreads);
            check_done<<<nblocks, nthreads>>>(d_scratch[i],
                d_DKap_A[i] + startrow, d_DKap_old_A[i] + startrow,
                d_mrow_A[i] + startrow, d_mrow_old_A[i] + startrow,
                d_done[i] + startrow,
                rowschnk, epsilon);
        }
#endif
        istep += lfil;
        if ((istep / lfil) >= maxIter) {
            refine = 0;
        }
    } while (refine);

#if defined(USEGPU)
    MY_CUDA_CHECK(cudaMemcpy(d_rhs[0], d_srhs[0], nrows * mmax * sizeof(REALafsai), cudaMemcpyDeviceToDevice));
    for (i = 1; i < ngpu; i++) {
        rowschnk = (i == (ngpu - 1) ? (nrows - (nrowsxgpu * i)) : nrowsxgpu);
#if defined(USEPEER2PEER)
        MY_CUDA_CHECK(cudaMemcpyPeerAsync(d_rhs[0] + (mmax * nrowsxgpu * i), 0,
            d_srhs[i] + (mmax * nrowsxgpu * i), i,
            rowschnk * mmax * sizeof(REALafsai), stream[i]));
        MY_CUDA_CHECK(cudaMemcpyPeerAsync(d_IWN[0] + (mmax * nrowsxgpu * i), 0,
            d_IWN[i] + (mmax * nrowsxgpu * i), i,
            rowschnk * mmax * sizeof(int), stream[i]));
        MY_CUDA_CHECK(cudaMemcpyPeerAsync(d_DKap_A[0] + (nrowsxgpu * i), 0,
            d_DKap_A[i] + (nrowsxgpu * i), i,
            rowschnk * sizeof(REALafsai), stream[i]));
        MY_CUDA_CHECK(cudaMemcpyPeerAsync(d_mrow_A[0] + (nrowsxgpu * i), 0,
            d_mrow_A[i] + (nrowsxgpu * i), i,
            rowschnk * sizeof(int), stream[i]));
        MY_CUDA_CHECK(cudaSetDevice(i));
        MY_CUDA_CHECK(cudaDeviceSynchronize());
#else
        MY_CUDA_CHECK(cudaSetDevice(i));
#if defined(USEASYNC)
        MY_CUDA_CHECK(cudaMemcpyAsync(rhs + (mmax * nrowsxgpu * i),
            d_srhs[i] + (mmax * nrowsxgpu * i),
            rowschnk * mmax * sizeof(REALafsai),
            cudaMemcpyDeviceToHost, stream[i]));
        MY_CUDA_CHECK(cudaMemcpyAsync(IWN + (mmax * nrowsxgpu * i),
            d_IWN[i] + (mmax * nrowsxgpu * i),
            rowschnk * mmax * sizeof(int),
            cudaMemcpyDeviceToHost, stream[i]));
        MY_CUDA_CHECK(cudaMemcpyAsync(DKap_A + (nrowsxgpu * i),
            d_DKap_A[i] + (nrowsxgpu * i),
            rowschnk * sizeof(REALafsai),
            cudaMemcpyDeviceToHost, stream[i]));
        MY_CUDA_CHECK(cudaMemcpyAsync(mrow_A + (nrowsxgpu * i),
            d_mrow_A[i] + (nrowsxgpu * i),
            rowschnk * sizeof(int),
            cudaMemcpyDeviceToHost, stream[i]));
#else
        MY_CUDA_CHECK(cudaMemcpy(rhs + (mmax * nrowsxgpu * i),
            d_srhs[i] + (mmax * nrowsxgpu * i),
            rowschnk * mmax * sizeof(REALafsai),
            cudaMemcpyDeviceToHost));
        MY_CUDA_CHECK(cudaMemcpy(IWN + (mmax * nrowsxgpu * i),
            d_IWN[i] + (mmax * nrowsxgpu * i),
            rowschnk * mmax * sizeof(int),
            cudaMemcpyDeviceToHost));
        MY_CUDA_CHECK(cudaMemcpy(DKap_A + (nrowsxgpu * i),
            d_DKap_A[i] + (nrowsxgpu * i),
            rowschnk * sizeof(REALafsai),
            cudaMemcpyDeviceToHost));
        MY_CUDA_CHECK(cudaMemcpy(mrow_A + (nrowsxgpu * i),
            d_mrow_A[i] + (nrowsxgpu * i),
            rowschnk * sizeof(int),
            cudaMemcpyDeviceToHost));
#endif
#endif
    }
#if defined(USEPEER2PEER)
    MY_CUDA_CHECK(cudaSetDevice(GPUDevId));
#else
#if defined(USEASYNC)
    for (i = 1; i < ngpu; i++) {
        MY_CUDA_CHECK(cudaSetDevice(i));
        MY_CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
    if (ngpu > 1) {
        MY_CUDA_CHECK(cudaSetDevice(GPUDevId));
#if defined(USEASYNC)
        MY_CUDA_CHECK(cudaMemcpyAsync(d_rhs[0] + (mmax * nrowsxgpu),
            rhs + (mmax * nrowsxgpu),
            (nrows - nrowsxgpu) * mmax * sizeof(REALafsai),
            cudaMemcpyHostToDevice, stream[0]));
        MY_CUDA_CHECK(cudaMemcpyAsync(d_IWN[0] + (mmax * nrowsxgpu),
            IWN + (mmax * nrowsxgpu),
            (nrows - nrowsxgpu) * mmax * sizeof(int),
            cudaMemcpyHostToDevice, stream[0]));
        MY_CUDA_CHECK(cudaMemcpyAsync(d_DKap_A[0] + (nrowsxgpu),
            DKap_A + (nrowsxgpu),
            (nrows - nrowsxgpu) * sizeof(REALafsai),
            cudaMemcpyHostToDevice, stream[0]));
        MY_CUDA_CHECK(cudaMemcpyAsync(d_mrow_A[0] + (nrowsxgpu),
            mrow_A + (nrowsxgpu),
            (nrows - nrowsxgpu) * sizeof(int),
            cudaMemcpyHostToDevice, stream[0]));
        MY_CUDA_CHECK(cudaDeviceSynchronize());
#else
        MY_CUDA_CHECK(cudaMemcpy(d_rhs[0] + (mmax * nrowsxgpu),
            rhs + (mmax * nrowsxgpu),
            (nrows - nrowsxgpu) * mmax * sizeof(REALafsai), cudaMemcpyHostToDevice));
        MY_CUDA_CHECK(cudaMemcpy(d_IWN[0] + (mmax * nrowsxgpu),
            IWN + (mmax * nrowsxgpu),
            (nrows - nrowsxgpu) * mmax * sizeof(int), cudaMemcpyHostToDevice));
        MY_CUDA_CHECK(cudaMemcpy(d_DKap_A[0] + (nrowsxgpu),
            DKap_A + (nrowsxgpu),
            (nrows - nrowsxgpu) * sizeof(REALafsai), cudaMemcpyHostToDevice));
        MY_CUDA_CHECK(cudaMemcpy(d_mrow_A[0] + (nrowsxgpu),
            mrow_A + (nrowsxgpu),
            (nrows - nrowsxgpu) * sizeof(int),
            cudaMemcpyHostToDevice));
#endif
    }
#endif

    MY_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_cub_temp_storage, cub_temp_storage_bytes, d_in, d_out, nrows));

    nblocks = (nrows * WARPSIZE + nthreads - 1) / nthreads;
    if (nblocks < 1 || nblocks > MAXNBLOCKS) {
        fprintf(stderr, "Invalid number of blocks $d, must be > 0 and < %ld\n",
            MAXNBLOCKS);
        exit(1);
    }
#if defined(DEBUGMGPU)
    {
        FILE* fp;
        char filename[256];
        REALafsai *h_coef_A, *h_rhs, *h_DKap_A;
        int *h_ind, *h_KWN, *h_IWN, *h_mrow_A;
        h_ind = MALLOC(int, Alocal->n, true);
        MY_CUDA_CHECK(cudaMemcpy(h_ind, d_ind, sizeof(int) * Alocal->n, cudaMemcpyDeviceToHost));
        sprintf(filename, "l_ind_%d_%d", myid, nprocs);
        fp = fopen(filename, "w");
        for (int i = 0; i < Alocal->n; i++) {
            fprintf(fp, "%d\n", h_ind[i]);
        }
        fclose(fp);
        FREE(h_ind);
        h_KWN = MALLOC(int, nrows + 1);
        MY_CUDA_CHECK(cudaMemcpy(h_KWN, d_KWN[0], sizeof(int) * (nrows + 1), cudaMemcpyDeviceToHost));
        sprintf(filename, "l_KWN_f_%d_%d", myid, nprocs);
        fp = fopen(filename, "w");
        for (int i = 0; i < (nrows + 1); i++) {
            fprintf(fp, "%d\n", h_KWN[i]);
        }
        fclose(fp);
        FREE(h_KWN);
        h_mrow_A = MALLOC(int, nrows);
        MY_CUDA_CHECK(cudaMemcpy(h_mrow_A, d_mrow_A[0], sizeof(int) * (nrows), cudaMemcpyDeviceToHost));
        sprintf(filename, "l_mrow_A_f_%d_%d", myid, nprocs);
        fp = fopen(filename, "w");
        for (int i = 0; i < (nrows); i++) {
            fprintf(fp, "%d\n", h_mrow_A[i]);
        }
        fclose(fp);
        FREE(h_mrow_A);
        h_IWN = MALLOC(int, mmax* nrows);
        MY_CUDA_CHECK(cudaMemcpy(h_IWN, d_IWN[0], sizeof(int) * (mmax * nrows), cudaMemcpyDeviceToHost));
        sprintf(filename, "l_IWN_f_%d_%d", myid, nprocs);
        fp = fopen(filename, "w");
        for (int i = 0; i < (mmax * nrows); i++) {
            fprintf(fp, "%d\n", h_IWN[i]);
        }
        fclose(fp);
        FREE(h_IWN);
        h_coef_A = MALLOC(REALafsai, nterms);
        MY_CUDA_CHECK(cudaMemcpy(h_coef_A, d_coef_A[0], sizeof(REALafsai) * (nterms), cudaMemcpyDeviceToHost));
        sprintf(filename, "l_coef_A_f_%d_%d", myid, nprocs);
        fp = fopen(filename, "w");
        for (int i = 0; i < (nterms); i++) {
            fprintf(fp, "%g\n", h_coef_A[i]);
        }
        fclose(fp);
        FREE(h_coef_A);
        h_DKap_A = MALLOC(REALafsai, nrows);
        MY_CUDA_CHECK(cudaMemcpy(h_DKap_A, d_DKap_A[0], sizeof(REALafsai) * (nrows), cudaMemcpyDeviceToHost));
        sprintf(filename, "l_DKap_A_f%d_%d", myid, nprocs);
        fp = fopen(filename, "w");
        for (int i = 0; i < (nrows); i++) {
            fprintf(fp, "%g\n", h_DKap_A[i]);
        }
        fclose(fp);
        FREE(h_DKap_A);
        h_rhs = MALLOC(REALafsai, mmax * nrows);
        MY_CUDA_CHECK(cudaMemcpy(h_rhs, d_rhs[0], sizeof(REALafsai) * (mmax * nrows), cudaMemcpyDeviceToHost));
        sprintf(filename, "l_rhs_f_%d_%d", myid, nprocs);
        fp = fopen(filename, "w");
        for (int i = 0; i < (mmax * nrows); i++) {
            fprintf(fp, "%g\n", h_rhs[i]);
        }
        fclose(fp);
        FREE(h_rhs);
    }
#endif
    CUDA_FREE(d_rowsToBeRequestedRet);
    createprc<<<nblocks, nthreads>>>(d_coef_A[0], d_ind, d_KWN[0], d_IWN[0],
        d_mrow_A[0], d_rhs[0], d_DKap_A[0],
        d_iat_Prec, d_ja_Prec, d_coef_Prec,
        mmax, wkrsppt, nrows - offsetstartrow, offsetstartrow, offcol);
    MY_CUDA_CHECK(cudaMemcpy(&ind_G, d_KWN[0], sizeof(int), cudaMemcpyDeviceToHost));
#if USECOLSHIFT
    shift_array(ind_G, d_ja_Prec, (Alocal->row_shift - offsetstartrow));
#else
    shift_array(ind_G, d_ja_Prec, (Alocal->row_shift - offsetstartrow));
#endif
    if (log_file) {
        debugArray("*** d_ja_Prec[%d] = %d\n", d_ja_Prec, ind_G, true, log_file);
    }
#endif

    END_PROF("AFSAI_TPREC");
    printf("task %d, Time to compute preconditioner: %f s\n", myid, T_prec);

    printf("task %d nterms %d  nterms_G %d  ind_G %d\n", myid, nterms, nterms_G, ind_G);
    // Franc90
    printf("density %15.6f\n", (REALafsai)ind_G / (REALafsai)nterms);

    printf("Solving ...\n");

    for (int i = 0; i < ngpu; i++) {
        CUDA_FREE(d_rhscratch[i]);
        CUDA_FREE(d_full_A[i]);
        CUDA_FREE(d_IWN[i]);
        CUDA_FREE(d_WI[i]);
        CUDA_FREE(d_WR[i]);

        CUDA_FREE(d_rhs[i]);
        CUDA_FREE(d_srhs[i]);
        CUDA_FREE(d_DKap_A[i]);
        CUDA_FREE(d_DKap_old_A[i]);
        CUDA_FREE(d_scratch[i]);
        CUDA_FREE(d_KWN[i]);
        CUDA_FREE(d_mrow_A[i]);
        CUDA_FREE(d_mrow_old_A[i]);
        CUDA_FREE(d_done[i]);
        if (concatenated_has_been_allocated) {
            CUDA_FREE(d_iat_A[i]);
            CUDA_FREE(d_ja_A[i]);
        }
    }

    double* d_coef_PrecDouble;
    if (sizeof(REALafsai) == sizeof(double)) {
        d_coef_PrecDouble = (REALafsai*)d_coef_Prec;
    } else {
        d_coef_PrecDouble = CUDA_MALLOC(double, nterms_G);
        nblocks = (ind_G + MAXNTHREADS - 1) / MAXNTHREADS;
        f2d<<<nblocks, MAXNTHREADS>>>(d_coef_PrecDouble, d_coef_Prec, ind_G);
    }

    CSR* APrec = CSRm::init(Alocal->n, Alocal->m, ind_G, false, true, false, Alocal->full_n, Alocal->row_shift);
    APrec->row = d_iat_Prec;
    APrec->col = d_ja_Prec;
    APrec->val = d_coef_PrecDouble;

    // CSRm::printMM(APrec,"Precondizionatore");

    CSRm::shift_cols(APrec, -APrec->row_shift);
    APrec->col_shifted = -APrec->row_shift;
    halo_info haloPrec = haloSetup(APrec, NULL);
    APrec->halo = haloPrec;
    shrink_col(APrec, NULL);

    if (ISMASTER) {
        printf("Computing APrecT\n");
    }
    CSR* APrecT = CSRm::transpose(APrec, log_file);
    if (ISMASTER) {
        printf("APrecT computed\n");
    }

    // CSRm::printMM(APrec,"Precondizionatore");

    halo_info haloPrecT = haloSetup(APrecT, NULL);
    APrecT->halo = haloPrecT;
    shrink_col(APrecT, NULL);

    // CSRm::printMM(APrecT,"TraspostaPrecondizionatore");

// Franc90
#undef DUMPPREC
#if defined(DUMPPREC)
#define MAXFILENAME 1024
    int crow = 0;
    FILE* fpprec;
    char name[MAXFILENAME];
    snprintf(name, sizeof(name), "prec_j_coef_%d", getpid());
    fpprec = fopen(name, "w");
    if (fpprec == NULL) {
        fprintf(stderr, "Could not create %s\n", name);
        exit(1);
    }
    for (i = 0; i < ind_G; i++) {
        if (i == ia_G[crow + 1]) {
            crow++;
        };
        fprintf(fpprec, "%d %d %g\n", crow, ja_G[i], coef_G[i]);
    }
    fclose(fpprec);
#undef DUMPPRECT
#if defined(DUMPPRECT)
    double* coef_GT = MALLOC(double, nterms_G);
    MY_CUDA_CHECK(cudaMemcpy(ia_G, d_iat_PrecT, (nrows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    MY_CUDA_CHECK(cudaMemcpy(ja_G, d_ja_PrecT, nterms_G * sizeof(int), cudaMemcpyDeviceToHost));
    MY_CUDA_CHECK(cudaMemcpy(coef_GT, d_coef_PrecT, nterms_G * sizeof(double), cudaMemcpyDeviceToHost));
    snprintf(name, sizeof(name), "precT_j_coef_%d", getpid());
    crow = 0;
    fpprec = fopen(name, "w");
    if (fpprec == NULL) {
        fprintf(stderr, "Could not create %s\n", name);
        exit(1);
    }
    jcol = 0;
    for (i = 0; i < nrows; i++) {
        for (int j = ia_G[i]; j < ia_G[i + 1]; j++) {
            fprintf(fpprec, "%d %d %g\n", i, ja_G[j], coef_GT[jcol++]);
        }
    }
    fclose(fpprec);
#endif
#endif

#if !USECOLSHIFT
    CSRm::shift_cols(Alocal, -Alocal->row_shift);
    Alocal->col_shifted = -Alocal->row_shift;
#endif

    pr->afsai.Aprec = APrec;
    pr->afsai.AprecT = APrecT;

    CUDA_FREE(d_ind);
    CUDA_FREE(d_coef_ADouble);
    CUDA_FREE(diags);
    CUDA_FREE_HOST(IWN);
    CUDA_FREE_HOST(DKap_A);
    CUDA_FREE_HOST(mrow_A);
    CUDA_FREE_HOST(rhs);
    CUDA_FREE(d_cub_temp_storage);
    if (concatenated_has_been_allocated) {
        CUDA_FREE(d_coef_A[0]);
    }
}

void afsai_apply(handles* h, CSR* Alocal, vector<double>* v_d_r, vector<double>* v_d_pr, cgsprec* pr, const params& p, PrecOut* out)
{
    AfsaiData* afsaiP = &pr->afsai;
    static int first = 1;
    if (first) {
        afsaiP->v_d_dummy = Vector::init<double>(afsaiP->Aprec->n, true, true);
        first = 0;
    }

    CSRm::CSRVector_product_adaptive_miniwarp_witho(afsaiP->Aprec, v_d_r, afsaiP->v_d_dummy, 1., 0.);
    CSRm::CSRVector_product_adaptive_miniwarp_witho(afsaiP->AprecT, afsaiP->v_d_dummy, v_d_pr, 1., 0.);
}

void afsai_finalize(CSR* Alocal, cgsprec* pr, const params& p)
{
    CSRm::free(pr->afsai.Aprec);
    CSRm::free(pr->afsai.AprecT);
    Vector::free(pr->afsai.v_d_dummy);
}
