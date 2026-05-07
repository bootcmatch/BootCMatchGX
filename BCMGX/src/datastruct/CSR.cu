#include "CSR.h"

#include "halo_communication/extern2.h"
#include "halo_communication/halo_communication.h"
#include "op/spspmpi.h"

#include "datastruct/matrixItem.h"
#include "utility/MatrixItemSender.h"
#include "utility/arrays.h"
#include "utility/col8.h"
#include "utility/cuCompactorXT.cuh"
#include "utility/cudamacro.h"
#include "utility/deviceMax.h"
#include "utility/deviceMemEq.h"
#include "utility/devicePartition.h"
#include "utility/devicePrefixSum.h"
#include "utility/deviceSort.h"
#include "utility/deviceUnique.h"
#include "utility/distribute.h"
#include "utility/hostPartition.h"
#include "utility/hostSort.h"
#include "utility/profiling.h"

#include <cub/cub.cuh>
#include <string.h>
#include <unordered_set>

#define DEBUG 0
#define MAXMATRIXFILENAME 256

int CSRm::choose_mini_warp_size(CSR* A)
{
    int density = A->nnz / A->n;

    if (density < MINI_WARP_THRESHOLD_2) {
        return 2;
    } else if (density < MINI_WARP_THRESHOLD_4) {
        return 4;
    } else if (density < MINI_WARP_THRESHOLD_8) {
        return 8;
    } else if (density < MINI_WARP_THRESHOLD_16) {
        return 16;
    } else {
        return 32;
    }
}

CSR* CSRm::init(stype n, gstype m, stype nnz, bool allocate_mem, bool on_the_device, bool is_symmetric, gstype full_n, gstype row_shift)
{
    // ---------- Pico ----------
    if (n <= 0 || m <= 0 || nnz <= 0) {
        fprintf(stderr, "error in CSRm::init:\n\tint  n: %d  m: %lu  nnz: %d\n", n, m, nnz);
    }
    assert(n > 0);
    assert(m > 0);
    assert(nnz > 0);
    // --------------------------

    CSR* A = NULL;

    // on the host
    A = MALLOC(CSR, 1, true);

    A->nnz = nnz;
    A->n = n;
    A->m = m;
    A->full_m = m;

    A->on_the_device = on_the_device;
    A->is_symmetric = false;
    A->custom_alloced = false;

    A->full_n = full_n;
    A->row_shift = row_shift;

    A->rows_to_get = NULL;

    A->shrinked_flag = false;
    A->shrinked_col = NULL;
    A->shrinked_m = m;
    A->halo.init = false;
    A->col_shifted = 0;

    A->post_local = 0;
    A->bitcolsize = 0;
    A->bitcol = NULL;
    A->col = NULL;
    A->col8 = NULL;
    if (allocate_mem) {
        if (on_the_device) {
            // on the device
            A->val = CUDA_MALLOC(vtype, nnz, true);
            A->col8 = CUDA_MALLOC(gsstype, nnz, true);
            A->col = CUDA_MALLOC(itype, nnz, true);
            A->row = CUDA_MALLOC(itype, n + 1, true);
        } else {
            // on the host
            A->val = MALLOC(vtype, nnz, true);
            A->col8 = MALLOC(gsstype, nnz, true);
            A->col = MALLOC(itype, nnz, true);
            A->row = MALLOC(itype, n + 1, true);
        }
    }

    return A;
}

void CSRm::printMM(CSR* A, FILE* fp)
{
    _MPI_ENV;
    CSR* A_ = NULL;
    if (A->on_the_device) {
        A_ = CSRm::copyToHost(A);
    } else {
        A_ = A;
    }

    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%d %lu %d "
                "%ld %ld %ld\n",
        A_->n, A_->m, A_->nnz,
        A_->row_shift, A_->full_n, A_->col_shifted);
    for (int i = 0; i < A_->n; i++) {
        for (int j = A_->row[i]; j < A_->row[i + 1]; j++) {
            // fprintf(fp, "%lu %ld %ld %ld %lf\n",
            fprintf(fp, "%lu %ld %lf\n",
                i + 1 + A_->row_shift,
                A_->col[j] + 1 - A_->col_shifted,
                // A_->col8[j],
                // A_->col8[j] + 1 - A_->col_shifted,
                A_->val[j]);
        }
    }

    if (A->on_the_device) {
        CSRm::free(A_);
    }
}

void CSRm::printMM(CSR* A, char* name, bool appendMyIdAndNprocs)
{
    _MPI_ENV;
    char localname[MAXMATRIXFILENAME];
    if (appendMyIdAndNprocs) {
        snprintf(localname, sizeof(localname), "%s_%d_%d", name, myid, nprocs);
    } else {
        snprintf(localname, sizeof(localname), "%s", name);
    }
    FILE* fp = fopen(localname, "w");
    printMM(A, fp);
    fclose(fp);
}

void CSRm::printMMsimple(CSR* A, FILE* fp)
{
    _MPI_ENV;
    CSR* A_ = NULL;
    if (A->on_the_device) {
        A_ = CSRm::copyToHost(A);
    } else {
        A_ = A;
    }

    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%d %lu %d "
                "\n",
        A_->n, A_->m, A_->nnz);
    for (int i = 0; i < A_->n; i++) {
        for (int j = A_->row[i]; j < A_->row[i + 1]; j++) {
            fprintf(fp, "%d %d %15le\n",
                i + 1,
                A_->col[j] + 1,
                A_->val[j]);
        }
    }

    if (A->on_the_device) {
        CSRm::free(A_);
    }
}

void CSRm::printMMsimple(CSR* A, char* name, bool appendMyIdAndNprocs)
{
    _MPI_ENV;
    char localname[MAXMATRIXFILENAME];
    if (appendMyIdAndNprocs) {
        snprintf(localname, sizeof(localname), "%s_%d_%d", name, myid, nprocs);
    } else {
        snprintf(localname, sizeof(localname), "%s", name);
    }
    FILE* fp = fopen(localname, "w");
    printMMsimple(A, fp);
    fclose(fp);
}



void CSRm::printMMsc(CSR* A, char* name, bool appendMyIdAndNprocs)
{
    _MPI_ENV;
    CSR* A_ = NULL;
    if (A->on_the_device) {
        A_ = CSRm::copyToHost(A);
    } else {
        A_ = A;
    }

    char localname[MAXMATRIXFILENAME];
    if (appendMyIdAndNprocs) {
        snprintf(localname, sizeof(localname), "%s_%d_%d", name, myid, nprocs);
    } else {
        snprintf(localname, sizeof(localname), "%s", name);
    }
    FILE* fp = fopen(localname, "w");
    if (fp == NULL) {
        DIE("Could not open %s", localname);
    }
    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%d %lu %d "
                "%ld %ld %ld\n",
        A_->n, A_->m, A_->nnz,
        A_->row_shift, A_->full_n, A_->col_shifted);
    for (int i = 0; i < A_->n; i++) {
        for (int j = A_->row[i]; j < A_->row[i + 1]; j++) {
            fprintf(fp, "%lu %ld %d %lf\n",
                i + 1 + A_->row_shift,
                A_->col[j] + 1 - A_->col_shifted,
                A_->shrinked_col[j] + 1,
                A_->val[j]);
        }
    }
    fclose(fp);

    if (A->on_the_device) {
        CSRm::free(A_);
    }
}

void CSRm::print(CSR* A, int type, int limit, FILE* fp, bool show_header, bool show_footer)
{
    CSR* A_ = NULL;

    if (A->on_the_device) {
        A_ = CSRm::copyToHost(A);
    } else {
        A_ = A;
    }

    switch (type) {
    case 0:
        if (show_header) {
            fprintf(fp, "ROW: %d (%lu)\n\t", A_->n, A_->full_n);
        }
        if (limit == 0) {
            limit = A_->full_n + 1;
        }
        for (int i = 0; i < limit; i++) {
            fprintf(fp, "%3d ", A_->row[i]);
        }
        break;
    case 1:
        if (show_header) {
            fprintf(fp, "COL:\n");
        }
        if (limit == 0) {
            limit = A_->nnz;
        }
        for (int i = 0; i < limit; i++) {
            fprintf(fp, "%d\n", A_->col[i]);
        }
        break;
    case 2:
        if (show_header) {
            fprintf(fp, "VAL:\n");
        }
        if (limit == 0) {
            limit = A_->nnz;
        }
        for (int i = 0; i < limit; i++) {
            fprintf(fp, "%14.12g\n", A_->val[i]);
        }
        break;
    case 3:
        if (show_header) {
            fprintf(fp, "MATRIX_Form:\n");
        }
        for (int i = 0; i < A_->n; i++) {
            fprintf(fp, "\t");
            int next_col = 0;
            for (int j = A_->row[i]; j < A_->row[i+1]; j++) {
                int icol = A_->col[j] + A_->row_shift;
                while (next_col < icol) {
                    fprintf(fp, "%g ", 0.0);
                    next_col++;
                }
                fprintf(fp, "%g ", A_->val[j]);
                next_col = icol + 1;
            }
            while (next_col < A_->m) {
                fprintf(fp, "%g ", 0.0);
                next_col++;
            }

            fprintf(fp, "\n");
        }
        break;
    case 4:
        if (show_header) {
            fprintf(fp, "boolMATRIX_Form:\n");
        }
        for (int i = 0; i < A_->n; i++) {
            fprintf(fp, "\t");
            for (int j = 0; j < A_->m; j++) {
                if (j % 32 == 0) {
                    fprintf(fp, "| ");
                }
                int flag = 0, temp = A_->row[i];
                for (temp = A_->row[i]; flag == 0 && (i != (A_->n) - 1 ? temp < (A_->row[i + 1]) : temp < A_->nnz); temp++) {
                    if (A_->col[temp] == j) {
                        fprintf(fp, "\033[0;31mX\033[0m ");
                        flag = 1;
                    }
                }
                if (flag == 0) {
                    fprintf(fp, "O ");
                }
            }
            fprintf(fp, "\n");
        }
        break;
    case 5:
        if (show_header) {
            fprintf(fp, "SHRINKED COL:\n");
        }
        if (limit == 0) {
            limit = A_->shrinked_m;
        }
        for (int i = 0; i < limit; i++) {
            fprintf(fp, "%d\n", A_->shrinked_col[i]);
        }
        break;
    }
    if (show_footer) {
        fprintf(fp, "\n\n");
    }

    if (A->on_the_device) {
        CSRm::free(A_);
    }
}

void CSRm::free_rows_to_get(CSR* A)
{
    if (A->rows_to_get != NULL) {
        FREE(A->rows_to_get->rcvprow);
        FREE(A->rows_to_get->whichprow);
        FREE(A->rows_to_get->rcvpcolxrow);
        FREE(A->rows_to_get->scounts);
        FREE(A->rows_to_get->displs);
        FREE(A->rows_to_get->displr);
        FREE(A->rows_to_get->rcounts2);
        FREE(A->rows_to_get->scounts2);
        FREE(A->rows_to_get->displs2);
        FREE(A->rows_to_get->displr2);
        FREE(A->rows_to_get->rcvcntp);
        FREE(A->rows_to_get->P_n_per_process);
        if (A->rows_to_get->nnz_per_row_shift != NULL) {
            Vector::free(A->rows_to_get->nnz_per_row_shift);
        }
        FREE(A->rows_to_get);
    }
    A->rows_to_get = NULL;
}

void CSRm::free(CSR* A)
{
    if (A->on_the_device) {
        CUDA_FREE(A->val);
        CUDA_FREE(A->col8);
        //        CUDA_FREE(A->col);
        CUDA_FREE(A->row);
        CUDA_FREE(A->shrinked_col);
    } else {
        FREE(A->val);
        FREE(A->col);
        if (A->col8 != NULL) {
            FREE(A->col8);
        }
        FREE(A->row);
    }
    if (A->rows_to_get != NULL) {
        FREE(A->rows_to_get->rcvprow);
        FREE(A->rows_to_get->whichprow);
        FREE(A->rows_to_get->rcvpcolxrow);
        FREE(A->rows_to_get->scounts);
        FREE(A->rows_to_get->displs);
        FREE(A->rows_to_get->displr);
        FREE(A->rows_to_get->rcounts2);
        FREE(A->rows_to_get->scounts2);
        FREE(A->rows_to_get->displs2);
        FREE(A->rows_to_get->displr2);
        FREE(A->rows_to_get->rcvcntp);
        FREE(A->rows_to_get->P_n_per_process);
        if (A->rows_to_get->nnz_per_row_shift != NULL) {
            Vector::free(A->rows_to_get->nnz_per_row_shift);
        }
        FREE(A->rows_to_get);
        A->rows_to_get = NULL;
    }

    CUDA_FREE(A->bitcol);
    A->bitcol = NULL;

    if (A->halo.init == true) {
        // Free the halo_info halo halo_info halo;
        Vector::free(A->halo.to_receive);
        Vector::free(A->halo.to_receive_d);
        FREE(A->halo.to_receive_counts);
        FREE(A->halo.to_receive_spls);
        CUDA_FREE_HOST(A->halo.what_to_receive);
        CUDA_FREE(A->halo.what_to_receive_d);
        Vector::free(A->halo.to_send);
        Vector::free(A->halo.to_send_d);
        FREE(A->halo.to_send_counts);
        FREE(A->halo.to_send_spls);
        CUDA_FREE_HOST(A->halo.what_to_send);
        CUDA_FREE(A->halo.what_to_send_d);
        A->halo.init = false;
    }

    Vector::free(A->os.loc_rows);
    A->os.loc_rows = NULL;
    Vector::free(A->os.needy_rows);
    A->os.needy_rows = NULL;
}

void shift_cpucol(itype* Arow, itype* Acol, unsigned int n, stype row_shift)
{
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = Arow[i]; j < Arow[i + 1]; j++) {
            Acol[j] += row_shift;
        }
    }
}

CSR* CSRm::clone(CSR* A)
{
    CSR *ret = CSRm::init(A->n, A->m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);

    if (A->on_the_device) {
        ret->row = CUDA_MALLOC(itype, A->n + 1, false);
        CHECK_DEVICE(cudaMemcpy(ret->row, A->row, (A->n + 1) * sizeof(itype), cudaMemcpyDeviceToDevice));

        ret->col = CUDA_MALLOC(itype, A->nnz, false);
        CHECK_DEVICE(cudaMemcpy(ret->col, A->col, A->nnz * sizeof(itype), cudaMemcpyDeviceToDevice));

        ret->col8 = CUDA_MALLOC(gsstype, A->nnz, false);
        CHECK_DEVICE(cudaMemcpy(ret->col8, A->col8, A->nnz * sizeof(gsstype), cudaMemcpyDeviceToDevice));

        ret->val = CUDA_MALLOC(vtype, A->nnz, false);
        CHECK_DEVICE(cudaMemcpy(ret->val, A->val, A->nnz * sizeof(vtype), cudaMemcpyDeviceToDevice));
    } else {
        ret->row = MALLOC(itype, A->n + 1, false);
        memcpy(ret->row, A->row, (A->n + 1) * sizeof(itype));

        ret->col = MALLOC(itype, A->nnz, false);
        memcpy(ret->col, A->col, A->nnz * sizeof(itype));

        ret->col8 = MALLOC(gsstype, A->nnz, false);
        memcpy(ret->col8, A->col8, A->nnz * sizeof(gsstype));

        ret->val = MALLOC(vtype, A->nnz, false);
        memcpy(ret->val, A->val, A->nnz * sizeof(vtype));
    }
    ret->custom_alloced = true;
    ret->col_shifted = A->col_shifted;

    return ret;
}

__global__ void _diagonalScaling(stype An,
    itype* Arow, vtype* Aval,
    vtype* retval,
    vtype* D,
    int warpSize)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int irow = tid / warpSize;
    int lane = tid % warpSize;
    int rstart, rend;

    if (irow < An) {
        rstart = Arow[irow] + lane;
        rend = Arow[irow + 1];
        for (int we = rstart; we < rend; we += warpSize) {
            retval[we] = Aval[we] / D[irow];
        }
    }
}

/**
 * Returns B = D^-1 A.
 */
CSR *CSRm::diagonalScaling(CSR *A, vector<vtype> *D) {
    assert(A->on_the_device);
    assert(D->on_the_device);
    assert(A->full_n == A->m);
    assert(A->n == D->n);
    assert(!A->is_symmetric);

    CSR* ret = CSRm::clone(A);
    
    int warpSize = CSRm::choose_mini_warp_size(A);
    GridBlock gb = getKernelParams(A->n * warpSize); // One mini-warp per row

    _diagonalScaling<<<gb.g, gb.b>>>(A->n,
        A->row, A->val,
        ret->val,
        D->val,
        warpSize);
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_DEVICE(err);

    return ret;
}

void CSRm::diagonalScalingInPlace(CSR *A, vector<vtype> *D) {
    assert(A->on_the_device);
    assert(D->on_the_device);
    assert(A->full_n == A->m);
    assert(A->n == D->n);
    assert(!A->is_symmetric);

    CSR* ret = A;
    
    int warpSize = CSRm::choose_mini_warp_size(A);
    GridBlock gb = getKernelParams(A->n * warpSize); // One mini-warp per row

    _diagonalScaling<<<gb.g, gb.b>>>(A->n,
        A->row, A->val,
        ret->val,
        D->val,
        warpSize);
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_DEVICE(err);
}

__global__ void _rowSum(stype An,
    itype* Arow, vtype* Aval,
    vtype* retval,
    vtype* D,
    int warpSize)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int irow = tid / warpSize;
    int lane = tid % warpSize;
    int rstart, rend;

    if (irow < An) {
        rstart = Arow[irow] + lane;
        rend = Arow[irow + 1];
        for (int we = rstart; we < rend; we += warpSize) {
            retval[we] = Aval[we] / D[irow];
        }
    }
}

vtype CSRm::localInfinityNorm(CSR *A) {
    vector<vtype> *sum = CSRm::absoluteRowSum(A, NULL);
    vtype ret = deviceMax(sum->val, sum->n);
    Vector::free(sum);
    return ret;
}

vtype CSRm::globalInfinityNorm(CSR *A) {
    vector<vtype> *sum = CSRm::absoluteRowSum(A, NULL);
    vtype localMax = deviceMax(sum->val, sum->n);
    Vector::free(sum);
    vtype ret;
    CHECK_MPI(MPI_Allreduce(&localMax, &ret, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    return ret;
}

CSR* CSRm::copyToDevice(CSR* A)
{
    assert(!A->on_the_device);

    itype n, nnz;
    gstype m;
    n = A->n;
    m = A->m;

    nnz = A->nnz;

    // allocate CSR matrix on the device memory
    CSR* A_d = CSRm::init(n, m, nnz, true, true, A->is_symmetric, A->full_n, A->row_shift);
    A_d->full_m = A->full_m;
    A_d->col_shifted = A->col_shifted;

    cudaError_t err;
    err = cudaMemcpy(A_d->val, A->val, nnz * sizeof(vtype), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);
    err = cudaMemcpy(A_d->row, A->row, (n + 1) * sizeof(itype), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);
    err = cudaMemcpy(A_d->col8, A->col8, nnz * sizeof(gsstype), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);
    err = cudaMemcpy(A_d->col, A->col, nnz * sizeof(itype), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);

    return A_d;
}

CSR* CSRm::copyToHost(CSR* A_d)
{
    assert(A_d->on_the_device);

    itype n, m, nnz;

    n = A_d->n;
    m = A_d->m;

    nnz = A_d->nnz;

    // allocate CSR matrix on the device memory
    CSR* A = CSRm::init(n, m, nnz, true, false, A_d->is_symmetric, A_d->full_n, A_d->row_shift);
    A->full_m = A_d->full_m;
    A->col_shifted = A_d->col_shifted;

    cudaError_t err;

    assert(A->val);
    assert(A_d->val);
    err = cudaMemcpy(A->val, A_d->val, nnz * sizeof(vtype), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    assert(A->row);
    assert(A_d->row);
    err = cudaMemcpy(A->row, A_d->row, (n + 1) * sizeof(itype), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    assert(A->col);
    assert(A_d->col);
    err = cudaMemcpy(A->col, A_d->col, nnz * sizeof(itype), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    if (A_d->col8) {
        assert(A->col8);
        err = cudaMemcpy(A->col8, A_d->col8, nnz * sizeof(gsstype), cudaMemcpyDeviceToHost);
        CHECK_DEVICE(err);
    }

    if (A_d->shrinked_m && A_d->shrinked_col) {
        A->shrinked_col = MALLOC(itype, nnz, false);
        err = cudaMemcpy(A->shrinked_col, A_d->shrinked_col, nnz * sizeof(itype), cudaMemcpyDeviceToHost);
        CHECK_DEVICE(err);
    } else {
        A->shrinked_col = NULL;
    }

    return A;
}

__global__ void _shift_cols(itype n, itype* col, gsstype shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }
    gsstype scratch = col[i];
    scratch += shift;
    col[i] = scratch;
}

__global__ void _shift_cols8(itype n, gsstype* col8, gsstype shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }
    gsstype scratch = col8[i];
    scratch += shift;
    col8[i] = scratch;
}

void CSRm::shift_cols(CSR* A, gsstype shift)
{
    if (A->on_the_device) {
        GridBlock gb = gb1d(A->nnz, BLOCKSIZE);
        _shift_cols<<<gb.g, gb.b>>>(A->nnz, A->col, shift);
        _shift_cols8<<<gb.g, gb.b>>>(A->nnz, A->col8, shift);
    } else {
        itype n = A->nnz;
        for (itype i = 0; i < n; i++) {
            A->col[i] += shift;
            A->col8[i] += shift;
        }
    }
}

__global__ void _prepare_column_ptr(stype A_nrows, itype* A_row, itype* A_col, itype* T_row)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < A_nrows) {
        itype start_idx = A_row[tid];
        itype end_idx = A_row[tid + 1];
        // Count the number of nnz per column
        for (itype i = start_idx; i < end_idx; i++) {
            atomicAdd(&T_row[A_col[i] + 1], 1);
        }
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void _write_row_indices(stype A_nrows, itype* A_row, itype* T_col)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < A_nrows) {
        itype start_idx = A_row[tid];
        itype end_idx = A_row[tid + 1];
        for (itype i = start_idx; i < end_idx; i++) {
            T_col[i] = tid;
        }
        tid += blockDim.x * gridDim.x;
    }
}

template <int OP_TYPE>
__global__ void CSRm::_CSR_vector_mul_mini_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    vtype T_i = 0.;

    for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
        if (OP_TYPE == 0) {
            T_i += (alpha * A_val[j]) * __ldg(&x[A_col[j]]);
        } else if (OP_TYPE == 1) {
            T_i += A_val[j] * __ldg(&x[A_col[j]]);
        } else if (OP_TYPE == 2) {
            T_i += -A_val[j] * __ldg(&x[A_col[j]]);
        }
    }

    for (int k = MINI_WARP_SIZE >> 1; k > 0; k = k >> 1) {
        T_i += __shfl_down_sync(warp_mask, T_i, k);
    }

    if (lane == 0) {
        if (OP_TYPE == 0) {
            y[warp] = T_i + (beta * y[warp]);
        } else if (OP_TYPE == 1) {
            y[warp] = T_i;
        } else if (OP_TYPE == 2) {
            y[warp] = T_i + y[warp];
        }
    }
}

template <int OP_TYPE>
__global__ void CSRm::_CSR_vector_mul_mini_warp_indirect(itype n, itype* rows, unsigned offset, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    warp = rows[warp];

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    vtype T_i = 0.;

    for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
        if (OP_TYPE == 0) {
            T_i += (alpha * A_val[j]) * __ldg(&x[A_col[j] - offset]);
        } else if (OP_TYPE == 1) {
            T_i += A_val[j] * __ldg(&x[A_col[j] - offset]);
        } else if (OP_TYPE == 2) {
            T_i += -A_val[j] * __ldg(&x[A_col[j] - offset]);
        }
    }

    for (int k = MINI_WARP_SIZE >> 1; k > 0; k = k >> 1) {
        T_i += __shfl_down_sync(warp_mask, T_i, k);
    }

    if (lane == 0) {
        if (OP_TYPE == 0) {
            y[warp] = T_i + (beta * y[warp]);
        } else if (OP_TYPE == 1) {
            y[warp] = T_i;
        } else if (OP_TYPE == 2) {
            y[warp] = T_i + y[warp];
        }
    }
}

template <int OP_TYPE>
__global__ void CSRm::_CSR_scale_mini_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    vtype T_i = 0.;

    for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
        if (OP_TYPE == 0) {
            T_i = (alpha * A_val[j]) * __ldg(&x[A_col[j]]);
        } else if (OP_TYPE == 1) {
            T_i = A_val[j] * __ldg(&x[A_col[j]]);
        } else if (OP_TYPE == 2) {
            T_i = -A_val[j] * __ldg(&x[A_col[j]]);
        }
        y[j] = T_i * __ldg(&x[warp]);
    }
}

__global__ void CSRm::_CSR_vector_mul_mini_indexed_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y, itype* to_comp, itype shift, int op_type)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;
    if (warp >= n) {
        return;
    }

    int target = warp;
    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    vtype T_i = 0.;
    warp = to_comp[warp] /* -shift */;
    if (op_type == 0) {
        for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
            T_i += (alpha * A_val[j]) * __ldg(&x[A_col[j]]);
        }
    }
    if (op_type == 1) {
        for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
            T_i += A_val[j] * __ldg(&x[A_col[j]]);
        }
    }
    if (op_type == 2) {
        for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
            T_i += -A_val[j] * __ldg(&x[A_col[j]]);
        }
    }

    for (int k = MINI_WARP_SIZE >> 1; k > 0; k = k >> 1) {
        T_i += __shfl_down_sync(warp_mask, T_i, k);
    }

    if (lane == 0) {
        if (op_type == 0) {
            y[target] = T_i + (beta * y[warp]);
        } else if (op_type == 1) {
            y[target] = T_i;
        } else if (op_type == 2) {
            y[target] = T_i + y[warp];
        }
    }
}

vector<vtype>* CSRm::CSRVector_product_adaptive_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, vtype alpha, vtype beta)
{
    BEGIN_PROF(__FUNCTION__);

    itype n = A->n;

    int density = A->nnz / A->n;

    int min_w_size;
#if 0
    if (density < MINI_WARP_THRESHOLD_2) {
        min_w_size = 2;
    } else if (density < MINI_WARP_THRESHOLD_4) {
        min_w_size = 4;
    } else if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 4;
    } else {
        min_w_size = 16;
    }
#else
    if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 2;
    } else {
    	min_w_size = 4;
    }
#endif

    if (y == NULL) {
        assert(beta == 0.);
        y = Vector::init<vtype>(n, true, true); // OK perchè vettore di output
    }

    GridBlock gb = gb1d(n, BLOCKSIZE, true, min_w_size);

    if (alpha == 1. && beta == 0.) {
        CSRm::_CSR_vector_mul_mini_warp<1><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
    } else if (alpha == -1. && beta == 1.) {
        CSRm::_CSR_vector_mul_mini_warp<2><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
    } else {
        CSRm::_CSR_vector_mul_mini_warp<0><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
    }
    cudaDeviceSynchronize();

    END_PROF(__FUNCTION__);
    return y;
}

vector<vtype>* CSRm::CSRVector_product_adaptive_indirect_row_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, itype n, itype* rows, cudaStream_t stream, unsigned int offset, vtype alpha, vtype beta)
{

    int density = A->nnz / A->n;

    int min_w_size;

#if 0
    if (density < MINI_WARP_THRESHOLD_2) {
        min_w_size = 2;
    } else if (density < MINI_WARP_THRESHOLD_4) {
        min_w_size = 4;
    } else if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 4;
    } else {
        min_w_size = 16;
    }
#else
    if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 2;
    } else {
    	min_w_size = 4;
    }
#endif


    if (y == NULL) {
        assert(beta == 0.);
        y = Vector::init<vtype>(n, true, true); // OK perchè vettore di output
    }

    GridBlock gb = gb1d(n, BLOCKSIZE, true, min_w_size);

    if (alpha == 1. && beta == 0.) {
        CSRm::_CSR_vector_mul_mini_warp_indirect<1><<<gb.g, gb.b, 0, stream>>>(n, rows, offset, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
    } else if (alpha == -1. && beta == 1.) {
        CSRm::_CSR_vector_mul_mini_warp_indirect<2><<<gb.g, gb.b, 0, stream>>>(n, rows, offset, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
    } else {
        CSRm::_CSR_vector_mul_mini_warp_indirect<0><<<gb.g, gb.b, 0, stream>>>(n, rows, offset, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
    }
#if 0
    cudaStreamSynchronize(stream);
#endif
    return y;
}

vector<vtype>* CSRm::CSRscale_adaptive_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, vtype alpha, vtype beta)
{
    itype n = A->n;

    int density = A->nnz / A->n;

    int min_w_size;

#if 0
    if (density < MINI_WARP_THRESHOLD_2) {
        min_w_size = 2;
    } else if (density < MINI_WARP_THRESHOLD_4) {
        min_w_size = 4;
    } else if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 4;
    } else {
        min_w_size = 16;
    }
#else
    if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 2;
    } else {
    	min_w_size = 4;
    }
#endif

    if (y == NULL) {
        assert(beta == 0.);
        y = Vector::init<vtype>(n, true, true); // OK perchè vettore di output
    }

    GridBlock gb = gb1d(n, BLOCKSIZE, true, min_w_size);

    if (alpha == 1. && beta == 0.) {
        CSRm::_CSR_scale_mini_warp<1><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
    } else if (alpha == -1. && beta == 1.) {
        CSRm::_CSR_scale_mini_warp<2><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
    } else {
        CSRm::_CSR_scale_mini_warp<0><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
    }
    cudaDeviceSynchronize();
    return y;
}

__global__ void _vector_sync(vtype* local_x, itype local_n, vtype* what_to_receive_d, itype receive_n, itype post_local, vtype* x, itype x_n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < x_n) {
        if (id < post_local) {
            x[id] = what_to_receive_d[id];
        } else {
            if (id < post_local + local_n) {
                x[id] = local_x[id - post_local];
            } else {
                x[id] = what_to_receive_d[id - local_n];
            }
        }
    }
}

vector<vtype>* CSRm::CSRVector_product_adaptive_miniwarp_new(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha, vtype beta)
{
    BEGIN_PROF(__FUNCTION__);

    _MPI_ENV;

    if (nprocs == 1) {
        vector<vtype>* w_ = NULL;
        if (w == NULL) {
            w_ = Vector::init<vtype>(A->n, true, true);
            Vector::fillWithValue(w_, 0.);
        } else {
            w_ = w;
        }
        if (A->shrinked_flag == 1) {
            A->col = A->shrinked_col;
        }
        CSRm::CSRVector_product_adaptive_miniwarp(A, local_x, w_, alpha, beta);
        END_PROF(__FUNCTION__);
        return (w_);
    }

    assert(A->shrinked_flag == 1);

    CSR* A_ = CSRm::init(A->n, (gstype)A->shrinked_m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);
    A_->row = A->row;
    A_->val = A->val;
    A_->col = A->shrinked_col;

    // ----------------------------------------- temp check -----------------------------------------
    //  assert( A->halo.to_receive_n + local_x->n == A_->m ); /* Massimo March 13 2024. To fix problem with Xtent */
    // ----------------------------------------------------------------------------------------------
    int post_local = A->post_local;

    vector<vtype>* x_ = NULL;
    if (A->halo.to_receive_n > 0) {
        x_ = Vector::init<vtype>(A_->m, false, true);
        if (A_->m > xsize) {
            CUDA_FREE(xvalstat);
            xsize = A_->m;
            xvalstat = CUDA_MALLOC(vtype, xsize, true);
        }
        x_->val = xvalstat;

        // ===============================================================
        // stype nr_of_cols = A->n;
        // if (A->full_n < A->m) {
        //     nr_of_cols = A->m / nprocs;
        //     if (myid == nprocs - 1) {
        //         nr_of_cols += A->m % nprocs;
        //     }
        // }

        stype nr_of_cols = x_->n - A->halo.to_receive_d->n;
        // ===============================================================
        ASSERT(nr_of_cols + A->halo.to_receive_d->n == x_->n);
        GridBlock gb = gb1d(A_->m, BLOCKSIZE);
        _vector_sync<<<gb.g, gb.b>>>(local_x->val, nr_of_cols, A->halo.what_to_receive_d, A->halo.to_receive_d->n, post_local, x_->val, x_->n);
    } else {
        x_ = local_x;
    }

    vector<vtype>* w_ = NULL;
    if (w == NULL) {
        w_ = Vector::init<vtype>(A->n, true, true);
        Vector::fillWithValue(w_, 1.);
    } else {
        w_ = w;
    }
    CSRm::CSRVector_product_adaptive_miniwarp(A_, x_, w_, alpha, beta);

    // --------------------------------------- print -----------------------------------------
    //   vector<vtype> *what_to_receive_d = Vector::init<vtype>(A->halo.to_receive_n, false, true);
    //   what_to_receive_d->val = A->halo.what_to_receive_d;
    //
    //   PICO_PRINT(  \
    //     fprintf(fp, "A->halo:\n\tto_receive: "); Vector::print(A->halo.to_receive, -1, fp); \
    //     fprintf(fp, "\tto_send: "); Vector::print(A->halo.to_send, -1, fp); \
    //     fprintf(fp, "post_local = %d\n", post_local); \
    //     fprintf(fp, "what_to_receive_d: "); Vector::print(what_to_receive_d, -1, fp); \
    //     fprintf(fp, "local_x: "); Vector::print(local_x, -1, fp); \
    //     fprintf(fp, "x_: "); Vector::print(x_, -1, fp); \
    //   )
    //
    //   FREE(what_to_receive_d);
    // ---------------------------------------------------------------------------------------

    if (A->halo.to_receive_n > 0) {
        FREE(x_);
    }
    A_->col = NULL;
    A_->row = NULL;
    A_->val = NULL;
    FREE(A_);

    END_PROF(__FUNCTION__);
    return (w_);
}

#define SYNCSOL_TAG 4321
#define MAXNTASKS 4096
#define USESTREAM 1

#if DEBUG
vector<vtype>* CSRm::CSRVector_product_adaptive_miniwarp_witho(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha, vtype beta, const nostd::source_location& loc)
#else
vector<vtype>* CSRm::CSRVector_product_adaptive_miniwarp_witho(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha, vtype beta)
#endif
{
    BEGIN_PROF(__FUNCTION__);

    _MPI_ENV;

    // fprintf(stderr, "myid = %d, A->name = %s, full_n = %d, m = %d\n", myid, A->name, A->full_n, A->m);

    if (nprocs == 1) {
        vector<vtype>* w_ = NULL;
        if (w == NULL) {
            w_ = Vector::init<vtype>(A->n, true, true);
            Vector::fillWithValue(w_, 0.);
        } else {
            w_ = w;
        }
        if (A->shrinked_flag == 1) {
            A->col = A->shrinked_col;
        }
        CSRm::CSRVector_product_adaptive_miniwarp(A, local_x, w_, alpha, beta);
        END_PROF(__FUNCTION__);
        return (w_);
    }

    if (A->os.loc_n == 0 && A->os.needy_n == 0) {
        setupOverlapped(A);
    }

    assert(A->shrinked_flag == 1);

    assert(A->halo.init);

    if (A->halo.to_receive_n == 0 && A->halo.to_send_n == 0) {
        vector<vtype>* ret = CSRm::CSRVector_product_adaptive_miniwarp_new(A, local_x, w, alpha, beta);
        END_PROF(__FUNCTION__);
        return ret;
    }

    CSR* A_ = CSRm::init(A->n, (gstype)A->shrinked_m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);
    A_->row = A->row;
    A_->val = A->val;
    A_->col = A->shrinked_col;

    overlapped os = A->os;

    assert(os.loc_n != 0 || os.needy_n != 0);

    halo_info hi = A->halo;
    static MPI_Request requests[MAXNTASKS];
    static int ntr = 0;

    int post_local = A->post_local;

    vector<vtype>* x_ = NULL;
    if (A->halo.to_receive_n > 0) {
        x_ = Vector::init<vtype>(A_->m, false, true);
        if (A_->m > xsize) {
            CUDA_FREE(xvalstat);
            xsize = A_->m;
            xvalstat = CUDA_MALLOC(vtype, xsize, true);
        }
        x_->val = xvalstat;
    } else {
        x_ = local_x;
    }

    vector<vtype>* w_ = NULL;
    if (w == NULL) {
        w_ = Vector::init<vtype>(A->n, true, true);
        Vector::fillWithValue(w_, 1.);
    } else {
        w_ = w;
    }

    cudaStreamSynchronize(*(os.streams->comm_stream));
    if (hi.to_send_n) {
        assert(hi.what_to_send != NULL);
        assert(hi.what_to_send_d != NULL);
        GridBlock gb = gb1d(hi.to_send_n, BLOCKSIZE);
#if defined(USESTREAM)
        _getToSend_new<<<gb.g, gb.b, 0, *(os.streams->comm_stream)>>>(hi.to_send_d->n, local_x->val, hi.what_to_send_d, hi.to_send_d->val, A->row_shift);
        CHECK_DEVICE(cudaMemcpyAsync(hi.what_to_send, hi.what_to_send_d, hi.to_send_n * sizeof(vtype), cudaMemcpyDeviceToHost, *(os.streams->comm_stream)));
#else
        _getToSend_new<<<gb.g, gb.b>>>(hi.to_send_d->n, local_x->val, hi.what_to_send_d, hi.to_send_d->val, A->row_shift);
        CHECK_DEVICE(cudaMemcpy(hi.what_to_send, hi.what_to_send_d, hi.to_send_n * sizeof(vtype), cudaMemcpyDeviceToHost));
#endif
    }

    if (os.loc_n) {
        // start compute local
        CSRm::CSRVector_product_adaptive_indirect_row_miniwarp(A_, local_x, w_,
            os.loc_n, os.loc_rows->val, *(os.streams->local_stream), post_local, alpha, beta);
    }

    int j = 0;
    for (int t = 0; t < nprocs; t++) {
        if (t == myid) {
            continue;
        }
        if (hi.to_receive_counts[t] > 0) {
            CHECK_MPI(
                MPI_Irecv(hi.what_to_receive + (hi.to_receive_spls[t]), hi.to_receive_counts[t], VTYPE_MPI, t, SYNCSOL_TAG, MPI_COMM_WORLD, requests + j));
            j++;
            if (j == MAXNTASKS) {
                DIE("Too many tasks in matrix-vector product, max is %d\n", MAXNTASKS);
            }
        }
    }

    ntr = j;
    if (hi.to_send_n) {
        cudaStreamSynchronize(*(os.streams->comm_stream));
    }

    for (int t = 0; t < nprocs; t++) {
        if (t == myid) {
            continue;
        }
        if (hi.to_send_counts[t] > 0) {
            CHECK_MPI(MPI_Isend(hi.what_to_send + (hi.to_send_spls[t]), hi.to_send_counts[t], VTYPE_MPI, t, SYNCSOL_TAG, MPI_COMM_WORLD, requests + ntr + t));
        }
    }

    // copy received data
    if (hi.to_receive_n) {
        if (ntr > 0) {
            CHECK_MPI(MPI_Waitall(ntr, requests, MPI_STATUSES_IGNORE));
        }
        assert(hi.what_to_receive != NULL);
        assert(hi.what_to_receive_d != NULL);
        
#if defined(USESTREAM)
        CHECK_DEVICE(cudaMemcpyAsync(hi.what_to_receive_d, hi.what_to_receive, hi.to_receive_n * sizeof(vtype), cudaMemcpyHostToDevice, *(os.streams->comm_stream)));
        #if DEBUG
        {
            const char *mat_name = strlen(A->name) ? A->name : (
                (A->full_n == A->m) ? "Aded" :
                (A->full_n > A->m ? "Pded" : "Rded"));
            fprintf(stderr,
                "[%s:%d] myid = %d, %s = %p, full_n = %d, m = %d, shrinked_m = %d, n = %d, halo.to_receive_d->n = %d, x_->n = %d, post_local = %d, local_x->val = %p, halo.what_to_receive_d = %p, x_->val = %p\n",
                loc.file_name(), loc.line(), myid, mat_name, A, A->full_n, A->m, A->shrinked_m, A->n, A->halo.to_receive_d->n, x_->n, post_local, local_x->val, A->halo.what_to_receive_d, x_->val
            );
        }
        #endif

        // ===============================================================
        // stype nr_of_cols = A->n;
        // if (A->full_n != A->m) {
        //     nr_of_cols = A->m / nprocs;
        //     if (myid == nprocs - 1) {
        //         nr_of_cols += A->m % nprocs;
        //     }
        // }

        stype nr_of_cols = x_->n - A->halo.to_receive_d->n;

        // ===============================================================
        // if (nr_of_cols + A->halo.to_receive_d->n != x_->n) {
        //     if (strlen(A->name)) {
        //         fprintf(stderr, "MPI[%d] Before _vector_sync on %s\n", myid, A->name);
        //     } else {
        //         fprintf(stderr, "MPI[%d] Before _vector_sync on %p\n", myid, A);
        //     }
        // }
        // #if 1
        // if (nr_of_cols + A->halo.to_receive_d->n != x_->n)
        // {
        //     fprintf(stderr, "myid = %d, ", myid);
        //     if (A->name) {
        //         fprintf(stderr, "A->name = %s, ", A->name);
        //     } else {
        //         fprintf(stderr, "A = %p, ", A);
        //     }
        //     fprintf(stderr, "A->n = %u, A->full_n = %lu, A->m = %lu, A->row_shift = %lu\n", A->n, A->full_n, A->m, A->row_shift);
        //     fprintf(stderr, "A->halo.to_receive_n = %d, A_->m = %lu, local_x->n = %d, post_local = %d, bitcolsize = %d\n", A->halo.to_receive_n, A_->m, local_x->n, post_local, A->bitcolsize);
        //     fprintf(stderr, "nr_of_cols (%u) + A->halo.to_receive_d->n (%d) = %d, x_->n = %d\n",
        //         nr_of_cols, A->halo.to_receive_d->n, nr_of_cols + A->halo.to_receive_d->n, x_->n);
        //     // debugArray("x[%d] = %lf\n", x_->val, x_->n, x_->on_the_device, stderr);
        //     // debugArray("to_receive_d[%d] = %d\n", A->halo.to_receive_d->val, A->halo.to_receive_d->n, A->halo.to_receive_d->on_the_device, stderr);
        // }
        // #endif
        // ASSERT(nr_of_cols + A->halo.to_receive_d->n == x_->n);
        GridBlock gb = gb1d(A_->m, BLOCKSIZE);
        _vector_sync<<<gb.g, gb.b, 0, *(os.streams->comm_stream)>>>(local_x->val, nr_of_cols, A->halo.what_to_receive_d, A->halo.to_receive_d->n, post_local, x_->val, x_->n);
#else
        CHECK_DEVICE(cudaMemcpy(hi.what_to_receive_d, hi.what_to_receive, hi.to_receive_n * sizeof(vtype), cudaMemcpyHostToDevice));
        GridBlock gb = gb1d(A_->m, BLOCKSIZE);
        _vector_sync<<<gb.g, gb.b>>>(local_x->val, A->n, A->halo.what_to_receive_d, A->halo.to_receive_d->n, post_local, x_->val, x_->n);
#endif
        // complete computation for halo
        if (os.needy_n) {
            CSRm::CSRVector_product_adaptive_indirect_row_miniwarp(A_, x_, w_,
                os.needy_n, os.needy_rows->val, *(os.streams->comm_stream), 0, alpha, beta);
        }
    }

    cudaStreamSynchronize(*(os.streams->local_stream));
    cudaStreamSynchronize(*(os.streams->comm_stream));

    if (A->halo.to_receive_n > 0) {
        FREE(x_);
    }

    A_->col = NULL;
    A_->row = NULL;
    A_->val = NULL;
    FREE(A_);

    END_PROF(__FUNCTION__);
    return (w_);
}

vector<vtype>* CSRm::CSRscaleA_0(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha, vtype beta)
{
    _MPI_ENV;

    if (nprocs == 1) {
        vector<vtype>* w_ = NULL;
        if (w == NULL) {
            w_ = Vector::init<vtype>(A->n, true, true);
            Vector::fillWithValue(w_, 0.);
        } else {
            w_ = w;
        }
        if (A->shrinked_flag == 1) {
            A->col = A->shrinked_col;
        }
        CSRm::CSRscale_adaptive_miniwarp(A, local_x, w_, alpha, beta);
        return (w_);
    }

    assert(A->shrinked_flag == 1);

    CSR* A_ = CSRm::init(A->n, (gstype)A->shrinked_m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);
    A_->row = A->row;
    A_->val = A->val;
    A_->col = A->shrinked_col;

    // ----------------------------------------- temp check -----------------------------------------
    assert(A->halo.to_receive_n + local_x->n == A_->m);
    // ----------------------------------------------------------------------------------------------
    int post_local = A->post_local;

    vector<vtype>* x_ = NULL;
    if (A->halo.to_receive_n > 0) {
        x_ = Vector::init<vtype>(A_->m, false, true);
        if (A_->m > xsize) {
            CUDA_FREE(xvalstat);
            xsize = A_->m;
            xvalstat = CUDA_MALLOC(vtype, xsize, true);
        }
        x_->val = xvalstat;
        
        ASSERT(A->n + A->halo.to_receive_d->n == x_->n);
        GridBlock gb = gb1d(A_->m, BLOCKSIZE);
        _vector_sync<<<gb.g, gb.b>>>(local_x->val, A->n, A->halo.what_to_receive_d, A->halo.to_receive_d->n, post_local, x_->val, x_->n);
    } else {
        x_ = local_x;
    }

    vector<vtype>* w_ = NULL;
    if (w == NULL) {
        w_ = Vector::init<vtype>(A->n, true, true);
        Vector::fillWithValue(w_, 1.);
    } else {
        w_ = w;
    }
    CSRm::CSRscale_adaptive_miniwarp(A_, x_, w_, alpha, beta);

    if (A->halo.to_receive_n > 0) {
        FREE(x_);
    }
    A_->col = NULL;
    A_->row = NULL;
    A_->val = NULL;
    FREE(A_);

    return (w_);
}

vector<vtype>* CSRm::CSRscaleA_0IP(CSR* A, vector<vtype>* local_x, vtype alpha, vtype beta)
{
    _MPI_ENV;

    vector<vtype>* w = Vector::init<vtype>(A->n, false, true);
    w->val = A->val;

    if (nprocs == 1) {
        vector<vtype>* w_ = NULL;
        if (w == NULL) {
            w_ = Vector::init<vtype>(A->n, true, true);
            Vector::fillWithValue(w_, 0.);
        } else {
            w_ = w;
        }
        if (A->shrinked_flag == 1) {
            A->col = A->shrinked_col;
        }
        CSRm::CSRscale_adaptive_miniwarp(A, local_x, w_, alpha, beta);
        return (w_);
    }

    assert(A->shrinked_flag == 1);

    CSR* A_ = CSRm::init(A->n, (gstype)A->shrinked_m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);
    A_->row = A->row;
    A_->val = A->val;
    A_->col = A->shrinked_col;

    // ----------------------------------------- temp check -----------------------------------------
    assert(A->halo.to_receive_n + local_x->n == A_->m);
    // ----------------------------------------------------------------------------------------------
    int post_local = A->post_local;

    vector<vtype>* x_ = NULL;
    if (A->halo.to_receive_n > 0) {
        x_ = Vector::init<vtype>(A_->m, false, true);
        if (A_->m > xsize) {
            CUDA_FREE(xvalstat);
            xsize = A_->m;
            xvalstat = CUDA_MALLOC(vtype, xsize, true);
        }
        x_->val = xvalstat;

        ASSERT(A->n + A->halo.to_receive_d->n == x_->n);
        GridBlock gb = gb1d(A_->m, BLOCKSIZE);
        _vector_sync<<<gb.g, gb.b>>>(local_x->val, A->n, A->halo.what_to_receive_d, A->halo.to_receive_d->n, post_local, x_->val, x_->n);
    } else {
        x_ = local_x;
    }

    vector<vtype>* w_ = NULL;
    if (w == NULL) {
        w_ = Vector::init<vtype>(A->n, true, true);
        Vector::fillWithValue(w_, 1.);
    } else {
        w_ = w;
    }
    CSRm::CSRscale_adaptive_miniwarp(A_, x_, w_, alpha, beta);

    if (A->halo.to_receive_n > 0) {
        FREE(x_);
    }
    A_->col = NULL;
    A_->row = NULL;
    A_->val = NULL;
    FREE(A_);

    return (w_);
}

template <int OP_TYPE>
__global__ void _shifted_CSR_vector_mul_mini_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y, itype shift)
{

    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    vtype T_i = 0.;

    for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
        if (OP_TYPE == 0) {
            T_i += (alpha * A_val[j]) * __ldg(&x[A_col[j]]);
        } else if (OP_TYPE == 1) {
            T_i += A_val[j] * __ldg(&x[A_col[j]]);
        } else if (OP_TYPE == 2) {
            T_i += -A_val[j] * __ldg(&x[A_col[j]]);
        }
    }

    for (int k = MINI_WARP_SIZE >> 1; k > 0; k = k >> 1) {
        T_i += __shfl_down_sync(warp_mask, T_i, k);
    }

    if (lane == 0) {
        if (OP_TYPE == 0) {
            y[shift + warp] = T_i + (beta * y[shift + warp]);
        } else if (OP_TYPE == 1) {
            y[shift + warp] = T_i;
        } else if (OP_TYPE == 2) {
            y[shift + warp] = T_i + y[shift + warp];
        }
    }
}

vector<vtype>* CSRm::shifted_CSRVector_product_adaptive_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, itype shift, vtype alpha, vtype beta)
{
    itype n = A->n;

    int density = A->nnz / A->n;

    int min_w_size;

#if 0
    if (density < MINI_WARP_THRESHOLD_2) {
        min_w_size = 2;
    } else if (density < MINI_WARP_THRESHOLD_4) {
        min_w_size = 4;
    } else if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 8;
    } else {
        min_w_size = 16;
    }
#else
    if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 2;
    } else {
    	min_w_size = 4;
    }
#endif


    if (y == NULL) {
        assert(beta == 0.);
        y = Vector::init<vtype>(n, true, true);
    }

    GridBlock gb = gb1d(n, BLOCKSIZE, true, min_w_size);

    if (alpha == 1. && beta == 0.) {
        _shifted_CSR_vector_mul_mini_warp<1><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val, shift);
    } else if (alpha == -1. && beta == 1.) {
        _shifted_CSR_vector_mul_mini_warp<2><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val, shift);
    } else {
        _shifted_CSR_vector_mul_mini_warp<0><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val, shift);
    }
    cudaDeviceSynchronize();
    return y;
}

__global__ void _shifted_CSR_vector_mul_mini_warp2(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y, itype shift)
{

    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    vtype T_i = 0.;

    for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
        T_i += A_val[j] * __ldg(&x[A_col[j] - shift]);
    }

    for (int k = MINI_WARP_SIZE >> 1; k > 0; k = k >> 1) {
        T_i += __shfl_down_sync(warp_mask, T_i, k);
    }

    if (lane == 0) {
        y[warp] = T_i;
    }
}

vector<vtype>* CSRm::shifted_CSRVector_product_adaptive_miniwarp2(CSR* A, vector<vtype>* x, vector<vtype>* y, itype shift, vtype alpha, vtype beta)
{
    itype n = A->n;

    int density = A->nnz / A->n;

    int min_w_size;

#if 0
    if (density < MINI_WARP_THRESHOLD_2) {
        min_w_size = 2;
    } else if (density < MINI_WARP_THRESHOLD_4) {
        min_w_size = 4;
    } else if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 8;
    } else {
        min_w_size = 16;
    }
#else
    if (density < MINI_WARP_THRESHOLD_8) {
        min_w_size = 2;
    } else {
    	min_w_size = 4;
    }
#endif


    if (y == NULL) {
        assert(beta == 0.);
        y = Vector::init<vtype>(n, true, true);
    }

    GridBlock gb = gb1d(n, BLOCKSIZE, true, min_w_size);

    if (alpha == 1. && beta == 0.) {
        _shifted_CSR_vector_mul_mini_warp2<<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val, shift);
    }

    return y;
}

vector<vtype>* CSRVector_product_MPI(CSR* Alocal, vector<vtype>* x, int type)
{
    assert(Alocal->on_the_device);
    assert(x->on_the_device);

    if (type == 0) {

        // everyone gets all
        vector<vtype>* out = Vector::init<vtype>(x->n, true, true);
        Vector::fillWithValue(out, 0.);

        CSRm::shifted_CSRVector_product_adaptive_miniwarp(Alocal, x, out, Alocal->row_shift);

        vector<vtype>* h_out = Vector::copyToHost(out);
        vector<vtype>* h_full_out = Vector::init<vtype>(x->n, true, false);

        CHECK_MPI(MPI_Allreduce(
            h_out->val,
            h_full_out->val,
            h_full_out->n * sizeof(vtype),
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD));

        Vector::free(out);
        Vector::free(h_out);

        return h_full_out;

    } else if (type == 1) {

        // local vector outputs
        vector<vtype>* out = Vector::init<vtype>(Alocal->n, true, true);
        CSRm::shifted_CSRVector_product_adaptive_miniwarp(Alocal, x, out, 0);
        return out;

    } else {
        assert(false);
        return NULL;
    }
}

__global__ void _getDiagonal_warp(itype n, int MINI_WARP_SIZE, vtype* A_val, itype* A_col, itype* A_row, vtype* D)
{

    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    itype j_start = A_row[warp];
    itype j_stop = A_row[warp + 1];

    int j_d = WARP_SIZE, j;

    for (j = j_start + lane;; j += MINI_WARP_SIZE) {
        int is_diag = __ballot_sync(warp_mask, ((j < j_stop) && (A_col[j] == warp)));
        j_d = __clz(is_diag);
        if (j_d != MINI_WARP_SIZE) {
            break;
        }
    }
}

// SUPER temp kernel
__global__ void _getDiagonal(itype n, vtype* val, itype* col, itype* row, vtype* D, itype row_shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    itype r = i;
    itype j_start = row[i];
    itype j_stop = row[i + 1];

    int j;
    for (j = j_start; j < j_stop; j++) {
        itype c = col[j];

        // if is a diagonal element
        if (c == (r /* + row_shift */)) {
            D[i] = val[j];
            break;
        }
    }
}

// get a copy of the diagonal
vector<vtype>* CSRm::diag(CSR* A)
{
    vector<vtype>* D = Vector::init<vtype>(A->n, true, true);
    GridBlock gb = gb1d(D->n, BLOCKSIZE);
    _getDiagonal<<<gb.g, gb.b>>>(D->n, A->val, A->col, A->row, D->val, A->row_shift);
    return D;
}

__global__ void _row_sum_2(itype n, vtype* A_val, itype* A_row, itype* A_col, vtype* sum)
{

    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    vtype local_sum = 0.;

    int j;
    for (j = A_row[i]; j < A_row[i + 1]; j++) {
        local_sum += fabs(A_val[j]);
    }

    sum[i] = local_sum;
}

vector<vtype>* CSRm::absoluteRowSum(CSR* A, vector<vtype>* sum)
{
    _MPI_ENV;

    assert(A->on_the_device);

    if (sum == NULL) {
        sum = Vector::init<vtype>(A->n, true, true);
    } else {
        assert(sum->on_the_device);
    }

    GridBlock gb = gb1d(A->n, BLOCKSIZE, false);
    _row_sum_2<<<gb.g, gb.b>>>(A->n, A->val, A->row, A->col, sum->val);

    return sum;
}

__global__ void _CSR_vector_mul_prolongator(itype n, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y)
{

    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= n) {
        return;
    }

    itype j = A_row[tid];
    y[tid] += A_val[j] * __ldg(&x[A_col[j]]);
}

vector<vtype>* CSRm::CSRVector_product_prolungator(CSR* A, vector<vtype>* x, vector<vtype>* y)
{
    itype n = A->n;

    assert(A->on_the_device);
    assert(x->on_the_device);

    GridBlock gb = gb1d(n, BLOCKSIZE);

    _CSR_vector_mul_prolongator<<<gb.g, gb.b>>>(n, A->val, A->row, A->col, x->val, y->val);

    return y;
}

// checks if the colmuns are in the correct order
void CSRm::checkColumnsOrder(CSR* A_)
{

    CSR* A;
    if (A_->on_the_device) {
        A = CSRm::copyToHost(A_);
    } else {
        A = A_;
    }

    for (int i = 0; i < A->n; i++) {
        itype _c = -1;
        for (int j = A->row[i]; j < A->row[i + 1]; j++) {
            itype c = A->col[j];

            if (c < _c) {
                DIE("WRONG ORDER COLUMNS: %d %d-%d\n", i, c, _c);
            }
            if (c > _c) {
                _c = c;
            }
            if (c > A->m - 1) {
                DIE("WRONG COLUMN TO BIG: %d %d-%d\n", i, c, _c);
            }
        }
    }
    if (A_->on_the_device) {
        CSRm::free(A);
    }
}

#define MY_EPSILON 0.0001
void CSRm::checkMatrix(CSR* A_, bool check_diagonal)
{
    _MPI_ENV;
    CSR* A = NULL;

    if (A_->on_the_device) {
        A = CSRm::copyToHost(A_);
    } else {
        A = A_;
    }

    for (int i = 0; i < A->n; i++) {
        for (int j = A->row[i]; j < A->row[i + 1]; j++) {
            int c = A->col[j];
            double v = A->val[j];
            int found = 0;
            for (int jj = A->row[c]; jj < A->row[c + 1]; jj++) {
                if (A->col[jj] == i) {
                    found = 1;
                    vtype diff = abs(v - A->val[jj]);
                    if (A->val[jj] != v && diff >= MY_EPSILON) {
                        DIE("\n\nNONSYM %lf %lf %lf\n\n", v, A->val[jj], diff);
                    }
                    break;
                }
            }
            if (!found) {
                DIE("BAD[%d]: %d %d\n", myid, i, c);
            }
        }
    }

    checkColumnsOrder(A);

    if (check_diagonal) {
        printf("CHECKING DIAGONAL\n");
        for (int i = 0; i < A->n; i++) {
            bool found = false;
            for (int j = A->row[i]; j < A->row[i + 1]; j++) {
                int c = A->col[j];
                vtype v = A->val[j];
                if (c == i && v > 0.) {
                    found = true;
                }
            }
            if (!found) {
                DIE("MISSING ELEMENT DIAG %d\n", i);
            }
        }
        if (A_->on_the_device) {
            CSRm::free(A);
        }
    }
}

/**
 * CUDA kernel.
 * Scans a matrix in CSR format and collects non zero items in ret.
 * Should be invoked using 1 (mini)warp per row.
 *
 * @param row CSR matrix row indexes
 * @param col CSR matrix column indexes
 * @param val CSR matrix (non zero) values
 * @param nrows number of rows
 * @param ret returned array
 */
__global__ void _combineRowAndCol(itype* row, itype* col, vtype* val,
    gstype row_shift, gsstype col_shift, stype nrows, int warpSize,
    matrixItem_t* ret)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int irow = tid / warpSize;
    int lane = tid % warpSize;
    int rstart, rend;
    if (irow < nrows) {
        rstart = row[irow] + lane;
        rend = row[irow + 1];
        for (int we = rstart; we < rend; we += warpSize) {
            ret[we].row = irow + row_shift;
            ret[we].col = col[we] - col_shift;
            ret[we].val = val[we];
        }
    }
}

matrixItem_t* CSRm::collectMatrixItems(CSR* dlA, FILE* debug, bool useColShift)
{
    // Allocate vector to collect non-zero items in dlA
    // ---------------------------------------------------------------------------
    matrixItem_t* d_nnzItems = CUDA_MALLOC(matrixItem_t, dlA->nnz, true);

    // Collect items
    // ---------------------------------------------------------------------------
    int warpSize = CSRm::choose_mini_warp_size(dlA);
    GridBlock gb = getKernelParams(dlA->n * warpSize); // One mini-warp per row
    _combineRowAndCol<<<gb.g, gb.b>>>(dlA->row, dlA->col, dlA->val, dlA->row_shift, useColShift ? dlA->col_shifted : 0, dlA->n, warpSize, d_nnzItems);
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_DEVICE(err);

    #if DEBUG
    if (debug) {
        debugMatrixItems("nnzItems", d_nnzItems, dlA->nnz, true, debug);
    }
    #endif

    return d_nnzItems;
}

matrixItem_t* CSRm::collectMatrixItems_nogpu(CSR* dlA, FILE* debug, bool useColShift)
{
    // Allocate vector to collect non-zero items in dlA
    // ---------------------------------------------------------------------------
    matrixItem_t* d_nnzItems = MALLOC(matrixItem_t, dlA->nnz, true);

    // Collect items
    // ---------------------------------------------------------------------------
    itype col_shift = useColShift ? dlA->col_shifted : 0;
    for (itype irow = 0; irow < dlA->n; irow++) {
        for (int we = dlA->row[irow]; we < dlA->row[irow + 1]; we += 1) {
            d_nnzItems[we].row = irow + dlA->row_shift;
            d_nnzItems[we].col = dlA->col[we] - col_shift;
            d_nnzItems[we].val = dlA->val[we];
        }
    }

    #if DEBUG
    if (debug) {
        debugMatrixItems("nnzItems", d_nnzItems, dlA->nnz, false, debug);
    }
    #endif

    return d_nnzItems;
}

// #undef DEBUG
// #define DEBUG 1

// Fix Giacomo 2026-04-15: added optional out_distr parameter.
// When out_distr is provided, its (row_shift, n) determine the output row distribution
// and its per-process row counts are gathered via MPI_Allgather for correct item routing.
// This is needed when the SUITOR matching produces a non-uniform coarse distribution:
// the original uniform estimate (full_n/nprocs * myid) can be off-by-one, causing items
// to be routed to the wrong process and leading to OOB accesses or solver divergence.
/**
 * @brief Transposes a CSR matrix.
 * @param A Pointer to the CSR matrix.
 * @param f File pointer for logging.
 * @param out_distr When provided, its fields (row_shift, n) determine the output row distribution.
 * @return Pointer to the transposed CSR matrix.
 */
 CSR* CSRm::transpose(CSR* dlA, FILE* f, CSR* out_distr)
 {
     assert(dlA->on_the_device);

     _MPI_ENV;

     // Compute transposed matrix properties
     // ---------------------------------------------------------------------------
     gstype full_n = dlA->m;
     gstype m = dlA->full_n;
     gstype row_shift;
     stype n;
     if (out_distr != NULL) {
         // Fix Giacomo 2026-04-15: use the actual distribution of the output matrix
         // (e.g. the coarse matrix built by SUITOR matching) instead of the uniform
         // estimate. The same distribution is also passed to ProcessSelector below so
         // that each matrix item is routed to the process that actually owns its output
         // row, even when the coarse distribution is non-uniform.
         row_shift = out_distr->row_shift;
         n = out_distr->n;
     } else {
         // Fall back to uniform distribution estimate
         n = full_n / nprocs;
         row_shift = n * (taskmap ? itaskmap[myid] : myid);
         if (myid == nprocs - 1) {
             n += full_n % nprocs;
         }
     }
     
     #if DEBUG
     if (f) {
        fprintf(f, "Inside %s\n", __func__);
        fprintf(f, "[Process %d] A->n (rows) : %d\n" , myid, dlA->n);
        fprintf(f, "[Process %d] A->full_n (rows) : %d\n" , myid, dlA->full_n);
        fprintf(f, "[Process %d] A->m (cols) : %lu\n", myid, dlA->m);
        fprintf(f, "[Process %d] A->nnz      : %d\n" , myid, dlA->nnz);
        fprintf(f, "[Process %d] A->row shift: %lu\n", myid, dlA->row_shift);

        fprintf(f, "[Process %d] T->n (rows) : %d\n" , myid, n);
        fprintf(f, "[Process %d] T->full_n (rows) : %d\n" , myid, full_n);
        fprintf(f, "[Process %d] T->m (cols) : %lu\n", myid, m);
        fprintf(f, "[Process %d] T->row shift: %lu\n", myid, row_shift);
    }
    #endif

     // Register custom MPI datatypes
     // ---------------------------------------------------------------------------
     registerMatrixItemMpiDatatypes();
 
     // Collect non-zero items in dlA
     // ---------------------------------------------------------------------------
     matrixItem_t* d_nnzItems = collectMatrixItems(dlA, f, true);
 
     // Identify the items to be sent: they are the ones whose column
     // index is before the first row index assigned to the process or
     // after the last row assigned to the process.
     // ---------------------------------------------------------------------------
 
     size_t nnzItemsToBeSentSize = 0;
     matrixItem_t* d_nnzItemsToBeSent = devicePartition(
         d_nnzItems,
         dlA->nnz,
         MatrixItemColumnIndexOutOfBoundsSelector(
             row_shift,
             row_shift + n - 1),
         &nnzItemsToBeSentSize);
 
     #if DEBUG
     if (f) {
         fprintf(f, "nnzItemsToBeSent effective size: %zu\n", nnzItemsToBeSentSize);
         debugMatrixItems("nnzItemsToBeSent", d_nnzItemsToBeSent, nnzItemsToBeSentSize, true, f);
     }
     #endif
 
     // Identify the items not to be requested: they are the ones whose column
     // index is between the first row index and the last row index
     // assigned to the process.
     // ---------------------------------------------------------------------------
     matrixItem_t* d_nnzItemsNotToBeSent = d_nnzItemsToBeSent + nnzItemsToBeSentSize;
     size_t nnzItemsNotToBeSentSize = dlA->nnz - nnzItemsToBeSentSize;
 
     #if DEBUG
     if (f) {
         fprintf(f, "nnzItemsNotToBeSent effective size: %zu\n", nnzItemsNotToBeSentSize);
         debugMatrixItems("nnzItemsNotToBeSent", d_nnzItemsNotToBeSent, nnzItemsNotToBeSentSize, true, f);
     }
     #endif
 
     // Release memory
     // ---------------------------------------------------------------------------
     CUDA_FREE(d_nnzItems);
 
     // Copy data to host in order to perform MPI communication
     // ---------------------------------------------------------------------------
     matrixItem_t* h_nnzItemsToBeSent = copyArrayToHost(d_nnzItemsToBeSent, nnzItemsToBeSentSize);
 
     // Exchange data with other processes
     // ---------------------------------------------------------------------------
     // Fix Giacomo 2026-04-15: when out_distr is provided, build ProcessSelector from it
     // (using MPI_Allgather of its actual per-process row counts) instead of from full_n
     // (which assumes a uniform distribution). This ensures each item is sent to the
     // process that truly owns its output row under the SUITOR coarse distribution.
     ProcessSelector processSelector = out_distr ? ProcessSelector(out_distr, f)
                                                 : ProcessSelector(full_n, f);
     //  ProcessSelector processSelector(full_n, f);
     MatrixItemSender itemSender(&processSelector, f);
     MpiBuffer<matrixItem_t> sendBuffer;
     MpiBuffer<matrixItem_t> rcvBuffer;
     itemSender.send(h_nnzItemsToBeSent, nnzItemsToBeSentSize,
         &sendBuffer, &rcvBuffer);
 
     // Now we have all the initially missing values in rcv_buffer and all the
     // initially interesting values in nnzItemsNotToBeSent. We need
     // to construct a new (transposed) matrix from all those values.
     // ---------------------------------------------------------------------------
     size_t concatenatedSize = rcvBuffer.size + nnzItemsNotToBeSentSize;
 
     matrixItem_t* d_concatenated = concatArrays<matrixItem_t>(
         rcvBuffer.buffer, // arr1
         rcvBuffer.size, // len1
         false, // onDevice1
         d_nnzItemsToBeSent + nnzItemsToBeSentSize, // arr2
         nnzItemsNotToBeSentSize, // len2
         true, // onDevice2
         true // retOnDevice
     );
 
     #if DEBUG
     if (f) {
         fprintf(f, "concatenatedItems effective size: %zu\n", concatenatedSize);
         debugMatrixItems("concatenatedItems", d_concatenated, concatenatedSize, true, f);
     }
     #endif
 
     // Release memory
     // ---------------------------------------------------------------------------
     CUDA_FREE(d_nnzItemsToBeSent);
     FREE(h_nnzItemsToBeSent);
 
     // Sort items by col, row (pratically: already transposed)
     // ---------------------------------------------------------------------------
     deviceSort<matrixItem_t, gstype, MatrixItemTransposedComparator>(d_concatenated, concatenatedSize, MatrixItemTransposedComparator(dlA->full_n));
 
     #if DEBUG
     if (f) {
         debugMatrixItems("sortedItems", d_concatenated, concatenatedSize, true, f);
     }
 
     if (!concatenatedSize) {
         fprintf(f ? f : stderr, "concatenatedSize in process %d is 0. Row shift: %ld, n: %d\n", myid, dlA->row_shift, dlA->n);
     }
     #endif
 
     // Create new CSR matrix
     // ---------------------------------------------------------------------------
    CSR* d_transposed = CSRm::init(
        n, // Nr of rows,
        m, // Nr of columns,
        concatenatedSize, // nnz
        true, // Allocate memory
        true, // On the device
        false, // Is symmetric?
        full_n,
        row_shift);
 
     // Fill CSR
     // ---------------------------------------------------------------------------
     fillCsrFromMatrixItems(
         d_concatenated,
         concatenatedSize,
         d_transposed->n,
         d_transposed->row_shift,
         &(d_transposed->row),
         &(d_transposed->col),
         &(d_transposed->col8),
         &(d_transposed->val),
         true, // Transposed
         false // Allocate memory
     );
 
     #if DEBUG
     if (f) {
         debugArray("d_transposed->row[%d] = %d\n", d_transposed->row, d_transposed->n + 1, true, f);
         debugArray("d_transposed->col[%d] = %d\n", d_transposed->col, d_transposed->nnz, true, f);
         debugArray("d_transposed->col8[%d] = %ld\n", d_transposed->col8, d_transposed->nnz, true, f);
         debugArray("d_transposed->val[%d] = %lf\n", d_transposed->val, d_transposed->nnz, true, f);
     }
     #endif
 
     CUDA_FREE(d_concatenated);
 
     if (d_transposed->row_shift) {
         CSRm::shift_cols(d_transposed, -d_transposed->row_shift);
         d_transposed->col_shifted = -d_transposed->row_shift;
     }
 
     return d_transposed;
 }

//  #undef DEBUG
//  #define DEBUG 0

/**
 * @param dlA device local A
 * @param f process-specific log file
 */
CSR* CSRm::Transpose_local(CSR* dlA, FILE* f)
{
    assert(dlA->on_the_device);

    _MPI_ENV;

    #if DEBUG
    if (f) {
        fprintf(f, "Inside %s\n", __func__);
        fprintf(f, "n (rows) : %d\n", dlA->n);
        fprintf(f, "m (cols) : %lu\n", dlA->m);
        fprintf(f, "nnz      : %d\n", dlA->nnz);
        fprintf(f, "row shift: %lu\n", dlA->row_shift);
    }
    #endif

    // Register custom MPI datatypes
    // ---------------------------------------------------------------------------
    registerMatrixItemMpiDatatypes();

    // Collect non-zero items in dlA
    // ---------------------------------------------------------------------------
    matrixItem_t* d_nnzItems = collectMatrixItems(dlA, f, true);

    // Sort items by col, row (pratically: already transposed)
    // ---------------------------------------------------------------------------
    deviceSort<matrixItem_t, gstype, MatrixItemTransposedComparator>(d_nnzItems, dlA->nnz, MatrixItemTransposedComparator(dlA->n));

    #if DEBUG
    if (f) {
        debugMatrixItems("sortedItems", d_nnzItems, dlA->nnz, true, f);
    }
    #endif

    // Create new CSR matrix
    // ---------------------------------------------------------------------------
    CSR* d_transposed = CSRm::init(
        dlA->m, // Nr of rows,
        dlA->full_n, // Nr of columns,
        dlA->nnz, // nnz
        true, // Allocate memory
        true, // On the device
        false, // Is symmetric?
        dlA->m,
        0); // row shift

    // Fill CSR
    // ---------------------------------------------------------------------------
    fillCsrFromMatrixItems(
        d_nnzItems,
        dlA->nnz,
        d_transposed->n,
        d_transposed->row_shift,
        &(d_transposed->row),
        &(d_transposed->col),
        &(d_transposed->col8),
        &(d_transposed->val),
        true, // Transposed
        false // Allocate memory
    );

    #if DEBUG
    if (f) {
        debugArray("d_transposed->row[%d] = %d\n", d_transposed->row, d_transposed->n + 1, true, f);
        debugArray("d_transposed->col[%d] = %d\n", d_transposed->col, d_transposed->nnz, true, f);
        debugArray("d_transposed->col8[%d] = %ld\n", d_transposed->col8, d_transposed->nnz, true, f);
        debugArray("d_transposed->val[%d] = %lf\n", d_transposed->val, d_transposed->nnz, true, f);
    }
    #endif

    // Release memory
    // ---------------------------------------------------------------------------
    CUDA_FREE(d_nnzItems);

    if (dlA->row_shift) {
        CSRm::shift_cols(d_transposed, -dlA->row_shift);
    }

    return d_transposed;
}

bool CSRm::checkSizeAndShift(CSR *A, CSR *AH) {
    return A->n == AH->n
        && A->m == AH->m
        && A->full_n == AH->full_n
        && A->row_shift == AH->row_shift
        && A->col_shifted == AH->col_shifted;
}

/**
 * CUDA kernel.
 * Scans a matrix in CSR format, counts non zero items in each requested row,
 * and returns the result in ret.
 * Should be invoked using 1 thread per requested row.
 *
 * @param row CSR matrix row indexes
 * @param col CSR matrix column indexes
 * @param val CSR matrix (non zero) values
 * @param nrows number of rows
 * @param ret returned array
 */
__global__ void CSRm::countNnzPerRow(itype* row,
    itype row_shift,
    gsstype* requestedRowIndexes,
    itype requestedRowIndexesSize,
    itype* ret)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < requestedRowIndexesSize) {
        itype irow = requestedRowIndexes[tid] - row_shift;
        itype rstart = row[irow];
        itype rend = row[irow + 1];
        ret[tid] = rend - rstart;
    }
}

/**
 * CUDA kernel.
 * Scans a matrix in CSR format and collects non zero items in ret.
 * Should be invoked using 1 (mini)warp per requested row index.
 *
 * @param row CSR matrix row indexes
 * @param col CSR matrix column indexes
 * @param val CSR matrix (non zero) values
 * @param row_shift Distributed CSR matrix row shift
 * @param n Number of requested rows
 * @param requestedRowIndexes Requested row indexes
 * @param counter Number of nnz per requested row index
 * @param offset Offset of each requested row with respect to the return buffer
 * @param ret returned buffer
 */
__global__ void CSRm::collectNnzPerRow(
    int warpSize,
    itype* row,
    itype* col,
    vtype* val,
    itype row_shift,
    itype n,
    gsstype* requestedRowIndexes,
    itype* counter,
    itype* offset,
    matrixItem_t* ret)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int index = tid / warpSize;
    int lane = tid % warpSize;
    if (index < n) {
        int off = offset[index];
        int irow = requestedRowIndexes[index] - row_shift;
        int rstart = row[irow];
        int rend = row[irow + 1];
        for (int we = rstart + lane; we < rend; we += warpSize) {
            int ibuf = off + we - rstart;
            ret[ibuf].row = irow + row_shift;
            ret[ibuf].col = col[we];
            ret[ibuf].val = val[we];
        }
    }
}

void check_and_fix_order(CSR* A)
{

    itype* Arow = A->row;
    gsstype* Acol8 = A->col8;
    itype* Acol = A->col;        // Giacomo: 2026-04-13
    vtype* Aval = A->val;
    itype prev;
    int wrongo;
    for (int i = 0; i < A->n; i++) {
        wrongo = 0;
        prev = A->col8[Arow[i]];
        for (int j = Arow[i] + 1; j < Arow[i + 1]; j++) {
            if (A->col8[j] < prev) {
                wrongo = 1;
                break;
            } else {
                prev = A->col8[j];
            }
        }
        if (wrongo) {
            bubbleSort(&Acol8[Arow[i]], &Aval[Arow[i]], (Arow[i + 1] - Arow[i]));
            // Giacomo: 2026-04-13 (begin)
            // sincronizza col da col8 ordinata
            for (int j = Arow[i]; j < Arow[i + 1]; j++) {
                Acol[j] = (itype)Acol8[j];
            }
            // Giacomo: 2026-04-13 (end)
        }
    }
}

void swap(gsstype* xcol, gsstype* ycol, vtype* xval, vtype* yval)
{
    gsstype temp = *xcol;
    vtype tempf = *xval;
    *xcol = *ycol;
    *xval = *yval;
    *ycol = temp;
    *yval = tempf;
}

void bubbleSort(gsstype arr[], vtype val[], itype n)
{
    itype i, j;
    for (i = 0; i < n - 1; i++) {

        // Last i elements are already in place
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1], &val[j], &val[j + 1]);
            }
        }
    }
}

CSR* read_matrix_from_file(const char* matrix_path, int m_type, bool loadOnDevice)
{
    CSR* A_host = NULL;

    switch (m_type) {
    case 0:
        A_host = readMTXDouble(matrix_path);
        break;
    case 1:
        A_host = readMTX2Double(matrix_path);
        break;
    default:
        DIE("You need to specify an input matrix type with the argument -F/--inputype\n");
    }

    assert(A_host != NULL);

    if (loadOnDevice) {
        CSR* A = CSRm::copyToDevice(A_host);
        CSRm::free(A_host);
        if (!A->col8) {
            A->col8 = CUDA_MALLOC(gsstype, A->nnz);
        }
        ASSERT(A->col8);
        col2col8(A->col, A->col8, A->nnz);
        return A;
    }

    return A_host;
}

// stolen from BootCMatch CPU
CSR* readMTXDouble(const char* file_name)
{
    FILE* fp;
    char banner[64], mtx[64], crd[64], data_type[64], storage_scheme[64];
    char buffer[BUFSIZE + 1];
    double *matrix_value, *matrix_data, val;
    unsigned long int *matrix_cooi, *matrix_i;
    unsigned long int *matrix_cooj, *matrix_j;
    unsigned long int num_rows, num_cols, ri, cj;
    unsigned long int fr_nonzeros, allc_nonzeros;
    unsigned long int num_nonzeros;
    unsigned long int max_col = 0, is_general = 0, is_symmetric = 0;
    unsigned long int row_shift = 0;
    unsigned long int full_n = 0;
    long int col_shifted = 0;

    int file_base = 1;

    long int i, j, k, k0, iad;
    double x;

    /*----------------------------------------------------------
     * Read in the data (matrix in MM format)
     *----------------------------------------------------------*/

    fp = fopen(file_name, "r");
    if (fp == NULL) {
        DIE("Error opening file %s, errno = %d: %s\n", file_name, errno, strerror(errno));
    }

    (void)fscanf(fp, "%s %s %s %s %s\n", banner, mtx, crd, data_type, storage_scheme);
    (void)fgets(buffer, BUFSIZE, fp);
    for (; buffer[0] == '%'; (void)fgets(buffer, BUFSIZE, fp))
        ;

    int readParams = sscanf(buffer, "%lu %lu %lu "
                                    "%lu %lu %ld",
        &num_rows, &num_cols, &fr_nonzeros,
        &row_shift, &full_n, &col_shifted);

    ASSERT(readParams == 3 || readParams == 6);
    if (readParams == 3) {
        row_shift = 0;
        full_n = num_rows;
        col_shifted = 0;
    }

    #if DEBUG
    if (log_file) {
        fprintf(log_file,
            "num_rows = %lu, num_cols = %lu, fr_nonzeros = %lu\n"
            "row_shift = %lu, full_n = %lu, col_shifted = %ld\n",
            num_rows, num_cols, fr_nonzeros,
            row_shift, full_n, col_shifted
        );
    }
    #endif

    if (strcmp(data_type, "real") != 0) {
        fprintf(stderr, "Error: we only read real matrices, not '%s'\n", data_type);
        fclose(fp);
        return (NULL);
    }

    if (strcmp(storage_scheme, "general") == 0) {
        allc_nonzeros = fr_nonzeros;
        is_general = 1;
    } else if (strcmp(storage_scheme, "symmetric") == 0) {
        allc_nonzeros = 2 * fr_nonzeros;
        is_symmetric = 1;
    } else {
        fprintf(stderr, "Error: unhandled storage scheme '%s'\n", storage_scheme);
        fclose(fp);
        return (NULL);
    }

    matrix_cooi = MALLOC(unsigned long int, allc_nonzeros, true);
    matrix_cooj = MALLOC(unsigned long int, allc_nonzeros, true);
    matrix_value = MALLOC(double, allc_nonzeros, true);
    if (is_general) {
        num_nonzeros = fr_nonzeros;
        for (j = 0; j < fr_nonzeros; j++) {
            if (fgets(buffer, BUFSIZE, fp) != NULL) {
                sscanf(buffer, "%lu %lu %le", &matrix_cooi[j], &matrix_cooj[j], &matrix_value[j]);
                // printf("Read %lu %lu %le\n", matrix_cooi[j], matrix_cooj[j], matrix_value[j]);
                matrix_cooi[j] -= file_base + row_shift;
                matrix_cooj[j] -= file_base;
                if (matrix_cooj[j] > max_col) {
                    max_col = matrix_cooj[j];
                }
            } else {
                DIE("Error while trying to read record %ld of %lu from MatrixMarket file %s\n",
                    j, fr_nonzeros, file_name);
            }
        }
    } else if (is_symmetric) {
        k = 0;
        for (j = 0; j < fr_nonzeros; j++) {
            if (fgets(buffer, BUFSIZE, fp) != NULL) {
                sscanf(buffer, "%lu %lu %le", &ri, &cj, &val);
                ri -= file_base;
                cj -= file_base;
                if (cj > max_col) {
                    max_col = cj;
                }
                matrix_cooi[k] = ri;
                matrix_cooj[k] = cj;
                matrix_value[k] = val;
                k++;
                if (ri != cj) {
                    matrix_cooi[k] = cj;
                    matrix_cooj[k] = ri;
                    matrix_value[k] = val;
                    k++;
                }
            } else {
                fprintf(stderr, "Reading from MatrixMarket file failed\n");
                fprintf(stderr, "Error while trying to read record %ld of %lu from file %s\n",
                    j, fr_nonzeros, file_name);
                fclose(fp);
                return (NULL);
            }
        }
        num_nonzeros = k;
    } else {
        fprintf(stderr, "Internal error: neither symmetric nor general ? \n");
        fclose(fp);
        return (NULL);
    }
    /*----------------------------------------------------------
     * Transform matrix from COO to CSR format
     *----------------------------------------------------------*/

    matrix_i = MALLOC(unsigned long, num_rows + 1, true);

    /* determine row lenght */
    for (j = 0; j < num_nonzeros; j++) {
        if (matrix_cooi[j] < num_rows) {
            matrix_i[matrix_cooi[j]] = matrix_i[matrix_cooi[j]] + 1;
        } else {
            fprintf(stderr, "Wrong row index %lu at position %ld\n", matrix_cooi[j], j);
        }
    }

    /* starting position of each row */
    k = 0;
    for (j = 0; j <= num_rows; j++) {
        k0 = matrix_i[j];
        matrix_i[j] = k;
        k = k + k0;
    }
    matrix_j = MALLOC(unsigned long int, num_nonzeros, true);
    matrix_data = MALLOC(double, num_nonzeros, true);

    /* go through the structure once more. Fill in output matrix */
    for (k = 0; k < num_nonzeros; k++) {
        i = matrix_cooi[k];
        j = matrix_cooj[k];
        x = matrix_value[k];
        iad = matrix_i[i];
        matrix_data[iad] = x;
        matrix_j[iad] = j;
        matrix_i[i] = iad + 1;
    }
    /* shift back matrix_i */
    for (j = num_rows - 1; j >= 0; j--) {
        matrix_i[j + 1] = matrix_i[j];
    }
    matrix_i[0] = 0;

    // ASSERT(num_rows > 0 && num_cols > 0 && num_nonzeros >= 0);
    CSR* A = CSRm::init(num_rows, num_cols, num_nonzeros, true, false, false, full_n, row_shift);
    FREE(A->val);
    A->val = matrix_data;

    for (j = 0; j <= num_rows; j++) {
        A->row[j] = matrix_i[j];
    }
    for (k = 0; k < num_nonzeros; k++) {
        A->col[k] = matrix_j[k];
        A->col8[k] = (gsstype)matrix_j[k]; // GIACOMO 2026-04-13
    }

    FREE(matrix_cooi);
    FREE(matrix_cooj);
    FREE(matrix_value);
    fclose(fp);

    return A;
}

CSR* readMTX2Double(const char* file_name)
{

    FILE* fp;

    double *matrix_value, *matrix_data;
    int *matrix_cooi, *matrix_i;
    int *matrix_cooj, *matrix_j;
    int num_rows;
    int num_nonzeros;
    int max_col = 0;

    int file_base = 1;

    int i, j, k, k0, iad;
    double x;

    /*----------------------------------------------------------
     * Read in the data (matrix in COO format)
     *----------------------------------------------------------*/

    fp = fopen(file_name, "r");
    if (fp == NULL) {
        DIE("Error opening file %s, errno = %d: %s, FILE NOT FOUND!\n", file_name, errno, strerror(errno));
    }

    (void)fscanf(fp, "%d", &num_rows);
    (void)fscanf(fp, "%d", &num_nonzeros);

    matrix_cooi = MALLOC(int, num_nonzeros, true);
    for (j = 0; j < num_nonzeros; j++) {
        (void)fscanf(fp, "%d", &matrix_cooi[j]);
        matrix_cooi[j] -= file_base;
    }
    matrix_cooj = MALLOC(int, num_nonzeros, true);
    for (j = 0; j < num_nonzeros; j++) {
        (void)fscanf(fp, "%d", &matrix_cooj[j]);
        matrix_cooj[j] -= file_base;
        if (matrix_cooj[j] > max_col) {
            max_col = matrix_cooj[j];
        }
    }
    matrix_value = MALLOC(double, num_nonzeros, true);
    for (j = 0; j < num_nonzeros; j++) {
        (void)fscanf(fp, "%le", &matrix_value[j]);
    }

    /*----------------------------------------------------------
     * Transform matrix from COO to CSR format
     *----------------------------------------------------------*/

    matrix_i = MALLOC(int, num_rows + 1, true);

    /* determine row lenght */
    for (j = 0; j < num_nonzeros; j++) {
        matrix_i[matrix_cooi[j]] = matrix_i[matrix_cooi[j]] + 1;
    }

    /* starting position of each row */
    k = 0;
    for (j = 0; j <= num_rows; j++) {
        k0 = matrix_i[j];
        matrix_i[j] = k;
        k = k + k0;
    }
    matrix_j = MALLOC(int, num_nonzeros, true);
    matrix_data = MALLOC(double, num_nonzeros, true);

    /* go through the structure once more. Fill in output matrix */
    for (k = 0; k < num_nonzeros; k++) {
        i = matrix_cooi[k];
        j = matrix_cooj[k];
        x = matrix_value[k];
        iad = matrix_i[i];
        matrix_data[iad] = x;
        matrix_j[iad] = j;
        matrix_i[i] = iad + 1;
    }
    /* shift back matrix_i */
    for (j = num_rows - 1; j >= 0; j--) {
        matrix_i[j + 1] = matrix_i[j];
    }
    matrix_i[0] = 0;

    ASSERT(num_rows > 0 && num_rows > 0 && num_nonzeros >= 0);
    CSR* A = CSRm::init(num_rows, num_rows, num_nonzeros, false, false, false, num_rows);
    A->val = matrix_data;
    A->row = matrix_i;
    A->col = matrix_j;

    FREE(matrix_cooi);
    FREE(matrix_cooj);
    FREE(matrix_value);
    fclose(fp);

    return A;
}

void CSRMatrixPrintMM(CSR* A_, const char* file_name)
{
    CSR* A = NULL;
    if (A_->on_the_device) {
        A = CSRm::copyToHost(A_);
    } else {
        A = A_;
    }

    FILE* fp;

    double* matrix_data;
    int* matrix_i;
    int* matrix_j;
    int num_rows;
    int num_cols, nnz;

    int file_base = 1;

    int i, j;

    matrix_data = A->val;
    matrix_i = A->row;
    matrix_j = A->col;
    num_rows = A->n;
    num_cols = A->m;
    nnz = A->nnz;

    fp = fopen(file_name, "w");
    fprintf(fp, "%s\n", "%%MatrixMarket matrix coordinate real general");

    fprintf(fp, "%d  %d %d \n", num_rows, num_cols, nnz);

    for (i = 0; i < num_rows; i++) {
        for (j = matrix_i[i]; j < matrix_i[i + 1]; j++) {
            fprintf(fp, "%d   %d  %lg\n", i + file_base, matrix_j[j] + file_base, matrix_data[j]);
        }
    }
    fclose(fp);
}

void CSRm::printInfo(CSR* A, FILE* fp)
{
    _MPI_ENV;

    fprintf(fp, "nnz                   : %d\n", A->nnz);
    fprintf(fp, "n                     : %d\n", A->n);
    fprintf(fp, "m                     : %lu\n", A->m);
    fprintf(fp, "shrinked_m            : %d\n", A->shrinked_m);
    fprintf(fp, "full_n                : %lu\n", A->full_n);
    fprintf(fp, "full_m                : %lu\n", A->full_m);
    fprintf(fp, "on_the_device         : %d\n", A->on_the_device);
    fprintf(fp, "is_symmetric          : %d\n", A->is_symmetric);
    fprintf(fp, "shrinked_flag         : %d\n", A->shrinked_flag);
    fprintf(fp, "custom_alloced        : %d\n", A->custom_alloced);
    fprintf(fp, "col_shifted           : %ld\n", A->col_shifted);
    // fprintf(fp, "shrinked_firstrow     : %lu\n", A->shrinked_firstrow);
    // fprintf(fp, "shrinked_lastrow      : %lu\n", A->shrinked_lastrow);
    fprintf(fp, "row_shift             : %lu\n", A->row_shift);
    fprintf(fp, "bitcolsize            : %d\n", A->bitcolsize);
    fprintf(fp, "post_local            : %d\n", A->post_local);

    // fprintf(fp, "val                   : 0x%X\n", A->val);
    // fprintf(fp, "col                   : 0x%X\n", A->col);
    // fprintf(fp, "row                   : 0x%X\n", A->row);
    // fprintf(fp, "shrinked_col          : 0x%X\n", A->shrinked_col);
    // fprintf(fp, "bitcol                : 0x%X\n", A->bitcol);

    fprintf(fp, "halo.init             : %d\n", A->halo.init);
    fprintf(fp, "halo.to_receive_n     : %d\n", A->halo.to_receive_n);
    if (A->halo.to_receive) {
        debugArray("halo.to_receive[%d]: %d\n", A->halo.to_receive->val, A->halo.to_receive->n, A->halo.to_receive->on_the_device, fp);
    }
    if (A->halo.to_receive_d) {
        debugArray("halo.to_receive_d[%d]: %d\n", A->halo.to_receive_d->val, A->halo.to_receive_d->n, A->halo.to_receive_d->on_the_device, fp);
    }
    if (A->halo.to_receive_counts) {
        debugArray("halo.to_receive_counts[%d]: %d\n", A->halo.to_receive_counts, nprocs, false, fp);
    }
    if (A->halo.to_receive_spls) {
        debugArray("halo.to_receive_spls[%d]: %d\n", A->halo.to_receive_spls, nprocs, false, fp);
    }
    if (A->halo.what_to_receive) {
        debugArray("halo.what_to_receive[%d]: %d\n", A->halo.what_to_receive, A->halo.to_receive_n, false, fp);
    }
    if (A->halo.what_to_receive_d) {
        debugArray("halo.what_to_receive_d[%d]: %d\n", A->halo.what_to_receive_d, A->halo.to_receive_n, true, fp);
    }

    fprintf(fp, "halo.to_send_n        : %d\n", A->halo.to_send_n);
    if (A->halo.to_send) {
        debugArray("halo.to_send[%d]: %d\n", A->halo.to_send->val, A->halo.to_send->n, A->halo.to_send->on_the_device, fp);
    }
    if (A->halo.to_send_d) {
        debugArray("halo.to_send_d[%d]: %d\n", A->halo.to_send_d->val, A->halo.to_send_d->n, A->halo.to_send_d->on_the_device, fp);
    }
    if (A->halo.to_send_counts) {
        debugArray("halo.to_send_counts[%d]: %d\n", A->halo.to_send_counts, nprocs, false, fp);
    }
    if (A->halo.to_send_spls) {
        debugArray("halo.to_send_spls[%d]: %d\n", A->halo.to_send_spls, nprocs, false, fp);
    }
    if (A->halo.what_to_send) {
        debugArray("halo.what_to_send[%d]: %d\n", A->halo.what_to_send, A->halo.to_send_n, false, fp);
    }
    if (A->halo.what_to_send_d) {
        debugArray("halo.what_to_send_d[%d]: %d\n", A->halo.what_to_send_d, A->halo.to_send_n, true, fp);
    }

    fprintf(fp, "os.loc_n              : %d\n", A->os.loc_n);
    if (A->os.loc_rows) {
        debugArray("os.loc_rows[%d]: %d\n", A->os.loc_rows->val, A->os.loc_rows->n, A->os.loc_rows->on_the_device, fp);
    }

    fprintf(fp, "os.needy_n            : %d\n", A->os.needy_n);
    if (A->os.needy_rows) {
        debugArray("os.needy_rows[%d]: %d\n", A->os.needy_rows->val, A->os.needy_rows->n, A->os.needy_rows->on_the_device, fp);
    }

    fprintf(fp, "\n");
}

void CSRm::fillWithValues(CSR* A, std::initializer_list<vtype> list)
{
    _MPI_ENV;

    ASSERT(!A->on_the_device);
    ASSERT(list.size() == A->full_n * A->m);

    int rowsPerProcess = A->full_n / nprocs;
    int rowShift = rowsPerProcess * (taskmap ? taskmap[myid] : myid);

    if (A->row_shift) {
        ASSERT(rowShift == A->row_shift);
    } else {
        A->row_shift = rowShift;
    }

    int firstIndex = rowShift * A->m;
    int lastIndex = firstIndex + (A->n * A->m) - 1;

    int nnz = 0;
    int col = 0;
    int row = 0;

    A->row[0] = 0;

    for (int index = firstIndex; index <= lastIndex; index++) {
        vtype val = *(list.begin() + index);
        if (val != 0) {
            A->val[nnz] = val;
            A->col[nnz] = col - A->row_shift;
            A->col8[nnz] = col - A->row_shift;
            nnz++;
        }
        col++;
        if (col == A->m) {
            row++;
            A->row[row] = nnz;
            col = 0;
        }
    }
    
    if (!A->nnz) {
        A->nnz = nnz;
    } else {
        ASSERT(A->nnz == nnz);
    }

    A->col_shifted = -A->row_shift;
}

// void CSRm::debug(CSR *A, FILE *out) {
//     _MPI_ENV;

//     cudaDeviceSynchronize();
//     MPI_Barrier(MPI_COMM_WORLD);
//     for (int i = 0; i < nprocs; i++) {
//         if (myid == i) {
//             CSRm::print(A, 3, -1, out, false, false);
//         }
//         MPI_Barrier(MPI_COMM_WORLD);
//     }
// }

void CSRm::debug(CSR* A, FILE* out)
{
    _MPI_ENV;

    cudaDeviceSynchronize();

    CSR* hA = A->on_the_device
        ? CSRm::copyToHost(A)
        : A;

    if (nprocs > 1) {
        CSR* A_glob = join_matrix_mpi(hA);
        if (ISMASTER) {
            CSRm::print(A_glob, 3, -1, out, false, false);
            CSRm::free(A_glob);
        }
    } else {
        CSRm::print(hA, 3, -1, out, false, false);
    }

    if (A->on_the_device) {
        CSRm::free(hA);
    }
}

CSR* CSRm::createTestMatrix(int col, std::initializer_list<vtype> values)
{
    ASSERT(values.size() % col == 0);

    _MPI_ENV;

    int full_n = values.size() / col;
    int n = full_n / nprocs;
    int row_shift = n * (taskmap ? taskmap[myid] : myid);
    if (myid == nprocs - 1) {
        n += full_n % nprocs;
    }
    int max_nnz = n * col;

    CSR* hA = CSRm::init(
        n, // Rows
        col, // Cols
        max_nnz, /* nnz */
        true, /* allocate_mem */
        false, /* on_the_device */
        false, /* symmetric */
        full_n, /* full_n */
        row_shift); /* row_shift */

    hA->nnz = 0; // This will force fillWithValues() to set the correct number
    CSRm::fillWithValues(hA, values);

    CSR* A = CSRm::copyToDevice(hA);
    CSRm::free(hA);
    return A;
}

bool CSRm::eq(CSR* A, CSR* B)
{
    // Compare size
    // ---------------------------------------------------
    if (A->n != B->n) {
        TRACE("A->n != B->n");
        return false;
    }

    if (A->m != B->m) {
        TRACE("A->m != B->m");
        return false;
    }

    if (A->nnz != B->nnz) {
        TRACE("A->nnz != B->nnz");
        return false;
    }

    if (A->full_n != B->full_n) {
        TRACE("A->full_n != B->full_n");
        return false;
    }

    // Compare row
    // ---------------------------------------------------

    if (A->on_the_device && B->on_the_device) {
        if (!deviceMemEq(A->row, B->row, (A->n + 1) * sizeof(itype))) {
            TRACE("A->row != B->row");
            return false;
        }
    } else if (!A->on_the_device && !B->on_the_device) {
        if (memcmp(A->row, B->row, (A->n + 1) * sizeof(itype))) {
            TRACE("A->row != B->row");
            return false;
        }
    } else {
        itype* Arow = A->on_the_device
            ? copyArrayToHost(A->row, A->n + 1)
            : A->row;

        itype* Brow = B->on_the_device
            ? copyArrayToHost(B->row, B->n + 1)
            : B->row;

        if (memcmp(Arow, Brow, (A->n + 1) * sizeof(itype))) {
            TRACE("A->row != B->row");
            return false;
        }

        if (A->on_the_device) {
            FREE(Arow);
        }

        if (B->on_the_device) {
            FREE(Brow);
        }
    }

    // Compare col
    // ---------------------------------------------------

    {
        itype* Acol = A->on_the_device
            ? copyArrayToHost(A->col, A->nnz)
            : A->col;

        itype* Bcol = B->on_the_device
            ? copyArrayToHost(B->col, B->nnz)
            : B->col;

        for (int i = 0; i < A->nnz; i++) {
            if (Acol[i] - A->col_shifted != Bcol[i] - B->col_shifted) {
                TRACE("A->col != B->col");
                return false;
            }
        }

        if (A->on_the_device) {
            FREE(Acol);
        }

        if (B->on_the_device) {
            FREE(Bcol);
        }
    }

    // Compare val
    // ---------------------------------------------------

    if (A->on_the_device && B->on_the_device) {
        if (!deviceMemEq(A->val, B->val, (A->nnz) * sizeof(vtype))) {
            TRACE("A->val != B->val");
            return false;
        }
    } else if (!A->on_the_device && !B->on_the_device) {
        if (memcmp(A->val, B->val, (A->nnz) * sizeof(vtype))) {
            TRACE("A->val != B->val");
            return false;
        }
    } else {
        vtype* Aval = A->on_the_device
            ? copyArrayToHost(A->val, A->nnz)
            : A->val;

        vtype* Bval = B->on_the_device
            ? copyArrayToHost(B->val, B->nnz)
            : B->val;

        bool differ = false;
        for (size_t i = 0; i < A->nnz; i++) {
            if (!vtypeEq(Aval[i], Bval[i])) {
                differ = true;
                break;
            }
        }
        if (differ) {
            TRACE("A->val != B->val;");
            for (int i = 0; i < A->nnz; i++) {
                if (Aval[i] != Bval[i]) {
                    TRACE("\nA->val[%d] = %lf\nB->val[%d] = %lf\n", i, Aval[i], i, Bval[i]);
                }
            }
            return false;
        }

        if (A->on_the_device) {
            FREE(Aval);
        }

        if (B->on_the_device) {
            FREE(Bval);
        }
    }

    return true;
}

CSR* CSRm::product(handles *h, CSR *dA, CSR *dP) {
    // int row_shift = 0;
    // // int col_shifted = 0;
    // if (dA->row_shift) {
    //     ASSERT(dA->col_shifted == -(long)dA->row_shift);
    //     // printf("A->row_shift %d, A->col_shifted %d\n", dA->row_shift, dA->col_shifted);
    //     // printf("P->row_shift %d, P->col_shifted %d\n", dP->row_shift, dP->col_shifted);
    //     row_shift = dA->row_shift;
    //     // col_shifted = dA->col_shifted;
    //     CSRm::shift_cols(dA, row_shift - dP->row_shift);
    //     dA->row_shift = dP->row_shift;
    //     dA->col_shifted = -dP->row_shift;
    // }

    TRACE("first_row = %lu, last_row = %lu", dP->row_shift, dP->row_shift + dP->n - 1);

    vector<gsstype>* _bitcol = get_missing_col(dA, dP);
    #if DEBUG
    if (log_file) {
        debugArray("Columns to request: bitcol[%d] = %d\n", _bitcol->val, _bitcol->n, false, log_file, [dA](const gsstype &c){ return c + dA->row_shift; });
    }
    #endif

    TRACE("Before compute_rows_to_rcv_CPU\n");
    compute_rows_to_rcv_CPU(dA, dP, _bitcol);
    TRACE("After compute_rows_to_rcv_CPU\n");

    Vector::free(_bitcol);

    if (dP->col_shifted) {
        CSRm::shift_cols(dP, -dP->col_shifted);
    }

    TRACE("Before nsparseMGPU_commu_new\n");
    CSR *dAP = nsparseMGPU_commu_new(h, dA, dP);
    ASSERT(dAP->col8);
    col2col8(dAP->col, dAP->col8, dAP->nnz);
    TRACE("After nsparseMGPU_commu_new\n");

    if (dP->col_shifted) {
        CSRm::shift_cols(dP, dP->col_shifted);
    }

    // if (row_shift) {
    //     CSRm::shift_cols(dA, dP->row_shift - row_shift);
    //     dA->row_shift = row_shift;
    //     dA->col_shifted = -row_shift;
    // }

    ASSERT(dAP->on_the_device);
    return dAP;
}

CSR *CSRm::local_product(handles *h, CSR *dA, CSR *dP) {
    csrlocinfo locinfo;
    locinfo.fr = 0;
    locinfo.lr = dP->n;
    locinfo.row = dP->row;
#if !defined(CSRSEG)
    locinfo.col = NULL;
    // locinfo.col8 = NULL;
#else
    locinfo.col = dP->col;
    locinfo.col8 = dP->col8;
#endif
    locinfo.val = dP->val;

    CSR *dAP = nsparseMGPU(dA, dP, &locinfo);
    return dAP;
}

bool CSRm::checkProduct(handles *h, CSR *dA, CSR *dP, CSR *dAP) {
    _MPI_ENV;

    ASSERT(dA->on_the_device);
    ASSERT(dP->on_the_device);
    ASSERT(dAP->on_the_device);

    CSR* hA = CSRm::copyToHost(dA);
    CSR* hP = CSRm::copyToHost(dP);

    // -------------------------------------------------------------------------
    // Expected result
    // -------------------------------------------------------------------------

    TRACE("join hA");
    CSR* hA_glob = nprocs > 1 ? join_matrix_mpi(hA) : hA;

    TRACE("join hP");
    CSR* hP_glob = nprocs > 1 ? join_matrix_mpi(hP) : hP;

    CSR* dExpectedAP_glob = NULL;

    if (ISMASTER) {
        // char filename[1024] = {0};
        // snprintf(filename, 1024, "%s/%s%s%s.mtx", output_dir.c_str(), output_prefix.c_str(), "A", output_suffix.c_str());
        // CSRm::printMM(hA_glob, filename);
        // snprintf(filename, 1024, "%s/%s%s%s.mtx", output_dir.c_str(), output_prefix.c_str(), "P", output_suffix.c_str());
        // CSRm::printMM(hP_glob, filename);

        TRACE("copyToDevice hA_glob");
        CSR* dA_glob = CSRm::copyToDevice(hA_glob);

        TRACE("copyToDevice hP_glob");
        CSR* dP_glob = CSRm::copyToDevice(hP_glob);

        // CSRm::forEach(dA_glob, []__device__(CSR *A, int irow, int innz){
        //     if (irow == 2) {
        //         int icol = A->col[innz];
        //         double val = A->val[innz];
        //         printf("A[%d,%d]=%lf\n", irow, icol, val);
        //     }
        // });

        // CSRm::forEach(dP_glob, []__device__(CSR *P, int irow, int innz){
        //     int icol = P->col[innz];
        //     if (icol == 2) {
        //         double val = P->val[innz];
        //         printf("P[%d,%d]=%lf\n", irow, icol, val);
        //     }
        // });

        

        TRACE("Computing expected AP");
        dExpectedAP_glob = CSRm::local_product(h, dA_glob, dP_glob);
        ASSERT(dExpectedAP_glob->on_the_device);

        TRACE("Free dA_glob, dP_glob");
        CSRm::free(dA_glob);
        CSRm::free(dP_glob);
    }

    CSRm::free(hA);
    CSRm::free(hP);

    // -------------------------------------------------------------------------
    // Compare results
    // -------------------------------------------------------------------------

    CSR* hAP = CSRm::copyToHost(dAP);
    CSR* hAP_glob = nprocs > 1 ? join_matrix_mpi(hAP) : hAP;
    if (hAP_glob != hAP) {
        CSRm::free(hAP);
    }

    bool ret = false;
    if (ISMASTER) {
        // trace_enabled = true;
        if (!CSRm::eq(hAP_glob, dExpectedAP_glob)) {
#if 0
            fprintf(stderr, "Computed result:\n");
            CSRm::print(hAP_glob, 3, -1, stderr);

            fprintf(stderr, "Expected result:\n");
            CSRm::print(dExpectedAP_glob, 3, -1, stderr);
#endif

            fprintf(stderr, "Ops, it doesn't seem to work properly.\n");
            char filename[1024] = {0};

            snprintf(filename, 1024, "%s/%s%s%s.mtx", output_dir.c_str(), output_prefix.c_str(), "A", output_suffix.c_str());
            CSRm::printMM(hA_glob, filename);

            snprintf(filename, 1024, "%s/%s%s%s.mtx", output_dir.c_str(), output_prefix.c_str(), "P", output_suffix.c_str());
            CSRm::printMM(hP_glob, filename);

            snprintf(filename, 1024, "%s/%s%s%s.mtx", output_dir.c_str(), output_prefix.c_str(), "expected", output_suffix.c_str());
            CSRm::printMM(dExpectedAP_glob, filename);

            snprintf(filename, 1024, "%s/%s%s%s.mtx", output_dir.c_str(), output_prefix.c_str(), "result", output_suffix.c_str());
            CSRm::printMM(hAP_glob, filename);

            fprintf(stderr, "Unexpected product. Please, chech .mtx files in %s\n", output_dir.c_str());
            ret = false;
        } else {
            fprintf(stderr, "Wow, it seems to work properly.\n");
            ret = true;
        }
        // trace_enabled = false;

        CSRm::free(hAP_glob);
        CSRm::free(dExpectedAP_glob);

        if (hA != hA_glob) {
            CSRm::free(hA_glob);
        }
        if (hP != hP_glob) {
            CSRm::free(hP_glob);
        }
    }

    CHECK_MPI(MPI_Bcast(
        &ret,
        1, MPI_C_BOOL, 0, MPI_COMM_WORLD
    ));

    return ret;
}

bool CSRm::checkUniqueIndeces(CSR *dA) {
    _MPI_ENV;
    CSR *hA = CSRm::copyToHost(dA);
    bool ret = true;
    for (long i = 0; ret && i < hA->n; i++) {
        std::unordered_set<long> s;
        for (int k = hA->row[i]; ret && k < hA->row[i + 1]; k++) {
            if (!s.insert(hA->col[k]).second) {
                printf("MPI[%d] Duplicate value at row %ld col %d\n", myid, i, hA->col[k]);
                ret = false;
            }
        }
    }
    CSRm::free(hA);
    return ret;
}

vector<long> *CSRm::getUniqueColumns(CSR *dA) {
    ASSERT(dA->on_the_device);
    long *col8 = cloneDeviceArray(dA->col8, dA->nnz);
    deviceSort<long, long, NumberComparator<long>>(col8, dA->nnz, NumberComparator<long>());
    size_t usize = deviceUniqueIP(col8, dA->nnz);
    vector<long> *ret = Vector::init<long>(usize, false, true);
    ret->val = col8;
    return ret;
}

void CSRm::printUniqueColumns(CSR *dA, const char *matrixName, FILE *out) {
    vector<long> *ucol8 = CSRm::getUniqueColumns(dA);
    char fmt[1024] = {};
    snprintf(fmt, 1024, "%s->col8[%%d]=%%ld\n", matrixName);
    debugArray(fmt, ucol8->val, ucol8->n, ucol8->on_the_device, out);
    Vector::free(ucol8);
}

bool CSRm::checkNonNullColumns(CSR *dA) {
    ASSERT(dA->on_the_device);
    _MPI_ENV;
    vector<long> *ucol8 = CSRm::getUniqueColumns(dA);
    bool ret = ucol8->n == dA->m;
    if (!ret) {
        fprintf(stderr, "myid = %d, %s unique cols %d, m = %d\n", myid, dA->name ? dA->name : "", ucol8->n, dA->m);
    }
    Vector::free(ucol8);
    return ret;
}

void CSRm::printGlobalMM(CSR *A, FILE *out) {
    _MPI_ENV;

    cudaDeviceSynchronize();

    CSR* hA = A->on_the_device
        ? CSRm::copyToHost(A)
        : A;

    if (nprocs > 1) {
        CSR* A_glob = join_matrix_mpi(hA);
        if (ISMASTER) {
            CSRm::printMM(A_glob, out);
            CSRm::free(A_glob);
        }
    } else {
        CSRm::printMM(hA, out);
    }

    if (A->on_the_device) {
        CSRm::free(hA);
    }
}

void CSRm::printGlobalMM(CSR* A, char* name)
{
    FILE* fp = fopen(name, "w");
    printGlobalMM(A, fp);
    fclose(fp);
}