#include "AMG.h"
#include "datastruct/CSR.h"
#include "matching.h"
#include "suitor.h"
#include "utility/cudamacro.h"
#include "utility/deviceAbsMin.h"
#include "utility/handles.h"
#include "utility/memory.h"
#include "utility/profiling.h"
#include <float.h>

// ------------ AH glob -----------
extern itype* AH_glob_row;
extern itype* AH_glob_col;
extern gsstype* AH_glob_col8;
extern vtype* AH_glob_val;

__forceinline__
    __device__ int
    binsearch(int array[], unsigned int size, int value)
{
    unsigned int low, high, medium;
    low = 0;
    high = size;
    while (low < high) {
        medium = (high + low) / 2;
        if (value > array[medium]) {
            low = medium + 1;
        } else {
            high = medium;
        }
    }
    return low;
}

__global__ void _write_T_warp(itype n, int MINI_WARP_SIZE, vtype* A_val, itype* A_col, itype* A_row, itype shift)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);
    vtype t;

    itype j_stop = A_row[warp + 1];

    for (int j = A_row[warp] + lane; j < j_stop; j += MINI_WARP_SIZE) {
        itype c = A_col[j];

        if (c < 0 || c >= n) {
            continue;
        }

        if (warp < c) {
            break;
        }

        int nc = A_row[c + 1] - A_row[c];

        int jj = binsearch(A_col + A_row[c], nc, warp);

        t = A_val[jj + A_row[c]];
        A_val[j] = t;
    }
}

__global__ void _makeAH_warp(itype n, int AH_MINI_WARP_SIZE, vtype* A_val, itype* A_col, gsstype* A_col8, itype* A_row, vtype* w, vtype* C, vtype* AH_val, itype* AH_col, gsstype* AH_col8, itype* AH_row, itype row_shift)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    itype warp = tid / AH_MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    int lane = tid % AH_MINI_WARP_SIZE;
    itype j_stop = A_row[warp + 1];

    for (int j = A_row[warp] + lane; j < j_stop; j += AH_MINI_WARP_SIZE) {
        gsstype c = A_col8[j];

        if (c < 0 || c >= n) {
            itype offset = c > (warp) ? warp + 1 : warp;
            AH_val[j - offset] = DBL_MAX;
            AH_col[j - offset] = (c < 0) ? (labs(c) > INT_MAX ? -INT_MAX : c) : (c > INT_MAX ? INT_MAX : c);
            AH_col8[j - offset] = c; // Giacomo 2025-04-30
        } else {

            if (c != warp) {
                vtype a = A_val[j];
                itype offset = c > (warp) ? warp + 1 : warp;
                AH_col[j - offset] = c;
                AH_col8[j - offset] = c; // Giacomo 2025-04-30
                vtype norm = c > (warp) ? C[warp] + C[c] : C[c] + C[warp];
                if (norm > DBL_EPSILON) {
                    vtype w_temp = c > (warp) ? w[warp] * w[c] : w[c] * w[warp];
                    AH_val[j - offset] = 1. - ((2. * a * w_temp) / norm);
                } else {
                    AH_val[j - offset] = DBL_EPSILON;
                }
            }
        }
    }

    if (lane == 0) {
        AH_row[warp + 1] = j_stop - (warp + 1);
    }

    if (tid == 0) {
        // set the first index of the row pointer to 0
        AH_row[0] = 0;
    }
}

__global__ void _makeC(stype n, vtype* val, gsstype* col8, itype* row, vtype* w, vtype* C, itype row_shift)
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
        gsstype c = col8[j];

        // if is a diagonal element
        if (c == r) {
            C[r] = val[j] * w[r] * w[r] /* pow(w[r], 2) */;
            break;
        }
    }
}

CSR* makeAH(buildData* amg_data, CSR* A, vector<vtype>* w)
{
    _MPI_ENV;
    static int cnt = 0;
    assert(A->on_the_device);
    assert(A->val);
    assert(A->col8);
    assert(A->row);
    assert(w->on_the_device);
    assert(w->val);

    // CSRm::print(A, 3, -1, stderr);
    // Vector::print(w, -1, stderr);

    itype n;
    n = A->n;

    // init a vector on the device
    // --------------------------------------------------------
    vector<vtype>* C = Vector::init<vtype>(A->n, false, true);
    C->val = amg_data->ws_buffer->val;

    // vector<vtype>* C = Vector::init<vtype>(A->n, true, true);
    // assert(C->val);
    // --------------------------------------------------------

    GridBlock gb = gb1d(n, BLOCKSIZE, false);
    // only local access to w, c must be local but with shift
    _makeC<<<gb.g, gb.b>>>(n, A->val, A->col8, A->row, w->val, C->val, A->row_shift);
    // CHECK_DEVICE(cudaDeviceSynchronize());

    // Diagonal MUST be non-empty!
    // -------------------------------------------------------------------------------------------------------
    CSR* AH = CSRm::init(A->n, A->m, (A->nnz - A->n), false, true, A->is_symmetric, A->full_n, A->row_shift);
    AH->val = AH_glob_val;
    AH->col = AH_glob_col;
    AH->col8 = AH_glob_col8;
    AH->row = AH_glob_row;
    // -------------------------------------------------------------------------------------------------------

    int miniwarp_size = CSRm::choose_mini_warp_size(A);
    gb = gb1d(n, BLOCKSIZE, true, miniwarp_size);
    _makeAH_warp<<<gb.g, gb.b>>>(n, miniwarp_size, A->val, A->col, A->col8, A->row, w->val, C->val, AH->val, AH->col, AH->col8, AH->row, AH->row_shift);
    // CHECK_DEVICE(cudaDeviceSynchronize());

    FREE(C);
    cnt++;

    AH->col_shifted = A->col_shifted; // Giacomo 2025-05-09
    assert(CSRm::checkSizeAndShift(A, AH)); // TODO: remove this

    // CSRm::print(AH, 3, -1, stderr);
    return AH;
}

__global__ void _make_w(stype nnz, vtype* val, vtype min)
{

    stype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nnz) {
        return;
    }
    vtype scratch = fabs(val[i]);
    val[i] = log(scratch ? scratch : SUITOR_EPS / (0.999 * (min)));
}

CSR* toMaximumProductMatrix(CSR* AH)
{
    _MPI_ENV;
    assert(AH->on_the_device);
    assert(AH->val);

    // CSRm::print(AH, 3, -1, stderr);

    stype nnz = AH->nnz;
    // find the min value
    // printf("nnz: %d\n", nnz);

    // vtype* min = find_Max_Min(AH->val, nnz, 1);
    // vtype h_local_min;
    // CHECK_DEVICE(cudaMemcpy(&h_local_min, min, sizeof(vtype), cudaMemcpyDeviceToHost));
    
    vtype h_local_min = deviceAbsMin(AH->val, nnz);

    if ((fabs(h_local_min)) < DBL_EPSILON) {
        h_local_min = DBL_EPSILON;
    }

    GridBlock gb = gb1d(nnz, BLOCKSIZE, false);
    _make_w<<<gb.g, gb.b>>>(nnz, AH->val, h_local_min);

    return AH;
}

vector<itype>* suitor(handles* h, buildData* amg_data, CSR* A, vector<vtype>* w)
{
    _MPI_ENV;
    assert(A->on_the_device && w->on_the_device);

    // BEGIN_DETAILED_TIMING(PREC_SETUP, MAKEAHW)
    CSR* AH = makeAH(amg_data, A, w);
    CSR* W = toMaximumProductMatrix(AH);
    assert(W == AH);
    // END_DETAILED_TIMING(PREC_SETUP, MAKEAHW)

    // BEGIN_DETAILED_TIMING(PREC_SETUP, SUITOR)
    itype n = W->n;

    vector<vtype>* ws_buffer = Vector::init<vtype>(n, false, true);
    ws_buffer->val = amg_data->ws_buffer->val;

    vector<itype>* mutex_buffer = Vector::init<itype>(n, false, true);
    mutex_buffer->val = amg_data->mutex_buffer->val;

    vector<itype>* _M = Vector::init<itype>(n, false, true);
    _M->val = amg_data->_M->val;

    int warp_size = CSRm::choose_mini_warp_size(W);
    GridBlock gb = gb1d(n, BLOCKSIZE, true, warp_size);
    _write_T_warp<<<gb.g, gb.b>>>(n, warp_size, W->val, W->col, W->row, W->row_shift);

    approx_match_gpu_suitor(h, A, W, _M, ws_buffer, mutex_buffer);

    // END_DETAILED_TIMING(PREC_SETUP, SUITOR)

    FREE(ws_buffer);
    FREE(mutex_buffer);
    FREE(W);

    return _M;
}
