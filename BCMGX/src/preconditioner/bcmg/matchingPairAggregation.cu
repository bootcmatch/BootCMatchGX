#include "datastruct/scalar.h"
#include "matchingAggregation.h"
#include "preconditioner/bcmg/matching.h"
#include "utility/memory.h"
#include "utility/profiling.h"
#include "utility/col8.h"

extern int* taskmap;
extern int* itaskmap;

// Functor type for selecting values less than some criteria
struct Matched {
    int compare;
    __host__ __device__ __forceinline__
    Matched(int compare)
        : compare(compare)
    {
    }
    __host__ __device__ __forceinline__ bool operator()(const int& a) const
    {
        return (a != compare);
    }
};

__global__ void _aggregate_symmetric_step_two(stype n, vtype* P_val, itype* M, itype* markc, vtype* w, itype* nuns, itype shift)
{

    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }
    if (markc[i] != -1) {
        return;
    }

    // only single vertex and no-good pairs reach this point
    if (fabs(w[i]) > DBL_EPSILON) { // good single
        int nuns_local = atomicAdd(nuns, 1);
        markc[i] = nuns_local;
        P_val[i] = w[i] / fabs(w[i]);
    } else { // bad single
        markc[i] = (*nuns) - 1;
        P_val[i] = 0.0;
    }
}

__global__ void _aggregate_symmetric_unsorted(stype n, vtype* P_val, itype* M, itype* markc, vtype* w, itype shift)
{

    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    itype v = i;
    itype u = M[i];

    // if it's a matched pair
    if (u != -1) {

        vtype wv = w[v], wu = w[u];
        vtype normwagg = sqrt(wv * wv + wu * wu);

        if (normwagg > DBL_EPSILON) { // good pair
            // if v is the pair master
            if (v < u) {
                markc[v] = v;
                markc[u] = -1;

                P_val[v] = wv / normwagg;
                P_val[u] = wu / normwagg;

                return;
            }
        }
    }
    markc[v] = -1;
}

__global__ void _aggregate_symmetric_sort(itype n, itype* P_col, const itype* __restrict__ M, const itype* __restrict__ comp_Pcol)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    itype v = comp_Pcol[i];
    itype u = M[v];
    P_col[v] = i;
    P_col[u] = i;
}

__global__ void _make_P_row(itype n, itype* P_row)
{

    itype v = blockDim.x * blockIdx.x + threadIdx.x;

    if (v > n) {
        return;
    }

    P_row[v] = v;
}

void* d_temp_storage = NULL;

CSR* makeP_GPU(CSR* A, vector<itype>* M, vector<vtype>* w)
{
    BEGIN_PROF(__FUNCTION__);

    CSR* P = CSRm::init(A->n, 1, A->n, true, true, false, A->full_n, A->row_shift);

    // recycle  allocated memory
    itype* comp_Pcol = P->row;

    Matched m(-1);
    scalar<itype>* d_num_selected_out = Scalar::init<itype>(0, true);

    GridBlock gb = gb1d(A->n, BLOCKSIZE);
    _aggregate_symmetric_unsorted<<<gb.g, gb.b>>>(A->n, P->val, M->val, P->col, w->val, A->row_shift);

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::If(
        NULL,
        temp_storage_bytes,
        P->col,
        comp_Pcol,
        d_num_selected_out->val,
        M->n,
        m);

    // Allocate temporary storage
    if (d_temp_storage == NULL) {
        d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes);
    }

    cub::DeviceSelect::If(
        d_temp_storage,
        temp_storage_bytes,
        P->col,
        comp_Pcol,
        d_num_selected_out->val,
        M->n,
        m);

    itype* num_selected_out = Scalar::getValueFromDevice(d_num_selected_out);
    gb = gb1d(*num_selected_out, BLOCKSIZE);
    _aggregate_symmetric_sort<<<gb.g, gb.b>>>(*num_selected_out, P->col, M->val, comp_Pcol);

    gb = gb1d(A->n, BLOCKSIZE);
    _aggregate_symmetric_step_two<<<gb.g, gb.b>>>(A->n, P->val, M->val, P->col, w->val, d_num_selected_out->val, A->row_shift);

    FREE(num_selected_out);
    num_selected_out = Scalar::getValueFromDevice(d_num_selected_out);
    P->m = (*num_selected_out);

    gb = gb1d(A->n + 1, BLOCKSIZE);
    _make_P_row<<<gb.g, gb.b>>>(A->n, P->row);
    cudaDeviceSynchronize();
    FREE(num_selected_out);
    CUDA_FREE(d_temp_storage);
    Scalar::free(d_num_selected_out);

    col2col8(P->col, P->col8, P->nnz);

    END_PROF(__FUNCTION__);
    return P;
}

void matchingPairAggregation(handles* h, buildData* amg_data, CSR* A, vector<vtype>* w, CSR** _P, CSR** _R)
{
    static int counter = 0;

    BEGIN_PROF(__FUNCTION__);

    _MPI_ENV;

    assert(A);
    assert(A->col8);
    vector<itype>* M = suitor(h, amg_data, A, w);

    CSR *P = makeP_GPU(A, M, w);

    // {
    //     char fname[1024] = {0};
    //     snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), "", "preP", counter, myid, nprocs);
    //     CSRm::printMM(P, fname, false);
    // }

    CSR *R = CSRm::Transpose_local(P, log_file);
    CSRm::shift_cols(R, P->row_shift);

    // get colum numbers other process
    if (nprocs > 1) {
        gstype shifts[nprocs];
        gstype cols[nprocs];

        CHECK_MPI(MPI_Allgather(
            &P->m,
            sizeof(gstype),
            MPI_BYTE,
            cols,
            sizeof(gstype),
            MPI_BYTE,
            MPI_COMM_WORLD
        ));

        gstype tot_cols = 0;

        for (int i = 0; i < nprocs; i++) {
            shifts[taskmap ? taskmap[i] : i] = tot_cols;
            tot_cols += cols[taskmap ? taskmap[i] : i];
        }

        P->m = tot_cols;

        TRACE("Shifting P's cols = %lu", shifts[myid]);
        R->row_shift = shifts[myid];
        CSRm::shift_cols(P, shifts[myid]);
    }
    R->full_n = P->m;

    // {
    //     char fname[1024] = {0};
    //     snprintf(fname, 1024, "%s/%smpa%d_%s_id%d_nprocs%d.mtx", output_dir.c_str(), "", counter, "P", myid, nprocs);
    //     CSRm::printMM(P, fname, false);
    // }
    // {
    //     char fname[1024] = {0};
    //     snprintf(fname, 1024, "%s/%smpa%d_%s_id%d_nprocs%d.mtx", output_dir.c_str(), "", counter, "R", myid, nprocs);
    //     CSRm::printMM(R, fname, false);
    // }

    // CSR *Raux = CSRm::transpose(P, log_file);
    // {
    //     char fname[1024] = {0};
    //     snprintf(fname, 1024, "%s/%smpa%d_%s_id%d_nprocs%d.mtx", output_dir.c_str(), "", counter, "Raux", myid, nprocs);
    //     CSRm::printMM(Raux, fname, false);
    // }
    // CSRm::free(Raux);

    FREE(M);

    // col2col8(R->col, R->col8, R->nnz);

    *_P = P;
    *_R = R;

    counter++;

    END_PROF(__FUNCTION__);
}
