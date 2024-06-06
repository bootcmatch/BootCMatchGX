#include "spspmpi.h"

#include "basic_kernel/halo_communication/local_permutation.h"
#include "datastruct/CSR.h"
#include "utility/cudamacro.h"
#include "utility/function_cnt.h"
#include "utility/setting.h"

#include <unistd.h>

#define BITXBYTE 8
// #define CSRSEG

#ifdef NSP2_NSPARSE
#include "nsp2.h"
#else
#include "nsparse.h"
#endif

extern int *itaskmap, *taskmap;

__global__ void _getNNZ(itype n, const itype* __restrict__ to_get_form_prow, const itype* __restrict__ row, itype* nnz_to_get_form_prow)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    itype j = to_get_form_prow[i];
    nnz_to_get_form_prow[i] = row[j + 1] - row[j];
}

__forceinline__
    __device__ int
    binsearch(const itype array[], itype size, itype value)
{
    itype low, high, medium;
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

__global__ void _fillPRow(itype n, itype rows2bereceived, const itype* __restrict__ whichproc, itype* p_nnz_map,
    itype mypfirstrow, itype myplastrow, itype nzz_pre_local, itype Plocalnnz, const itype* __restrict__ local_row, itype* row)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
#if !defined(CSRSEG)
    if (i >= mypfirstrow && i <= myplastrow + 1) {
        row[i] = local_row[i - mypfirstrow] + nzz_pre_local;
        return;
    }
#else
    if (i > mypfirstrow && i <= myplastrow) {
        return;
    }
    if (i == mypfirstrow && i == (myplastrow + 1)) {
        row[i] = local_row[i - mypfirstrow] + nzz_pre_local;
        return;
    }
#endif

    itype iv = binsearch(whichproc, rows2bereceived, i);

    itype shift = Plocalnnz * (i > myplastrow);
    if (iv > 0) {
        row[i] = p_nnz_map[iv - 1] * (iv > 0) + shift;
    } else {
        row[i] = shift;
    }
}

__global__ void _fillPRowNoComm(itype n, itype mypfirstrow, itype myplastrow, itype Plocalnnz, const itype* __restrict__ local_row, itype* row)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (i >= mypfirstrow && i <= (myplastrow + 1)) {
        row[i] = local_row[i - mypfirstrow];
        return;
    }
    row[i] = Plocalnnz * (i > myplastrow);
}

__global__ void _getColMissingMap(itype nnz, itype mypfirstrow, itype myplastrow, itype* col, int* mask)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= nnz) {
        return;
    }
    itype c = col[tid];

    int mask_idx = c / (sizeof(itype) * BITXBYTE);
    unsigned int m = 1 << ((c % (sizeof(itype) * BITXBYTE)));

    if (c < mypfirstrow || c > myplastrow) {
        atomicOr(&mask[mask_idx], m);
    }
}

__global__ void _getColVal(itype n, itype* rcvprow, itype* nnz_per_row, const itype* __restrict__ row,
    const itype* __restrict__ col, const vtype* __restrict__ val, itype* col2get, vtype* val2get, itype row_shift)
{
    itype q = blockDim.x * blockIdx.x + threadIdx.x;
    if (q >= n) {
        return;
    }
    itype I = rcvprow[q] - row_shift;
    itype start = row[I];
    itype end = row[I + 1];
    for (itype i = start, j = nnz_per_row[q]; i < end; i++, j++) {
        col2get[j] = col[i];
        val2get[j] = val[i];
    }
}

itype merge(itype a[], itype b[], itype c[], itype n1, itype n2)
{
    itype i, j, k;

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = 0; // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (a[i] < b[j]) {
            c[k] = a[i];
            i++;
        } else if (a[i] > b[j]) {
            c[k] = b[j];
            j++;
        } else {
            c[k] = b[j];
            i++;
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there are any */
    while (i < n1) {
        c[k] = a[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there are any */
    while (j < n2) {
        c[k] = b[j];
        j++;
        k++;
    }
    return k;
}

int bswhichprocess(gsstype* P_n_per_process, int nprocs, gsstype e)
{
    unsigned int low, high, medium;
    low = 0;
    high = nprocs;
    while (low < high) {
        medium = (high + low) / 2;
        if (e > P_n_per_process[medium]) {
            low = medium + 1;
        } else {
            high = medium;
        }
    }
    return low;
}

int dumpP = 0;
extern char idstring[];

CSR* nsparseMGPU(CSR* Alocal, CSR* Pfull, csrlocinfo* Plocal, bool used_by_solver)
{
    _MPI_ENV;

    sfCSR mat_a, mat_p, mat_c;
    assert(Alocal->on_the_device && Pfull->on_the_device);

    mat_a.M = Alocal->n;
    mat_a.N = Alocal->m;
    mat_a.nnz = Alocal->nnz;

    mat_a.d_rpt = Alocal->row;
    mat_a.d_col = Alocal->col;
    mat_a.d_val = Alocal->val;

    mat_p.M = Pfull->n;
    mat_p.N = Pfull->m;
    mat_p.nnz = Pfull->nnz;

    mat_p.d_rpt = Pfull->row;
    mat_p.d_col = Pfull->col;
    mat_p.d_val = Pfull->val;

    // spgemm_csrseg_kernel_hash(&mat_a, &mat_p, &mat_c, Plocal, used_by_solver);
#ifdef NSP2_NSPARSE
    nsp2_spgemm_kernel_hash(&mat_a, &mat_p, &mat_c, used_by_solver);
    // nsp2_spgemm_kernel_hash(&mat_a, &mat_p, &mat_c);
#else
    spgemm_csrseg_kernel_hash(&mat_a, &mat_p, &mat_c, Plocal, used_by_solver);
#endif

    mat_c.M = mat_a.M;
    mat_c.N = mat_p.N;
    if (dumpP) {
        char MName[256];
        sprintf(MName, "Alocal_%s", idstring);
        CSRm::printMM(Alocal, MName);
        sprintf(MName, "Pfull_%s", idstring);
        CSRm::printMM(Pfull, MName);
        dumpP = 0;
    }

    CSR* C = CSRm::init(mat_c.M, Pfull->m, mat_c.nnz, false, true, false, Alocal->full_n, Alocal->row_shift);

    C->row = mat_c.d_rpt;
    C->col = mat_c.d_col;
    C->val = mat_c.d_val;
    // ------------- custom cudaMalloc -------------
    C->custom_alloced = true;
    // ---------------------------------------------

    return C;
}

vector<int>* get_missing_col(CSR* Alocal, CSR* Plocal)
{
    _MPI_ENV;
    stype myplastrow;
    if (nprocs == 1) {
        vector<int>* _bitcol = Vector::init<int>(1, true, false);
        return _bitcol;
    }

    // GridBlock gb;
    int* getmct(itype*, itype, itype, itype, int*, int**, int*, int);

    if (Plocal != NULL) {
        myplastrow = Plocal->n - 1;
    } else {
        myplastrow = Alocal->n - 1;
    } // P_n_per_process[i]: number of rows that process i have of matrix P

    if (Alocal->nnz == 0) {
        return NULL;
    }
    int uvs;
    int* ptr = getmct(Alocal->col, Alocal->nnz, 0, myplastrow, &uvs, &(Alocal->bitcol), &(Alocal->bitcolsize), NUM_THR);
    if (uvs == 0) {
        vector<int>* _bitcol = Vector::init<int>(1, true, false);
        return _bitcol;
    } else {
        vector<int>* _bitcol = Vector::init<int>(uvs, false, false);
        _bitcol->val = ptr;
        return _bitcol;
    }
}

vector<int>* get_shrinked_col(CSR* Alocal, CSR* Plocal)
{
    _MPI_ENV;
    stype myplastrow;

    // GridBlock gb;
    int* getmct_4shrink(itype*, itype, itype, itype, int, int*, int**, int*, int*, int);

    if (Plocal != NULL) {
        myplastrow = Plocal->n - 1;
    } else {
        myplastrow = Alocal->n - 1;
    } // P_n_per_process[i]: number of rows that process i have of matrix P

    if (Alocal->nnz == 0) {
        return NULL;
    }
    int uvs;
    int first_or_last = 0;
    if (myid == 0) {
        first_or_last = -1;
    }
    if (myid == (nprocs - 1)) {
        first_or_last = 1;
    }

    int* ptr = getmct_4shrink(Alocal->col, Alocal->nnz, 0, myplastrow, first_or_last, &uvs, &(Alocal->bitcol), &(Alocal->bitcolsize), &(Alocal->post_local), NUM_THR);
    vector<int>* _bitcol = Vector::init<int>(uvs, false, true);
    _bitcol->val = ptr;
    return _bitcol;
}

vector<int>* get_shrinked_col(CSR* Alocal, stype firstlocal, stype lastlocal)
{
    _MPI_ENV;
    stype myplastrow;

    // GridBlock gb;
    int* getmct_4shrink(itype*, itype, itype, itype, int, int*, int**, int*, int*, int);

    myplastrow = lastlocal;

    Alocal->shrinked_lastrow = myplastrow;

    if (Alocal->nnz == 0) {
        return NULL;
    }
    int uvs;
    int first_or_last = 0;
    if (myid == 0) {
        first_or_last = -1;
    }
    if (myid == (nprocs - 1)) {
        first_or_last = 1;
    }

    int* ptr = getmct_4shrink(Alocal->col, Alocal->nnz, 0, myplastrow, first_or_last, &uvs, &(Alocal->bitcol), &(Alocal->bitcolsize), &(Alocal->post_local), NUM_THR);
    vector<int>* _bitcol = Vector::init<int>(uvs, false, true);
    _bitcol->val = ptr;
    return _bitcol;
}

void compute_rows_to_rcv_CPU(CSR* Alocal, CSR* Plocal, vector<int>* _bitcol)
{
    PUSH_RANGE(__func__, 6)

    _MPI_ENV;
    static int cnt = 0;

    if (nprocs == 1) {
        return;
    }

    Alocal->rows_to_get = (rows_to_get_info*)Malloc(sizeof(rows_to_get_info));

    gstype row_shift[nprocs], ends[nprocs];

    itype* P_n_per_process;
    P_n_per_process = (itype*)Malloc(sizeof(itype) * nprocs);

    // send rows numbers to each process, Plocal->n local number of rows
    if (Plocal != NULL) {
        CHECK_MPI(MPI_Allgather(&Plocal->n, sizeof(itype), MPI_BYTE, P_n_per_process, sizeof(itype), MPI_BYTE, MPI_COMM_WORLD));
    } else {
        CHECK_MPI(MPI_Allgather(&Alocal->n, sizeof(itype), MPI_BYTE, P_n_per_process, sizeof(itype), MPI_BYTE, MPI_COMM_WORLD));
    } // P_n_per_process[i]: number of rows that process i owns of matrix P

    CHECK_MPI(
        MPI_Allgather(
            &Alocal->row_shift,
            sizeof(gstype),
            MPI_BYTE,
            row_shift,
            sizeof(gstype),
            MPI_BYTE,
            MPI_COMM_WORLD));

    itype *whichprow = NULL, *rcvpcolxrow = NULL, *rcvprow = NULL;

    int *displr, *displs, *scounts, *rcounts2, *scounts2, *displr2, *displs2;
    int rcounts[nprocs];
    unsigned int* rcvcntp;
    displr = (int*)Malloc(sizeof(int) * nprocs);
    rcounts2 = (int*)Malloc(sizeof(int) * nprocs);
    scounts2 = (int*)Malloc(sizeof(int) * nprocs);
    displs2 = (int*)Malloc(sizeof(int) * nprocs);
    displr2 = (int*)Malloc(sizeof(int) * nprocs);
    displs = (int*)Malloc(sizeof(int) * nprocs);
    scounts = (int*)Malloc(sizeof(int) * nprocs);
    rcvcntp = (unsigned int*)Malloc(sizeof(unsigned int) * nprocs);

    unsigned int countp[nprocs], offset[nprocs];
    unsigned int sothercol = 0;
    int cntothercol = 0;
    int whichproc;
    itype* othercol[1] = { NULL };

    unsigned int i, j;
    cntothercol = _bitcol->n;
    othercol[sothercol] = _bitcol->val;

    // the last list is in othercol[sothercol]
    for (i = 0; i < nprocs; i++) {
        countp[i] = 0;
    }

    itype* aofwhichproc = (itype*)Malloc(sizeof(itype) * cntothercol);

    gsstype cum_p_n_per_process[nprocs];
    ends[0] = P_n_per_process[0];
    cum_p_n_per_process[0] = P_n_per_process[0] - 1;
    for (int i = 1; i < nprocs; i++) {
        cum_p_n_per_process[i] = cum_p_n_per_process[i - 1] + (gstype)P_n_per_process[taskmap[i]];
        ends[i] = ends[i - 1] + ((gstype)P_n_per_process[i]);
    }
    assert(ends[nprocs - 1] == Alocal->full_n);

    itype countall = 0;

    for (j = 0; j < cntothercol; j++) {
        int pwp;
        whichproc = bswhichprocess(cum_p_n_per_process, nprocs, othercol[sothercol][j] + Alocal->row_shift);
        if (whichproc > (nprocs - 1)) {
            whichproc = nprocs - 1;
        }
        pwp = whichproc;
        whichproc = taskmap[whichproc];

        if (whichproc == myid) {
            fprintf(stderr, "Task %d, unexpected whichproc for col %ld (was %d), line=%d\n", myid, othercol[sothercol][j] + Alocal->row_shift, pwp, __LINE__);
            for (itype i = 0; i < nprocs; i++) {
                fprintf(stderr, "cnt=%d,row_shift[%d]=%ld,end[%d]=%ld, cum_p_n_per_process[%d]=%ld,\n", cnt, i, row_shift[i], i, ends[i], i, cum_p_n_per_process[i]);
            }
            exit(1);
        }
        countp[whichproc]++;
        aofwhichproc[countall] = whichproc;
        countall++;
    }
    offset[0] = 0;
    for (i = 1; i < nprocs; i++) {
        offset[i] = offset[i - 1] + countp[i - 1];
        countp[i - 1] = 0;
    }
    countp[nprocs - 1] = 0;
    if (countall > 0) {
        whichprow = (itype*)Malloc(sizeof(itype) * countall);
        Alocal->rows_to_get->whichprow = (gstype*)Malloc(sizeof(gstype) * countall);
        rcvpcolxrow = (itype*)Malloc(sizeof(itype) * countall);
    }

    Alocal->rows_to_get->rows2bereceived = countall;

    for (j = 0; j < cntothercol; j++) {
        whichproc = aofwhichproc[j];
        whichprow[offset[whichproc] + countp[whichproc]] = othercol[sothercol][j] + (Alocal->row_shift - row_shift[whichproc]);
        countp[whichproc]++;
    }
    free(aofwhichproc);

    if (countp[myid] != 0) {
        fprintf(stderr, "self countp should be zero! %d\n", myid);
        exit(1);
    }

    if (MPI_Alltoall(countp, sizeof(itype), MPI_BYTE, rcvcntp, sizeof(itype), MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Error in MPI_Alltoall of P rows\n");
        exit(1);
    }
    if (rcvcntp[myid] != 0) {
        fprintf(stderr, "self rcvcntp should be zero! %d\n", myid);
        exit(1);
    }

    if (Alocal->halo.init == true && 0) {
        countall = 0;
        for (i = 0; i < nprocs; i++) {
            displr[i] = Alocal->halo.to_send_spls[i] * sizeof(itype);
            displs[i] = Alocal->halo.to_receive_spls[i] * sizeof(itype);
            displs2[i] = Alocal->halo.to_send_spls[i] * sizeof(itype);
            displr2[i] = Alocal->halo.to_receive_spls[i] * sizeof(itype);
            scounts[i] = rcounts2[i] = Alocal->halo.to_receive_counts[i] * sizeof(itype);
            rcounts[i] = scounts2[i] = Alocal->halo.to_send_counts[i] * sizeof(itype);
            countall += rcvcntp[i];
        }
    } else {
        countall = 0;
        for (i = 0; i < nprocs; i++) {
            rcounts2[i] = scounts[i] = countp[i] * sizeof(itype);
            displr2[i] = displs[i] = ((i == 0) ? 0 : (displs[i - 1] + scounts[i - 1]));

            scounts2[i] = rcounts[i] = rcvcntp[i] * sizeof(itype);
            displs2[i] = displr[i] = ((i == 0) ? 0 : (displr[i - 1] + rcounts[i - 1]));
            countall += rcvcntp[i];
        }

        if (countall > 0) {
            rcvprow = (itype*)Malloc(sizeof(itype) * countall);
        }

        if (MPI_Alltoallv(whichprow, scounts, displs, MPI_BYTE, rcvprow, rcounts, displr, MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS) {
            fprintf(stderr, "Error in MPI_Alltoallv of whichprow rows\n");
            exit(1);
        }
    }
#if 1
    int k = 0;
    for (int i = 0; i < nprocs; i++) {
        for (int j = 0; j < (scounts[i] / sizeof(itype)); j++) {
            Alocal->rows_to_get->whichprow[k] = whichprow[k] + row_shift[i];
            k++;
        }
    }
    free(whichprow);
#endif
    memset(scounts, 0, nprocs * sizeof(int));
    memset(displs, 0, nprocs * sizeof(int));
    vector<itype>* nnz_per_row_shift = NULL;
    // total_row_to_rec actually store the total rows to send, the sum of the number of rows we must send to each process i
    // rcvcntp[i] = number of rows to send to process i
    itype total_row_to_rec = countall;
    countall = 0;
    if (total_row_to_rec) {
        itype q = 0;
        itype tot_shift = 0;
        nnz_per_row_shift = Vector::init<itype>(total_row_to_rec, true, false); // no temp buff is used on the HOST only;

        q = 0;
        for (i = 0; i < nprocs; i++) {
            displs[i] = (i == 0) ? 0 : (displs[i - 1] + scounts[i - 1]);
            if (i == myid) {
                continue;
            }
            for (j = 0; j < rcvcntp[i]; j++) {
                scounts[i] += 1;
                nnz_per_row_shift->val[q] = tot_shift;
                tot_shift += 1;
                q++;
            }
            countall += scounts[i];
            scounts[i] *= sizeof(itype);
            displs[i] = ((i == 0) ? 0 : (displs[i - 1] + scounts[i - 1]));
        }
    }
    // Alocal->rows_to_get->total_row_to_rec = total_row_to_rec;
    Alocal->rows_to_get->nnz_per_row_shift = nnz_per_row_shift;
    Alocal->rows_to_get->countall = countall /* Alocal->hi.to_receive_n */;

    if (Alocal->halo.init == true && 0) {
        Alocal->rows_to_get->rcvprow = Alocal->halo.to_send->val;
        /*     Alocal->rows_to_get->whichprow = Alocal->halo.to_receive->val; */
    } else {
        Alocal->rows_to_get->rcvprow = rcvprow;
    }
    Alocal->rows_to_get->rcvpcolxrow = rcvpcolxrow;
    Alocal->rows_to_get->displr = displr;
    Alocal->rows_to_get->displs = displs;
    Alocal->rows_to_get->scounts = scounts;
    Alocal->rows_to_get->rcounts2 = rcounts2;
    Alocal->rows_to_get->scounts2 = scounts2;
    Alocal->rows_to_get->displs2 = displs2;
    Alocal->rows_to_get->displr2 = displr2;
    Alocal->rows_to_get->rcvcntp = rcvcntp;
    Alocal->rows_to_get->P_n_per_process = P_n_per_process;

    cnt++;

    POP_RANGE
    return;
}

__global__ void _completedP_rows(itype completedP_n, itype* new_rows)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < completedP_n) {
        new_rows[id] = id;
    }
}

__global__ void _completedP_rows2(itype completedP_n, itype rows_pre_local, itype local_rows, itype nzz_pre_local, itype Plocal_nnz, itype* Plocal_row, itype* P_nnz_map, itype* completedP_row)
{
    itype id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < completedP_n) {
        if (id < rows_pre_local) {
            completedP_row[id] = P_nnz_map[id];
        } else {
            if (id < rows_pre_local + local_rows) {
                completedP_row[id] = nzz_pre_local + Plocal_row[id - rows_pre_local]; // id - rows_pre_local
            } else {
                completedP_row[id] = nzz_pre_local + Plocal_nnz + P_nnz_map[id - local_rows]; // id - local_rows
            }
        }
    }
}

CSR* nsparseMGPU_noCommu_new(handles* h, CSR* Alocal, CSR* Plocal, bool used_by_solver)
{
    PUSH_RANGE(__func__, 6)

    _MPI_ENV;

    CSR* Alocal_ = get_shrinked_matrix(Alocal, Plocal);

    cudaDeviceSynchronize();

    csrlocinfo Pinfo1p;
    Pinfo1p.fr = 0;
    Pinfo1p.lr = Plocal->n;
    Pinfo1p.row = Plocal->row;
    Pinfo1p.col = NULL;
    Pinfo1p.val = Plocal->val;

    cudaDeviceSynchronize();

    CSR* C = nsparseMGPU(Alocal_, Plocal, &Pinfo1p, used_by_solver);
    Alocal_->col = NULL;
    Alocal_->row = NULL;
    Alocal_->val = NULL;
    std::free(Alocal_);

    POP_RANGE
    return C;
}

CSR* nsparseMGPU_commu_new(handles* h, CSR* Alocal, CSR* Plocal, bool used_by_solver)
{
    PUSH_RANGE(__func__, 6)

    _MPI_ENV;

    csrlocinfo Pinfo1p;
    Pinfo1p.fr = 0;
    Pinfo1p.lr = Plocal->n;
    Pinfo1p.row = Plocal->row;
#if !defined(CSRSEG)
    Pinfo1p.col = NULL;
#else
    Pinfo1p.col = Plocal->col;
#endif
    Pinfo1p.val = Plocal->val;

    if (nprocs == 1) {
        return nsparseMGPU(Alocal, Plocal, &Pinfo1p, used_by_solver);
    }
    static int cnt = 0;
    itype *Pcol, *col2send = NULL;
    vtype* Pval;
    vtype* val2send = NULL;
    GridBlock gb;
    unsigned int i, j, k;

    int *displr, *displs, *scounts, *rcounts2, *scounts2, *displr2, *displs2;
    int rcounts[nprocs];
    int rcounts_src[nprocs], displr_src[nprocs];
    int displr_target[nprocs];
    unsigned int* rcvcntp;
    itype* P_n_per_process;
    P_n_per_process = Alocal->rows_to_get->P_n_per_process;
    displr = Alocal->rows_to_get->displr;
    rcounts2 = Alocal->rows_to_get->rcounts2;
    scounts2 = Alocal->rows_to_get->scounts2;
    displs2 = Alocal->rows_to_get->displs2;
    displr2 = Alocal->rows_to_get->displr2;
    rcvcntp = Alocal->rows_to_get->rcvcntp;
    displs = Alocal->rows_to_get->displs;
    scounts = Alocal->rows_to_get->scounts;

    itype mycolp = Plocal->nnz; // number of nnz stored by the process
    gstype Pm = (unsigned long)Plocal->m; // number of columns in P
    gstype mypfirstrow = Plocal->row_shift;
    gstype myplastrow = Plocal->n + Plocal->row_shift - 1;
    // itype Pn = Plocal->full_n;

    vector<itype>* nnz_per_row_shift = NULL;
    nnz_per_row_shift = Alocal->rows_to_get->nnz_per_row_shift;

    itype* p2rcvprow;
    itype countall = 0;
    countall = Alocal->rows_to_get->countall;
    itype q = 0;

    memset(rcounts, 0, nprocs * sizeof(int));

    gstype* whichprow = NULL;
    itype *rcvpcolxrow = NULL, *rcvprow = NULL;
    rcvprow = Alocal->rows_to_get->rcvprow;
    whichprow = Alocal->rows_to_get->whichprow;
    rcvpcolxrow = Alocal->rows_to_get->rcvpcolxrow;

    itype* dev_col2send = idevtemp1;
    vtype* dev_val2send = vdevtemp1;
    if (countall > 0) {
        col2send = iAtemp1;
        val2send = vAtemp1;
        // ------------- TEST -----------------
        static int nnz_per_row_shift_n_stat = 0;
        // ------------------------------------

        // sync call to make async stream1 stream2 one event cp1
        vector<itype>* dev_nnz_per_row_shift = NULL;
        if (nnz_per_row_shift->n > 0) {
            if (nnz_per_row_shift->n > 2000000) {
                fprintf(stderr, "Task %d, n=%d\n", myid, nnz_per_row_shift->n);
                exit(0);
            }
            dev_nnz_per_row_shift = Vector::init<itype>(nnz_per_row_shift->n, false, true);
            dev_nnz_per_row_shift->val = idevtemp2;

            CHECK_DEVICE(cudaMemcpyAsync(dev_nnz_per_row_shift->val, nnz_per_row_shift->val, dev_nnz_per_row_shift->n * sizeof(itype), cudaMemcpyHostToDevice, h->stream1));
            // ---------------------------------- TEST --------------------------------------------------
            if (dev_nnz_per_row_shift->n > nnz_per_row_shift_n_stat) {
                if (nnz_per_row_shift_n_stat > 0) {
                    MY_CUDA_CHECK(cudaFree(dev_rcvprow_stat));
                }
                nnz_per_row_shift_n_stat = dev_nnz_per_row_shift->n;
                cudaMalloc_CNT
                    CHECK_DEVICE(cudaMalloc((void**)&dev_rcvprow_stat, nnz_per_row_shift_n_stat * sizeof(itype)));
            }
            CHECK_DEVICE(cudaMemcpyAsync(dev_rcvprow_stat, rcvprow, dev_nnz_per_row_shift->n * sizeof(itype), cudaMemcpyHostToDevice, h->stream1));
            // ------------------------------------------------------------------------------------------
        }
        q = 0;
        for (i = 0; i < nprocs; i++) {
            if (i == myid) {
                continue;
            }
            // shift
            p2rcvprow = &rcvprow[displr[i] / sizeof(itype)]; // recvprow will be modified
            for (j = 0; j < rcvcntp[i]; j++) {
                p2rcvprow[j] = 1; // recycle rcvprow to send the number of columns in each row
                q++;
            }
        }
        if (nnz_per_row_shift->n > 0) {
            gb = gb1d(dev_nnz_per_row_shift->n, NUM_THR);
            // -------- TEST ---------
            _getColVal<<<gb.g, gb.b, 0, h->stream1>>>(dev_nnz_per_row_shift->n, dev_rcvprow_stat, dev_nnz_per_row_shift->val, Plocal->row, Plocal->col, Plocal->val, dev_col2send, dev_val2send, 0 /* mypfirstrow */);
            // -----------------------
            cudaStreamSynchronize(h->stream1);
        }

        CHECK_DEVICE(cudaMemcpyAsync(col2send, dev_col2send, countall * sizeof(itype), cudaMemcpyDeviceToHost, h->stream1));
        CHECK_DEVICE(cudaMemcpyAsync(val2send, dev_val2send, countall * sizeof(vtype), cudaMemcpyDeviceToHost, h->stream2));

        if (nnz_per_row_shift->n > 0) {
            dev_nnz_per_row_shift->val = NULL;
            std::free(dev_nnz_per_row_shift);
        }
    }

    if (MPI_Alltoallv(rcvprow, scounts2, displs2, MPI_BYTE, rcvpcolxrow, rcounts2, displr2, MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Error in MPI_Alltoallv of rcvprow\n");
        exit(1);
    }

    itype nzz_pre_local = 0;
    itype rows2bereceived = Alocal->rows_to_get->rows2bereceived;

    if (rows2bereceived) {
        // we have rows from other process
        bool flag = true;
        gsstype r = 0;

        vector<itype>* P_nnz_map = Vector::init<itype>(rows2bereceived, true, false);
        k = 0;
        int whichproc;

        gsstype cum_p_n_per_process[nprocs];
        cum_p_n_per_process[0] = P_n_per_process[0] - 1;
        for (int j = 1; j < nprocs; j++) {
            cum_p_n_per_process[j] = cum_p_n_per_process[j - 1] + P_n_per_process[taskmap[j]];
        }
        for (i = 0; i < rows2bereceived; i++) {
            r = whichprow[i];
            // count nnz per process for comunication
            int pwp;
            whichproc = bswhichprocess(cum_p_n_per_process, nprocs, r);
            if (whichproc > (nprocs - 1)) {
                whichproc = nprocs - 1;
            }
            pwp = whichproc;
            whichproc = taskmap[whichproc];
            if (whichproc == myid) {
                fprintf(stderr, "Task %d, unexpected whichproc for col %ld (was %d), line=%d\n", myid, r + Alocal->row_shift, pwp, __LINE__);
                exit(1);
            }
            rcounts[whichproc] += rcvpcolxrow[i];
            if (r < mypfirstrow) {
                nzz_pre_local += rcvpcolxrow[i];
                flag = false;
            }
            k += rcvpcolxrow[i];
            P_nnz_map->val[i] = k;
        }

        Vector::free(P_nnz_map);
    }

    if (rcounts[myid] != 0) {
        fprintf(stderr, "task: %d, unexpected rcount[%d]=%d. It should be zero\n", myid, myid, rcounts[myid]);
        exit(1);
    }

    int totcell = 0;
    static int s_totcell_new;
    memcpy(rcounts_src, rcounts, nprocs * sizeof(itype));
    itype whichdisplr[nprocs], ndisplr = 0;
    {
        for (int i = 0; i < nprocs; i++) {
            if (rcounts[i] > 0 || i == myid) {
                whichdisplr[ndisplr] = i;
                ndisplr++;
            }
        }
        itype tempind;
        for (int i = 0; i < ndisplr - 1; i++) {
            // Last i elements are already in place
            for (int j = 0; j < ndisplr - i - 1; j++) {
                if (itaskmap[whichdisplr[j]] > itaskmap[whichdisplr[j + 1]]) {
                    tempind = whichdisplr[j];
                    whichdisplr[j] = whichdisplr[j + 1];
                    whichdisplr[j + 1] = tempind;
                }
            }
        }
    }
    int previous_index = 0;
    int flagaddmycolp = 1;
    for (j = 0; j < ndisplr; j++) {
        i = whichdisplr[j];
        totcell += rcounts[i];
        displr_target[i] = (j == 0) ? 0 : (displr_target[previous_index] + (((itaskmap[i] > itaskmap[myid]) && flagaddmycolp) ? mycolp : rcounts_src[previous_index]));
        if (itaskmap[i] > itaskmap[myid]) {
            flagaddmycolp = 0;
        }
        rcounts[i] *= sizeof(itype);
        displr[i] = (j == 0) ? 0 : (displr[previous_index] + rcounts[previous_index]);
        displr_src[i] = displr[i] / sizeof(itype);
        previous_index = i;
    }
    if (iPtemp1 == NULL && totcell > 0) { // first allocation
        cudaMalloc_CNT
            MY_CUDA_CHECK(cudaMallocHost(&iPtemp1, totcell * sizeof(itype)));
        vPtemp1 = (vtype*)Malloc(totcell * sizeof(vtype));
        s_totcell_new = totcell;
    }
    if (totcell > s_totcell_new) { // not enough space
        MY_CUDA_CHECK(cudaFreeHost(iPtemp1));
        printf("[Realloc] --- totcell: %d s_totcell_new: %d\n", totcell, s_totcell_new);
        cudaMalloc_CNT
            MY_CUDA_CHECK(cudaMallocHost(&iPtemp1, sizeof(itype) * totcell));
        vPtemp1 = (vtype*)Realloc(vPtemp1, totcell * sizeof(vtype));
        s_totcell_new = totcell;
    }
    Pcol = iPtemp1;
    Pval = vPtemp1;
    if (countall > 0) {
        cudaStreamSynchronize(h->stream1);
        dev_col2send = NULL;
    }
    if (MPI_Alltoallv(col2send, scounts, displs, MPI_BYTE, Pcol, rcounts, displr, MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Error in MPI_Alltoallv of col2send\n");
        exit(1);
    }

    CSR* completedP = NULL;
    itype completedP_n = Plocal->n + rows2bereceived; // Alocal->rows_to_get->total_row_to_rec; (?NOTE?)
    itype completedP_nnz = Plocal->nnz + totcell;

    // ------------------------------------- TEST -------------------------------------------
    static int completedP_stat_nnz = 0, completedP_stat_n = 0;

    completedP = CSRm::init(completedP_n, Pm, completedP_nnz, false, true, false, completedP_n, Alocal->row_shift);
    if (completedP_n > completedP_stat_n || completedP_nnz > completedP_stat_nnz) {
        cudaError_t err;
        if (completedP_n > completedP_stat_n) {
            if (completedP_stat_n > 0) {
                MY_CUDA_CHECK(cudaFree(completedP_stat_row));
            }
            completedP_stat_n = completedP_n;
            cudaMalloc_CNT
                err
                = cudaMalloc((void**)&completedP_stat_row, (completedP_stat_n + 1) * sizeof(itype));
            CHECK_DEVICE(err);
        }
        if (completedP_nnz > completedP_stat_nnz) {
            if (completedP_stat_nnz > 0) {
                MY_CUDA_CHECK(cudaFree(completedP_stat_col));
                MY_CUDA_CHECK(cudaFree(completedP_stat_val));
            }
            completedP_stat_nnz = completedP_nnz;
            cudaMalloc_CNT
                err
                = cudaMalloc((void**)&completedP_stat_val, completedP_stat_nnz * sizeof(vtype));
            CHECK_DEVICE(err);
            cudaMalloc_CNT
                err
                = cudaMalloc((void**)&completedP_stat_col, completedP_stat_nnz * sizeof(itype));
            CHECK_DEVICE(err);
        }
    }
    completedP->val = completedP_stat_val;
    completedP->col = completedP_stat_col;
    completedP->row = completedP_stat_row;
    // --------------------------------------------------------------------------------------

    gb = gb1d(completedP_n + 1, NUM_THR);
    _completedP_rows<<<gb.g, gb.b>>>(completedP_n + 1, completedP->row);

    for (j = 0; j < ndisplr; j++) {
        i = whichdisplr[j];
        if (rcounts_src[i] > 0) {
            CHECK_DEVICE(cudaMemcpyAsync(completedP->col + displr_target[i], Pcol + displr_src[i], rcounts_src[i] * sizeof(itype), cudaMemcpyHostToDevice, h->stream1));
        }
    }

    col2send = NULL;
    for (i = 0; i < nprocs; i++) {
        scounts[i] *= (sizeof(vtype) / sizeof(itype));
        displs[i] *= (sizeof(vtype) / sizeof(itype));
        rcounts[i] *= (sizeof(vtype) / sizeof(itype));
        displr[i] *= (sizeof(vtype) / sizeof(itype));
    }
    if (countall > 0) {
        cudaStreamSynchronize(h->stream2);
        dev_val2send = NULL;
    }
    if (MPI_Alltoallv(val2send, scounts, displs, MPI_BYTE, Pval, rcounts, displr, MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Error in MPI_Alltoallv of val2send\n");
        exit(1);
    }
    val2send = NULL;

    for (j = 0; j < ndisplr; j++) {
        i = whichdisplr[j];
        if (rcounts_src[i] > 0) {
            CHECK_DEVICE(cudaMemcpy(completedP->val + displr_target[i], Pval + displr_src[i], rcounts_src[i] * sizeof(vtype), cudaMemcpyHostToDevice));
        }
    }

#if !defined(CSRSEG)
    CHECK_DEVICE(cudaMemcpy(completedP->val + nzz_pre_local, Plocal->val, Plocal->nnz * sizeof(vtype), cudaMemcpyDeviceToDevice););
    CHECK_DEVICE(cudaMemcpy(completedP->col + nzz_pre_local, Plocal->col, Plocal->nnz * sizeof(itype), cudaMemcpyDeviceToDevice););
#endif

    csrlocinfo Plocalinfo;
#if !defined(CSRSEG)
    Plocalinfo.fr = mypfirstrow;
    Plocalinfo.lr = myplastrow;
    Plocalinfo.row = completedP->row;
    Plocalinfo.col = NULL;
    Plocalinfo.val = completedP->val;
#else
    Plocalinfo.fr = nzz_pre_local;
    Plocalinfo.lr = nzz_pre_local + Plocal->nnz;
    Plocalinfo.row = Plocal->row;
    Plocalinfo.col = completedP->col + nzz_pre_local;
    Plocalinfo.val = Plocal->val;
#endif

    CSR* Alocal_ = get_shrinked_matrix(Alocal, Plocal);

    cudaDeviceSynchronize();
    if (Alocal_->m != completedP->n) {
        fprintf(stderr, "[%d] Alocal_->m = %d != %d = completedP->n (totcell = %d, Plocal->n = %d, countall = %d, rows2berecived = %d, nzz_pre_local=%d)\n", myid, Alocal_->m, completedP->n, totcell, Plocal->n, countall, Alocal->rows_to_get->rows2bereceived, nzz_pre_local);
    }
    assert(Alocal_->m == completedP->n);

    CSR* C = nsparseMGPU(Alocal_, completedP, &Plocalinfo, used_by_solver);

    Pcol = NULL;
    Pval = NULL; // memory will free up in AMG
    CSRm::free_rows_to_get(Alocal);

    // --------------- TEST ----------------------
    std::free(completedP);
    // -------------------------------------------

    Alocal_->col = NULL;
    Alocal_->row = NULL;
    Alocal_->val = NULL;
    std::free(Alocal_);

    cnt++;

    POP_RANGE
    return C;
}
