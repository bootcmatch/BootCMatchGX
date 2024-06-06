#include "basic_kernel/halo_communication/halo_communication.h"
#include "basic_kernel/halo_communication/local_permutation.h"

#include "utility/cudamacro.h"
#include "utility/function_cnt.h"
#include "utility/utils.h"

#define USE_GETMCT
#define NUM_THR 1024
extern int scalennzmiss;
extern int* taskmap;

__global__ void _getMissingMask(itype nnz, itype* A_col, itype* missing, itype row_shift, itype n)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nnz) {
        return;
    }

    itype col = A_col[i];

    if (col < row_shift || col >= row_shift + n) {
        missing[i] = col;
    } else {
        missing[i] = -1;
    }
}

__global__ void _count(itype nnz, itype* missing, itype* c)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nnz) {
        return;
    }

    if (missing[i]) {
        atomicAdd(c, 1);
    }
}

void getMissing(CSR* A, vector<itype>** missing, CSR* R, gstype* row_shift)
{
    _MPI_ENV;

    stype row_ns[nprocs];
    gstype ends[nprocs];

    CHECK_MPI(
        MPI_Allgather(
            R == NULL ? &A->n : &R->n,
            sizeof(stype),
            MPI_BYTE,
            row_ns,
            sizeof(stype),
            MPI_BYTE,
            MPI_COMM_WORLD));

    CHECK_MPI(
        MPI_Allgather(
            R == NULL ? &A->row_shift : &R->row_shift,
            sizeof(gstype),
            MPI_BYTE,
            row_shift,
            sizeof(gstype),
            MPI_BYTE,
            MPI_COMM_WORLD));

    ends[0] = row_ns[0];
    for (itype i = 1; i < nprocs; i++) {
        ends[i] = row_ns[taskmap[i]] + ends[i - 1];
    }

    if (R == NULL) {
        assert(ends[nprocs - 1] == A->full_n);
    } else {
        assert(ends[nprocs - 1] == R->full_n);
    }

#ifndef USE_GETMCT
    itype nnz = 0;

    if (R != NULL) {
        nnz = R->nnz;
    } else {
        nnz = A->nnz;
    }

    vector<itype>*mask, *mask_sorted;
    Vectorinit_CNT
        mask
        = Vector::init<itype>(nnz, true, true);
    scalar<itype>* d_num_selected_out = Scalar::init<itype>(0, true);

    Vectorinit_CNT
        mask_sorted
        = Vector::init<itype>(nnz, true, true);

    GridBlock gb;

    if (R != NULL) {
        gb = gb1d(nnz, BLOCKSIZE);
        _getMissingMask<<<gb.g, gb.b>>>(nnz, R->col, mask->val, A->row_shift, A->n);
    } else {
        gb = gb1d(nnz, BLOCKSIZE);
        _getMissingMask<<<gb.g, gb.b>>>(nnz, A->col, mask->val, A->row_shift, A->n);
    }

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        mask->val,
        mask_sorted->val,
        nnz);

    cudaMalloc_CNT
        CHECK_DEVICE(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceRadixSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        mask->val,
        mask_sorted->val,
        nnz);

    MY_CUDA_CHECK(cudaFree(d_temp_storage));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;

    cub::DeviceSelect::Unique(
        d_temp_storage,
        temp_storage_bytes,
        mask_sorted->val,
        mask->val,
        d_num_selected_out->val,
        nnz);

    cudaMalloc_CNT
        CHECK_DEVICE(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceSelect::Unique(
        d_temp_storage,
        temp_storage_bytes,
        mask_sorted->val,
        mask->val,
        d_num_selected_out->val,
        nnz);

    itype* cp = Scalar::getvalueFromDevice(d_num_selected_out);
    itype c = (*cp) - 1;
    free(cp);

    for (int i = 0; i < nprocs; i++) {
        if (i != myid) {
            missing[i]->n = 0;
        }
    }

    if (c > 0) {
        itype* missing_flat = NULL;
        missing_flat = (itype*)Malloc(sizeof(itype) * c);
        CHECK_HOST(missing_flat);
        cudaMemcpy(missing_flat, mask->val + 1, c * sizeof(itype), cudaMemcpyDeviceToHost);

        itype J = 0, I = 0;
        for (itype i = 0; i < c; i++) {
            itype j = missing_flat[i];
            assert((I + 1) < (A->nnz / nprocs));

        CHECK_AGAIN:
            if (j >= ends[J]) {

                if (taskmap[J] != myid) {
                    missing[taskmap[J]]->n = I;
                }

                J++;
                I = 0;
                goto CHECK_AGAIN;
            }
            missing[taskmap[J]]->val[I] = j;
            I++;
        }

        if (I) {
            missing[taskmap[J]]->n = I;
        }

        free(missing_flat);
    }

    Vector::free(mask);
    Vector::free(mask_sorted);
    Scalar::free(d_num_selected_out);

#else
    // -----------------------------

    gstype mypfirstrow;
    mypfirstrow = R == NULL ? A->row_shift : R->row_shift;

    int uvs;
    int* getmct(itype*, itype, itype, itype, int*, int**, int*, int);
    int* ptr;
    if (R != NULL) {
        ptr = getmct(R->col, R->nnz, 0, R->n - 1, &uvs, &(R->bitcol), &(R->bitcolsize), NUM_THR);
    } else {
        ptr = getmct(A->col, A->nnz, 0, A->n - 1, &uvs, &(A->bitcol), &(A->bitcolsize), NUM_THR);
    }

    vector<int>* _bitcol;
    if (uvs == 0) {
        _bitcol = Vector::init<int>(1, true, false);
    } else {
        _bitcol = Vector::init<int>(uvs, false, false);
        _bitcol->val = ptr;
    }

    for (int i = 0; i < nprocs; i++) {
        if (i != myid) {
            missing[i]->n = 0;
        }
    }

    if (uvs > 0) {
        stype* missing_flat = NULL;
        missing_flat = (stype*)Malloc(sizeof(stype) * uvs);
        CHECK_HOST(missing_flat);
        memcpy(missing_flat, _bitcol->val, uvs * sizeof(stype));
        mypfirstrow = (R == NULL) ? A->row_shift : R->row_shift;
        stype J = 0, I = 0;
        for (itype i = 0; i < uvs; i++) {
            itype j = missing_flat[i];
            assert((I + 1) < ((((long)(A->nnz) * (long)scalennzmiss)) / ((long)nprocs)));

        CHECK_AGAIN:
            if ((j + mypfirstrow) >= ends[J]) {

                if (taskmap[J] != myid) {
                    missing[taskmap[J]]->n = I;
                }

                J++;
                I = 0;
                goto CHECK_AGAIN;
            }

            if (J >= nprocs) {
                fprintf(stderr, "Task id %d: unexpected missing source :%d\n", myid, J);
                fprintf(stderr, "Task=%d,R=%x,missing %d+%d\n", myid, R, j, mypfirstrow);
                for (int k = 0; k < nprocs; k++) {
                    fprintf(stderr, "Task=%d, ends[%d]=%d\n", myid, k, ends[k]);
                }
                char WrongMatName[256];
                sprintf(WrongMatName, "%s_%s", (R == NULL) ? "WA" : "WR", idstring);
                CSRm::printMM((R == NULL) ? A : R, WrongMatName);
                exit(1);
            }
            missing[taskmap[J]]->val[I] = j + (mypfirstrow - row_shift[taskmap[J]]);
            I++;
        }
        if (I) {
            missing[taskmap[J]]->n = I;
        }
        free(missing_flat);
    }
#endif
}

__global__ void _getToSendMask(itype n, itype* to_send, itype* to_send_mask, itype shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    to_send_mask[to_send[i] - shift] = i;
}

halo_info haloSetup(CSR* A, CSR* R = NULL)
{
    PUSH_RANGE(__func__, 5)

    _MPI_ENV;
    if (R != NULL) { // product compatibility check
        assert(R->m == A->full_n);
    } else {
        if (A->m != A->full_n) {
            fprintf(stderr, "Task %d, in haloSetup: A->m=%lu, A->full_n=%lu\n", myid, A->m, A->full_n);
            fflush(stderr);
        }
        assert(A->m == A->full_n);
    }
    vector<itype>** my_missing;

    int* sendcounts = (int*)Malloc(sizeof(int) * nprocs);
    int* sdispls = (int*)Malloc(sizeof(int) * nprocs);
    int* recvcounts = (int*)Malloc(sizeof(int) * nprocs);
    int* rdispls = (int*)Malloc(sizeof(int) * nprocs);

    vector<itype>* my_missing_flat = NULL;
    vector<itype>* their_missing_flat = NULL;
    itype their_missing_flat_total_n = 0;
    itype total_n = 0;
    gstype row_shift[nprocs];
    halo_info hi;

    if (1 || (A->rows_to_get == NULL || R != NULL)) {
        my_missing = (vector<itype>**)Malloc(sizeof(vector<itype>*) * nprocs);
        CHECK_HOST(my_missing);

        for (int i = 0; i < nprocs; i++) {
            if (i != myid) {
                my_missing[i] = Vector::init<itype>(((long)(A->nnz) * (long)(scalennzmiss)) / ((long)(nprocs)), true, false);
            } else {
                my_missing[i] = NULL;
            }
        }

        getMissing(A, my_missing, R, row_shift);

        for (itype i = 0; i < nprocs; i++) {
            if (myid == i) {
                continue;
            }
            total_n += my_missing[i]->n;
        }

        if (total_n > 0) {
            my_missing_flat = Vector::init<itype>(total_n, true, false);
            hi.to_receive = Vector::init<gstype>(total_n, true, false);
        }

        itype shift = 0;
        for (itype i = 0; i < nprocs; i++) {

            if (myid == i) {
                sendcounts[i] = 0;
                sdispls[i] = shift;
                continue;
            }

            if (my_missing[i]->n > 0) {
                memcpy(my_missing_flat->val + shift, my_missing[i]->val, my_missing[i]->n * sizeof(itype));
            }

            sendcounts[i] = my_missing[i]->n;
            sdispls[i] = shift;
            shift += my_missing[i]->n;
        }

        CHECK_MPI(
            MPI_Alltoall(
                sendcounts,
                1,
                ITYPE_MPI,
                recvcounts,
                1,
                ITYPE_MPI,
                MPI_COMM_WORLD));

        shift = 0;
        for (itype i = 0; i < nprocs; i++) {
            rdispls[i] = shift;
            shift += recvcounts[i];
            their_missing_flat_total_n += recvcounts[i];
        }

        if (their_missing_flat_total_n > 0) {
            their_missing_flat = Vector::init<itype>(their_missing_flat_total_n, true, false);
        }

        CHECK_MPI(
            MPI_Alltoallv(
                my_missing_flat != NULL ? my_missing_flat->val : NULL,
                sendcounts,
                sdispls,
                ITYPE_MPI,
                their_missing_flat != NULL ? their_missing_flat->val : NULL,
                recvcounts,
                rdispls,
                ITYPE_MPI,
                MPI_COMM_WORLD));
    }

    hi.init = true;
    if (1 || (A->rows_to_get == NULL || R != NULL)) {
        gstype reorderspls[nprocs];
        itype whichspls[nprocs], nspls = 0;
        hi.to_receive_n = total_n;
        int k = 0;
        for (int i = 0; i < nprocs; i++) {
            for (int j = 0; j < sendcounts[i]; j++) {
                hi.to_receive->val[k] = my_missing_flat->val[k] + row_shift[i];
                if (j == 0) {
                    reorderspls[nspls] = hi.to_receive->val[k];
                    whichspls[nspls] = i;
                    nspls++;
                }
                if (0 && myid == 1) {
                    printf("Task %d, receiving %d from %d\n", myid, hi.to_receive->val[k], i);
                }
                k++;
            }
        }
        gstype tempreorder;
        itype tempind;
        for (int i = 0; i < nspls - 1; i++) {
            // Last i elements are already in place
            for (int j = 0; j < nspls - i - 1; j++) {
                if (reorderspls[j] > reorderspls[j + 1]) {
                    tempreorder = reorderspls[j];
                    reorderspls[j] = reorderspls[j + 1];
                    reorderspls[j + 1] = tempreorder;
                    tempind = whichspls[j];
                    whichspls[j] = whichspls[j + 1];
                    whichspls[j + 1] = tempind;
                }
            }
        }
        itype shift = 0;
        for (int i = 0; i < nspls; i++) {
            sdispls[whichspls[i]] = shift;
            shift += sendcounts[whichspls[i]];
        }
        hi.to_receive_counts = sendcounts;
        hi.to_receive_spls = sdispls;
    } else {
        for (int i = 0; i < nprocs; i++) {
            A->halo.to_receive_counts[i] = A->rows_to_get->rcounts2[i] / sizeof(itype);
            A->halo.to_receive_spls[i] = A->rows_to_get->displr2[i] / sizeof(itype);
        }
        hi.to_receive = Vector::init<gstype>(A->rows_to_get->countall, true, false);
        hi.to_receive_n = A->rows_to_get->countall;
        hi.to_receive->val = A->rows_to_get->whichprow;
    }

    hi.what_to_receive = NULL;
    hi.to_receive_d = NULL;
    if (hi.to_receive_n > 0) {
        hi.what_to_receive = (vtype*)Malloc(sizeof(vtype) * hi.to_receive_n);
        CHECK_HOST(hi.what_to_receive);
        cudaMalloc_CNT
            CHECK_DEVICE(cudaMalloc((void**)&hi.what_to_receive_d, sizeof(vtype) * hi.to_receive_n));
        VectorcopyToDevice_CNT
            hi.to_receive_d
            = Vector::copyToDevice(hi.to_receive);
    }
    if (1 || (A->rows_to_get == NULL || R != NULL)) {
        hi.to_send = their_missing_flat;
        hi.to_send_n = their_missing_flat_total_n;
        hi.to_send_counts = recvcounts;
        hi.to_send_spls = rdispls;
    } else {
        for (int i = 0; i < nprocs; i++) {
            A->halo.to_send_counts[i] = A->rows_to_get->scounts2[i] / sizeof(itype);
            A->halo.to_send_spls[i] = A->rows_to_get->displs2[i] / sizeof(itype);
            their_missing_flat_total_n += A->halo.to_send_counts[i];
        }
        hi.to_send = Vector::init<itype>(their_missing_flat_total_n, true, false);
        hi.to_send_n = their_missing_flat_total_n;
        hi.to_send->val = A->rows_to_get->rcvprow;
    }

    hi.what_to_send_d = NULL;
    hi.what_to_send = NULL;

    if (hi.to_send_n > 0) {

        cudaMalloc_CNT
            CHECK_DEVICE(cudaMallocHost((void**)&hi.what_to_send, sizeof(vtype) * hi.to_send_n));

        CHECK_HOST(hi.what_to_send);
        cudaMalloc_CNT
            CHECK_DEVICE(cudaMalloc((void**)&hi.what_to_send_d, sizeof(vtype) * hi.to_send_n));
        VectorcopyToDevice_CNT
            hi.to_send_d
            = Vector::copyToDevice(hi.to_send);
    }

    if (1 || (A->rows_to_get == NULL || R != NULL)) {
        for (itype i = 0; i < nprocs; i++) {
            if (my_missing[i] != NULL) {
                Vector::free(my_missing[i]);
            }
        }
        std::free(my_missing);
        if (my_missing_flat != NULL) {
            Vector::free(my_missing_flat);
        }
    }
    POP_RANGE
    return hi;
}

__global__ void _getToSend(itype n, vtype* x, vtype* what_to_send, itype* to_send, itype shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    itype j = to_send[i];
    what_to_send[i] = x[j + shift];
}

__global__ void _getToSend_new(itype n, vtype* x, vtype* what_to_send, itype* to_send, itype shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    itype j = to_send[i];
    what_to_send[i] = x[j];
}

__global__ void setReceivedWithMask(itype n, vtype* x, vtype* received, gstype* receive_map, itype shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    itype j = receive_map[i];
    vtype val = received[i];
    x[j] = val;
}

__global__ void setReceivedWithMask_new(itype n, vtype* x, vtype* received, gstype* receive_map, itype shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }
}

#define SYNCSOL_TAG 4321
#define MAXNTASKS 4096

void halo_sync(halo_info hi, CSR* A, vector<vtype>* x, bool local_flag)
{
    _MPI_ENV;

    assert(A->on_the_device);
    assert(x->on_the_device);
    static MPI_Request requests[MAXNTASKS];
    static cudaStream_t sync_stream;
    static int first = 1;
    if (first) {
        first = 0;
        CHECK_DEVICE(cudaStreamCreate(&sync_stream));
    }

    GridBlock gb;

    if (hi.to_send_n) {
#if SMART_AGGREGATE_GETSET_GPU == 1
        GridBlock gb = gb1d(hi.to_send_n, BLOCKSIZE);
        if (local_flag) {
            _getToSend_new<<<gb.g, gb.b, 0, sync_stream>>>(hi.to_send_d->n, x->val, hi.what_to_send_d, hi.to_send_d->val, A->row_shift);
        } else {
            _getToSend<<<gb.g, gb.b, 0, sync_stream>>>(hi.to_send_d->n, x->val, hi.what_to_send_d, hi.to_send_d->val, A->row_shift);
        }
        CHECK_DEVICE(cudaMemcpyAsync(hi.what_to_send, hi.what_to_send_d, hi.to_send_n * sizeof(vtype), cudaMemcpyDeviceToHost, sync_stream));
#else
        vector<vtype>* x_host = Vector::copyToHost(x);
        int start = 0;
        for (int i = 0; i < nprocs; i++) {
            int end = start + hi.to_send_counts[i];
            for (int j = start; j < end; j++) {
                itype v = hi.to_send->val[j];
                hi.what_to_send[j] = x_host->val[v];
            }
            start = end;
        }
#endif
    }
    int j = 0, ntr;
    for (int t = 0; t < nprocs; t++) {
        if (t == myid) {
            continue;
        }
        if (hi.to_receive_counts[t] > 0) {
            CHECK_MPI(
                MPI_Irecv(hi.what_to_receive + (hi.to_receive_spls[t]), hi.to_receive_counts[t], VTYPE_MPI, t, SYNCSOL_TAG, MPI_COMM_WORLD, requests + j));
            j++;
            if (j == MAXNTASKS) {
                fprintf(stderr, "Too many tasks in halo_sync, max is %d\n", MAXNTASKS);
                exit(1);
            }
        }
    }
    ntr = j;
    if (hi.to_send_n) {
        cudaStreamSynchronize(sync_stream);
    }

    for (int t = 0; t < nprocs; t++) {
        if (t == myid) {
            continue;
        }
        if (hi.to_send_counts[t] > 0) {
            CHECK_MPI(MPI_Send(hi.what_to_send + (hi.to_send_spls[t]), hi.to_send_counts[t], VTYPE_MPI, t, SYNCSOL_TAG, MPI_COMM_WORLD));
        }
    }

    if (!hi.to_receive_n) {
        return;
    }
    if (ntr > 0) {
        CHECK_MPI(MPI_Waitall(ntr, requests, MPI_STATUSES_IGNORE));
    }

#if SMART_AGGREGATE_GETSET_GPU == 1
    CHECK_DEVICE(cudaMemcpy(hi.what_to_receive_d, hi.what_to_receive, hi.to_receive_n * sizeof(vtype), cudaMemcpyHostToDevice));

    gb = gb1d(hi.to_receive_n, BLOCKSIZE);
    if (local_flag) {
        if (x->n == A->full_n) {
            setReceivedWithMask<<<gb.g, gb.b>>>(hi.to_receive_n, x->val, hi.what_to_receive_d, hi.to_receive_d->val, A->row_shift);
        }
    } else {
        setReceivedWithMask<<<gb.g, gb.b>>>(hi.to_receive_n, x->val, hi.what_to_receive_d, hi.to_receive_d->val, A->row_shift);
    }
#else
    if (x->n == A->full_n) { // PICO
        vector<vtype>* x_host = Vector::copyToHost(x);
        int start = 0;
        for (int i = 0; i < nprocs; i++) {
            int end = start + hi.to_receive_counts[i];
            for (int j = start; j < end; j++) {
                gstype v = hi.to_receive->val[j];
                x_host->val[v] = hi.what_to_receive[j];
            }
            start = end;
        }
        CHECK_DEVICE(cudaMemcpy(x->val, x_host->val, x_host->n * sizeof(vtype), cudaMemcpyHostToDevice));
        Vector::free(x_host);
    }
#endif
}

bool checkSync(CSR* _A, vector<vtype>* _x0, vector<vtype>* _x1, int level)
{
    _MPI_ENV;
    CSR* A = CSRm::copyToHost(_A);
    vector<vtype>* x0 = Vector::copyToHost(_x0);
    vector<vtype>* x1 = Vector::copyToHost(_x1);

    bool flag = true;
    for (int i = 0; i < A->n; i++) {
        for (int j = A->row[i]; j < A->row[i + 1]; j++) {
            itype col = A->col[j];
            if (x0->val[col] != x1->val[col]) {
                printf("n %d] %d} -- col: %d | ", myid, level, col);
                std::cout << x0->val[col] << " ---- " << x1->val[col] << "\n";
                flag = false;
            }
        }
    }

    CSRm::free(A);
    Vector::free(x0);
    Vector::free(x1);

    return flag;
}

void halo_sync_stream(halo_info hi, CSR* A, vector<vtype>* x, cudaStream_t stream = 0, bool local_flag = false)
{
    _MPI_ENV;

    assert(A->on_the_device);
    assert(x->on_the_device);

    GridBlock gb;

    if (hi.to_send_n) {
        GridBlock gb = gb1d(hi.to_send_n, BLOCKSIZE);

        _getToSend<<<gb.g, gb.b, 0, stream>>>(hi.to_send_d->n, x->val, hi.what_to_send_d, hi.to_send_d->val, A->row_shift);
        CHECK_DEVICE(cudaMemcpyAsync(hi.what_to_send, hi.what_to_send_d, hi.to_send_n * sizeof(vtype), cudaMemcpyDeviceToHost, stream));
    }

    cudaStreamSynchronize(stream);

    CHECK_MPI(
        MPI_Alltoallv(
            hi.what_to_send,
            hi.to_send_counts,
            hi.to_send_spls,
            VTYPE_MPI,
            hi.what_to_receive,
            hi.to_receive_counts,
            hi.to_receive_spls,
            VTYPE_MPI,
            MPI_COMM_WORLD));

    if (!hi.to_receive_n) {
        return;
    }

    CHECK_DEVICE(cudaMemcpyAsync(hi.what_to_receive_d, hi.what_to_receive, hi.to_receive_n * sizeof(vtype), cudaMemcpyHostToDevice, stream));
    if (x->n == A->full_n) { // PICO
        gb = gb1d(hi.to_receive_n, BLOCKSIZE);
        setReceivedWithMask<<<gb.g, gb.b, 0, stream>>>(hi.to_receive_n, x->val, hi.what_to_receive_d, hi.to_receive_d->val, A->row_shift);
    }
}
