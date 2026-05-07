#include "halo_communication/halo_communication.h"
#include "halo_communication/local_permutation.h"

#include "op/getmct.h"
#include "utility/bswhichprocess.h"
#include "utility/cudamacro.h"
#include "utility/memory.h"
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

#define USE_GET_MISSING_NEW 1
#if USE_GET_MISSING_NEW
// #ifdef DEBUG
// #undef DEBUG
// #endif
// #define DEBUG 1
void getMissing(CSR* A, vector<itype>** missing, CSR* R, gstype* row_shift)
{
    _MPI_ENV;

    stype col_ns[nprocs];
    gstype ends[nprocs];

    TRACE("1");

    stype nr_of_cols;
    if (R == NULL) {
        if (A->full_n == A->m) {
            nr_of_cols = A->n;
        } else {
            nr_of_cols = A->m / nprocs;
            if (myid == nprocs - 1) {
                nr_of_cols += A->m % nprocs;
            }
        }
        // if (A->n > nr_of_cols) {
        //     nr_of_cols = A->n;
        // }
    } else {
        nr_of_cols = R->n;
    }

    TRACE("2");

    CHECK_MPI(
        MPI_Allgather(
            &nr_of_cols,
            sizeof(stype),
            MPI_BYTE,
            col_ns,
            sizeof(stype),
            MPI_BYTE,
            MPI_COMM_WORLD));

    TRACE("3");

    

    TRACE("4");

    ends[0] = col_ns[0];
    for (itype i = 1; i < nprocs; i++) {
        ends[i] = col_ns[taskmap[i]] + ends[i - 1];
    }

    TRACE("5");

    if (R == NULL) {
        // assert(ends[nprocs - 1] == A->full_n);
        ASSERT(ends[nprocs - 1] == A->m);
    } else {
        ASSERT(ends[nprocs - 1] == R->full_n);
    }

    // -----------------------------

    TRACE("6");

    int uvs;
    long* ptr;
    if (R != NULL) {
        // Fix Giacomo 2026-04-15: use A->col8 / A->nnz (the matrix whose halo we are
        // setting up, e.g. newP) instead of R->col8 / R->nnz (the coarse matrix passed
        // only for its partitioning info: R->n = exact local coarse count, R->row_shift =
        // exact coarse offset). Using R->col8 scanned the wrong column array and could
        // map a column back to taskmap[J] == myid, causing a NULL dereference on
        // missing[myid]->val (SIGSEGV at address 0x8) on non-master processes.
        ptr = getmct(A->col8, A->nnz, 0, R->n - 1, &uvs, &(A->bitcol), &(A->bitcolsize), &(A->nonuniquesize), NUM_THR);
    } else {
        #if DEBUG
        if (log_file) {
            fprintf(log_file, "----------------------------------------\n", nr_of_cols);
            debugArray("col8[%d]=%d\n", A->col8, A->nnz, true, log_file);
            fprintf(log_file, "nr_of_cols = %d\n", nr_of_cols);
            fprintf(log_file, "tot_cols = %d\n", A->m + A->col_shifted);
            // debugArray("ptr[%d]=%d\n", ptr, uvs, true, log_file);
        }
        #endif
        ptr = getmct(A->col8, A->nnz, 0, nr_of_cols - 1, &uvs, &(A->bitcol), &(A->bitcolsize), &(A->nonuniquesize), NUM_THR);
        #if DEBUG
        if (log_file) {
            fprintf(log_file, "uvs = %d\n", uvs);
            debugArray("ptr[%d] = %d\n", ptr, uvs, false, log_file);
            fprintf(log_file, "----------------------------------------\n", nr_of_cols);
        }
        #endif
        

        // long* getmct2(gsstype*, itype, itype, itype, int*, long**, int*, int);
        // gsstype aux = (A->m / nprocs) * (taskmap ? taskmap[myid] : myid) - (gsstype)A->row_shift;
        // ptr = getmct2(A->col8, A->nnz, -aux, nr_of_cols - 1 - aux, &uvs, &(A->bitcol), &(A->bitcolsize), NUM_THR);
    }

    TRACE("7");

    vector<gsstype>* _bitcol;
    if (uvs == 0) {
        _bitcol = Vector::init<gsstype>(1, true, false);
    } else {
        _bitcol = Vector::init<gsstype>(uvs, false, false);
        _bitcol->val = ptr;
    }

    // #if DEBUG
    // if (log_file) {
    //     fprintf(log_file, "uvs = %d\n", uvs);
    //     debugArray("bitcol[%d] = %d\n", _bitcol->val, uvs, _bitcol->on_the_device, log_file);
    // }
    // #endif

    gstype mypfirstrow = R == NULL ? A->row_shift : R->row_shift;
    gstype myrealfirstrow = R == NULL ? A->row_shift : R->row_shift;
    // gstype mypfirstrow;
    // if (R == NULL) {
    //     mypfirstrow = (A->m / nprocs) * (taskmap ? taskmap[myid] : myid);
    // } else {
    //     mypfirstrow = R->row_shift;
    // }

    CHECK_MPI(
        MPI_Allgather(
            &mypfirstrow,
            sizeof(gstype),
            MPI_BYTE,
            row_shift,
            sizeof(gstype),
            MPI_BYTE,
            MPI_COMM_WORLD));

    for (int i = 0; i < nprocs; i++) {
        if (i != myid) {
            #if DEBUG
            if (log_file) {
                fprintf(log_file, "initial missing[%d]->n = %d\n", i, missing[i]->n);
            }
            #endif
            missing[i]->n = 0;
        }
    }

    // int proc_to_trace = 5;
    TRACE("8");

    #if DEBUG
    if (log_file) {
        // fprintf(log_file, "uvs = %d\n", uvs);
        fprintf(log_file, "myrealfirstrow = %d\n", myrealfirstrow);
        // debugArray("bitcol[%d] = %d\n", _bitcol->val, _bitcol->n, _bitcol->on_the_device, log_file);
    }
    #endif

    if (uvs > 0) {
        stype J = 0, I = 0;
        for (itype i = 0; i < uvs; i++) {

            #if DEBUG
            if (log_file) {
                fprintf(log_file, "\ni = %d\n", i);
            }
            #endif

            ASSERT(i < _bitcol->n);
            gsstype j = _bitcol->val[i];


            #if DEBUG
            if (log_file) {
                fprintf(log_file, "j = %d\n", j);
            }
            #endif

            ASSERT((I + 1) < ((((long)(A->nnz) * (long)scalennzmiss)) / ((long)nprocs)));

        CHECK_AGAIN:
            
            #if DEBUG
            if (log_file) {
                fprintf(log_file, "\ntaskmap = %p, missing = %p\n", taskmap, missing);
                fprintf(log_file, "J = %d, tarkmap[J] = %d\n", J, taskmap[J]);
                fprintf(log_file, "missing[taskmap[J]] = %p\n", missing[taskmap[J]]);
                fprintf(log_file, "j = %d, mypfirstrow = %d, ends[J] = %d\n", j, mypfirstrow, ends[J]);
                fprintf(log_file, "I = %d\n", I);
            }
            #endif

            if ((j + myrealfirstrow) >= ends[J]) {
                if (taskmap[J] != myid) {
                    missing[taskmap[J]]->n = I;
                }

                // #if DEBUG
                // if (log_file) {
                //     fprintf(log_file, "8.3.2\n");
                // }
                // #endif

                J++;
                I = 0;

                if (J >= nprocs) {
                    fprintf(stderr, "[getMissing] Task id %d: unexpected missing source :%d (%p)\n", myid, J, A);
                    fprintf(stderr, "Task=%d,R=%p,missing %lu+%lu\n", myid, R, j, mypfirstrow);
                    for (int k = 0; k < nprocs; k++) {
                        fprintf(stderr, "Task=%d, ends[%d]=%lu\n", myid, k, ends[k]);
                    }
                    char WrongMatName[256];
                    sprintf(WrongMatName, "%s_%s", (R == NULL) ? "WAn" : "WRn", idstring);
                    CSRm::printMM((R == NULL) ? A : R, WrongMatName);
                    DIE("");
                }

                goto CHECK_AGAIN;
            }

            #if DEBUG
            if (log_file) {
                fprintf(log_file, "Out of CHECK_AGAIN\n");
            }
            #endif

            if (J >= nprocs) {
                fprintf(stderr, "Task id %d: unexpected missing source :%d (%p)\n", myid, J, A);
                fprintf(stderr, "Task=%d,R=%p,missing %lu+%lu\n", myid, R, j, mypfirstrow);
                for (int k = 0; k < nprocs; k++) {
                    fprintf(stderr, "Task=%d, ends[%d]=%lu\n", myid, k, ends[k]);
                }
                char WrongMatName[256];
                sprintf(WrongMatName, "%s_%s", (R == NULL) ? "WA" : "WR", idstring);
                CSRm::printMM((R == NULL) ? A : R, WrongMatName);
                DIE("");
            }
            
            #if DEBUG
            if (log_file) {
                fprintf(log_file, "8.5\n");
                fprintf(log_file, "J = %d, tarkmap[J] = %d\n", J, taskmap[J]);
                fprintf(log_file, "row_shift[taskmap[J]] = %d\n", row_shift[taskmap[J]]);
                fprintf(log_file, "I = %d, missing[taskmap[J]] = %p\n", I, missing[taskmap[J]]);
                fprintf(log_file, "missing[taskmap[J]]->val[I] = %d\n", missing[taskmap[J]]->val[I]);
            }
            #endif


            missing[taskmap[J]]->val[I] = j + (myrealfirstrow - row_shift[taskmap[J]]);
            
            #if DEBUG
            if (log_file) {
                fprintf(log_file, "8.6\n");
            }
            #endif
            
            I++;
        }
        if (I) {
            #if DEBUG
            if (log_file) {
                fprintf(log_file, "8.7\n");
            }
            #endif
            missing[taskmap[J]]->n = I;
            #if DEBUG
            if (log_file) {
                fprintf(log_file, "8.8\n");
            }
            #endif
        }
    }

    #if DEBUG
    if (log_file) {
        fprintf(log_file, "9\n");
    }
    #endif
}
#else
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

    // -----------------------------
    gstype mypfirstrow;
    mypfirstrow = R == NULL ? A->row_shift : R->row_shift;

    int uvs;
    // long* getmct(gsstype*, itype, itype, itype, int*, long**, int*, int); // FIX
    long* ptr;
    if (R != NULL) {
        ptr = getmct(R->col8, R->nnz, 0, R->n - 1, &uvs, &(R->bitcol), &(R->bitcolsize), NUM_THR);
    } else {
        ptr = getmct(A->col8, A->nnz, 0, A->n - 1, &uvs, &(A->bitcol), &(A->bitcolsize), NUM_THR);
    }

    vector<gsstype>* _bitcol;
    if (uvs == 0) {
        _bitcol = Vector::init<gsstype>(1, true, false);
    } else {
        _bitcol = Vector::init<gsstype>(uvs, false, false);
        _bitcol->val = ptr;
    }

    for (int i = 0; i < nprocs; i++) {
        if (i != myid) {
            missing[i]->n = 0;
        }
    }

    if (uvs > 0) {
        mypfirstrow = (R == NULL) ? A->row_shift : R->row_shift;
        stype J = 0, I = 0;
        for (itype i = 0; i < uvs; i++) {
            gsstype j = _bitcol->val[i];
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
                fprintf(stderr, "Task id %d: unexpected missing source :%d (%x)\n", myid, J, A);
                fprintf(stderr, "Task=%d,R=%x,missing %d+%lu\n", myid, R, j, mypfirstrow);
                for (int k = 0; k < nprocs; k++) {
                    fprintf(stderr, "Task=%d, ends[%d]=%lu\n", myid, k, ends[k]);
                }
                char WrongMatName[256];
                sprintf(WrongMatName, "%s_%s", (R == NULL) ? "WA" : "WR", idstring);
                CSRm::printMM((R == NULL) ? A : R, WrongMatName);
                DIE("");
            }
            missing[taskmap[J]]->val[I] = j + (mypfirstrow - row_shift[taskmap[J]]);
            I++;
        }
        if (I) {
            missing[taskmap[J]]->n = I;
        }
    }
}
#endif

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
    _MPI_ENV;

    // if (A->full_n < A->m) {
    //     if (ISMASTER) {
    //         const char *mat_name = strlen(A->name) ? A->name : (
    //             (A->full_n == A->m) ? "Aded" :
    //             (A->full_n > A->m ? "Pded" : "Rded"));
    //         fprintf(stderr, "%s = %p:\n", mat_name, A);
    //     }
    //     CSRm::debug(A, stderr);
    // }

    vector<itype>** my_missing; // FIX

    int* sendcounts = MALLOC(int, nprocs);
    int* sdispls = MALLOC(int, nprocs);
    int* recvcounts = MALLOC(int, nprocs);
    int* rdispls = MALLOC(int, nprocs);

    vector<itype>* my_missing_flat = NULL; // FIX
    vector<itype>* their_missing_flat = NULL; // FIX
    itype their_missing_flat_total_n = 0;
    itype total_n = 0;
    gstype row_shift[nprocs];
    halo_info hi;

    if (1 || (A->rows_to_get == NULL || R != NULL)) {
        my_missing = MALLOC(vector<itype>*, nprocs); // FIX

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
            
            // if (log_file && A->full_n < A->m) {
            //     const char *mat_name = strlen(A->name) ? A->name : (
            //     (A->full_n == A->m) ? "Aded" :
            //     (A->full_n > A->m ? "Pded" : "Rded"));
            //     fprintf(log_file, "myid = %d, %s = %p\n"
            //                       "my_missing[%d]\n"
            //                       "==============\n", 
            //                       myid, mat_name, A,
            //                       i);
            //     debugArray("my_missing[%d]=%d\n", my_missing[i]->val, my_missing[i]->n, my_missing[i]->on_the_device, log_file,
            //     [&](long a){ return a + row_shift[i]; });
            // }
        }

        // if (A->name) fprintf(stderr, "myid = %d, A->name = %s, total_n = %d\n", myid, A->name, total_n);

        // if (log_file) {
        //     const char *mat_name = strlen(A->name) ? A->name : (
        //         (A->full_n == A->m) ? "Aded" :
        //         (A->full_n > A->m ? "Pded" : "Rded"));
        //     fprintf(log_file,
        //         "myid = %d, %s = %p, full_n = %d, m = %d, shrinked_m = %d, n = %d, row_shift = %d, total_n = %d\n",
        //         myid, mat_name, A, A->full_n, A->m, A->shrinked_m, A->n, A->row_shift, total_n
        //     );
        //     fprintf(log_file, "\n");
        // }

        if (total_n > 0) {
            my_missing_flat = Vector::init<itype>(total_n, true, false); // FIX
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
                memcpy(my_missing_flat->val + shift, my_missing[i]->val, my_missing[i]->n * sizeof(itype)); // FIX
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
            their_missing_flat = Vector::init<itype>(their_missing_flat_total_n, true, false); // FIX
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
        hi.what_to_receive = CUDA_MALLOC_HOST(vtype, hi.to_receive_n);
        hi.what_to_receive_d = CUDA_MALLOC(vtype, hi.to_receive_n);
        hi.to_receive_d = Vector::copyToDevice(hi.to_receive);
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
        hi.what_to_send = CUDA_MALLOC_HOST(vtype, hi.to_send_n);
        hi.what_to_send_d = CUDA_MALLOC(vtype, hi.to_send_n);
        hi.to_send_d = Vector::copyToDevice(hi.to_send);
    }

    if (1 || (A->rows_to_get == NULL || R != NULL)) {
        for (itype i = 0; i < nprocs; i++) {
            if (my_missing[i] != NULL) {
                Vector::free(my_missing[i]);
            }
        }
        FREE(my_missing);
        if (my_missing_flat != NULL) {
            Vector::free(my_missing_flat);
        }
    }
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
    what_to_send[i] = x[j /* - shift */];
}

__global__ void setReceivedWithMask(itype n, vtype* x, vtype* received, gstype* receive_map, itype shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    itype j = receive_map[i];
    vtype val = received[i];
    x[j /* +shift */] = val;
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
                DIE("Too many tasks in halo_sync, max is %d\n", MAXNTASKS);
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
