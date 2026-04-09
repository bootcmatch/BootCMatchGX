#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "utility/cudamacro.h"
#include "utility/memory.h"
#include "utility/mpi.h"

#include <cuda_runtime.h>

extern int *taskmap, *itaskmap;

template <typename T>
__inline__ void chop_array_MPI_same(int nprocs, unsigned n, unsigned* chunks, unsigned* chunkn)
{
    int i;
    int e4chunk = n / nprocs * sizeof(T);
    for (i = 0; i < nprocs - 1; i++) {
        chunkn[i] = e4chunk;
        chunks[i] = (i)*e4chunk;
    }
    chunkn[nprocs - 1] = (n * sizeof(T)) - i * e4chunk;
    chunks[i] = e4chunk * (nprocs - 1);
}

template <typename T>
__inline__ void chop_array_MPI_old(int nprocs, int n, int n_local, int* chunks, int* chunkn)
{
    itype ns[nprocs];
    itype tmpns[nprocs], tmpchunks[nprocs];

    // std::cerr << "Before MPI_Allgather\n";
    CHECK_MPI(
        MPI_Allgather(
            &n_local, sizeof(itype), MPI_BYTE,
            tmpns, sizeof(itype), MPI_BYTE,
            MPI_COMM_WORLD));
    // std::cerr << "After MPI_Allgather\n";

    // std::cerr << "1\n";
    int i;
    for (i = 0; i < nprocs - 1; i++) {
        ns[i] = tmpns[i];
        chunkn[i] = ns[i] * sizeof(T);
    }

    // std::cerr << "2\n";
    itype tot = 0;
    for (i = 0; i < nprocs - 1; i++) {
        tmpchunks[i] = tot;
        tot += (tmpns[taskmap ? itaskmap[i] : i] * sizeof(T));
    }

    // std::cerr << "3\n";
    for (i = 0; i < nprocs - 1; i++) {
        chunks[i] = tmpchunks[taskmap ? itaskmap[i] : i];
    }

    // std::cerr << "4\n";
    chunkn[nprocs - 1] = (n * sizeof(T)) - tot;
    chunks[i] = tot;
}

vector<vtype>* aggregate_vector_all_old(vector<vtype>* u_local, itype full_n)
{
    _MPI_ENV;

    ASSERT(u_local);

    if (full_n == 0) {
        CHECK_MPI(MPI_Allreduce(&u_local->n, &full_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    }

    vector<vtype>* h_u_local = u_local->on_the_device
        ? Vector::copyToHost(u_local)
        : u_local;
    vector<vtype>* h_u = Vector::init<vtype>(full_n, true, false);

    // std::cerr << "Before chop_array_MPI\n";
    int chunks[nprocs], chunkn[nprocs];
    chop_array_MPI_old<vtype>(nprocs, full_n, u_local->n, chunks, chunkn);

    CHECK_MPI(
        MPI_Allgatherv(
            h_u_local->val,
            u_local->n * sizeof(vtype),
            MPI_BYTE,
            h_u->val,
            chunkn,
            chunks,
            MPI_BYTE,
            MPI_COMM_WORLD));

    if (u_local->on_the_device) {
        Vector::free(h_u_local);
    }

    return h_u;
}

template <typename T>
__inline__ void chop_array_MPI(int nprocs, int n_global, int n_local, int* chunks, int* chunkn)
{
    // 1. Ogni processo comunica agli altri quanto è grande il suo pezzo locale
    int local_ns[nprocs];
    CHECK_MPI(MPI_Allgather(
        &n_local, 1, MPI_INT,
        local_ns, 1, MPI_INT,
        MPI_COMM_WORLD
    ));

    // 2. Calcoliamo gli offset basandoci sulla mappa logica (taskmap)
    // taskmap[i] ci dice quale rank fisico possiede il blocco i-esimo della matrice
    int current_byte_offset = 0;

    for (int i = 0; i < nprocs; i++) {
        int physical_rank = taskmap[i]; // Chi ha il pezzo i?
        
        // Offset e dimensione in byte per la Allgatherv
        chunks[physical_rank] = current_byte_offset;
        chunkn[physical_rank] = local_ns[physical_rank] * sizeof(T);
        
        // Incrementiamo l'offset globale
        current_byte_offset += chunkn[physical_rank];
    }
    
    // NOTA: Non serve calcolare il resto a mano per l'ultimo processo,
    // perché local_ns[physical_rank] contiene già il valore corretto (incluso il resto)
    // calcolato nel main per quel specifico processo.
}

vector<vtype>* aggregate_vector_all(vector<vtype>* u_local, itype full_n)
{
    _MPI_ENV;
    ASSERT(u_local);

    // Se full_n non è noto, lo recuperiamo
    if (full_n == 0) {
        int local_n = (int)u_local->n;
        int total_n_tmp = 0;
        CHECK_MPI(MPI_Allreduce(&local_n, &total_n_tmp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
        full_n = (itype)total_n_tmp;
    }

    // Gestione memoria device/host
    vector<vtype>* h_u_local = u_local->on_the_device ? Vector::copyToHost(u_local) : u_local;
    
    // Inizializziamo il vettore globale (il 'true' esegue il memset a zero)
    vector<vtype>* h_u = Vector::init<vtype>(full_n, true, false);

    int chunks[nprocs];
    int chunkn[nprocs];

    // Chiamiamo la nuova chop_array che popola correttamente chunks e chunkn in byte
    chop_array_MPI<vtype>(nprocs, (int)full_n, (int)u_local->n, chunks, chunkn);

    // Eseguiamo la raccolta globale
    CHECK_MPI(
        MPI_Allgatherv(
            h_u_local->val,             // Buffer locale
            u_local->n * sizeof(vtype), // Byte da inviare
            MPI_BYTE,
            h_u->val,                   // Buffer globale
            chunkn,                     // Array dei byte da ricevere per ogni rank
            chunks,                     // Array degli offset in byte
            MPI_BYTE,
            MPI_COMM_WORLD));

    if (u_local->on_the_device) {
        Vector::free(h_u_local);
    }

    return h_u;
}

__global__ void _split_local(itype nstart, itype nrow, itype* Arow, vtype* Aval, itype* Acol, itype* Alocal_row, vtype* Alocal_val, itype* Alocal_col, itype* nnz)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nrow) {
        return;
    }

    itype shift = Arow[nstart];
    itype is = i + nstart;
    itype j_start = Arow[is];
    itype j_stop = Arow[is + 1];

    int j;
    Alocal_row[i] = Arow[is] - shift;
    for (j = j_start; j < j_stop; j++) {
        Alocal_val[j - shift] = Aval[j];
        Alocal_col[j - shift] = Acol[j];
    }

    if (i == nrow - 1) {
        *nnz = Arow[nrow + nstart] - shift;
        Alocal_row[nrow] = Arow[is + 1] - shift;
    }
}

CSR* split_local(CSR* A)
{
    _MPI_ENV;
    ASSERT(A->on_the_device && A->n == A->full_n);

    itype rowsxproc = 0;
    // Split A

    int nrows[nprocs];
    rowsxproc = A->n / nprocs;
    for (itype i = 0; i < nprocs - 1; i++) {
        nrows[i] = rowsxproc;
    }
    nrows[nprocs - 1] = A->n - (rowsxproc * (nprocs - 1));

    int nstart = 0;
    for (int j = 0; j < myid; j++) {
        nstart += nrows[j];
    }

    CSR* Alocal = CSRm::init(nrows[myid], A->m, A->nnz, true, true, false, A->n, nstart);

    scalar<itype>* nnz = Scalar::init<itype>(-1, true);

    GridBlock gb = gb1d(nrows[myid], BLOCKSIZE);
    _split_local<<<gb.g, gb.b>>>(nstart, nrows[myid], A->row, A->val, A->col, Alocal->row, Alocal->val, Alocal->col, nnz->val);

    int* h_nnz = Scalar::getValueFromDevice(nnz);
    Scalar::free(nnz);

    Alocal->nnz = *h_nnz;

    return Alocal;
}

CSR* split_matrix_mpi_host(CSR* A)
{
    _MPI_ENV;

    gstype colxproc[nprocs];
    stype rowsxproc = 0;

    if (ISMASTER) {
        ASSERT(!A->on_the_device);
        // Split A
        rowsxproc = A->full_n / nprocs;
        for (itype i = 1; i < nprocs; i++) {
            colxproc[i - 1] = A->row[i * rowsxproc] - A->row[(i - 1) * rowsxproc];
        }
        colxproc[nprocs - 1] = A->row[A->full_n] - A->row[(nprocs - 1) * rowsxproc];
    }

    gstype n, m;
    if (ISMASTER) {
        n = A->n;
        m = A->m;
    }

    CHECK_MPI(
        MPI_Bcast(&n, sizeof(gstype), MPI_BYTE, 0, MPI_COMM_WORLD));

    CHECK_MPI(
        MPI_Bcast(&m, sizeof(gstype), MPI_BYTE, 0, MPI_COMM_WORLD));

    if ((nprocs > 1) && myid == (nprocs - 1)) {
        // compute the number of rows for the last process
        rowsxproc = n - ((n / nprocs) * (nprocs - 1));
    } else {
        // compute the number of rows for the process
        rowsxproc = n / nprocs;
    }

    gstype mycol = 0;
    // send columns numbers to each process
    CHECK_MPI(
        MPI_Scatter(
            colxproc,
            sizeof(gstype),
            MPI_BYTE,
            &mycol,
            sizeof(gstype),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD));

    stype chunks[nprocs], chunkn[nprocs];
    chop_array_MPI_same<stype>(nprocs, (unsigned)n, chunks, chunkn);
    stype rows_shift = chunks[myid] / sizeof(stype);

    CSR* Alocal = CSRm::init(rowsxproc, m, (stype)mycol, true, false, false, n, rows_shift);

    // get row pointers
    CHECK_MPI(
        MPI_Scatterv(
            myid ? NULL : A->row,
            (int*)chunkn,
            (int*)chunks,
            MPI_BYTE,
            Alocal->row,
            sizeof(itype) * rowsxproc,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD));
    // set the last pointer in the row array
    Alocal->row[rowsxproc] = Alocal->row[0] + mycol;

    // get columns
    for (int i = 0; i < nprocs; i++) {
        chunkn[i] = (int)(colxproc[i] * sizeof(itype));
        chunks[i] = ((i == 0) ? 0 : (chunks[i - 1] + chunkn[i - 1]));
    }

    CHECK_MPI(
        MPI_Scatterv(
            myid ? NULL : A->col,
            (int*)chunkn,
            (int*)chunks,
            MPI_BYTE,
            Alocal->col,
            sizeof(itype) * mycol,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD));

    // get values
    for (int i = 0; i < nprocs; i++) {
        chunkn[i] = (int)(colxproc[i] * sizeof(vtype));
        chunks[i] = ((i == 0) ? 0 : (chunks[i - 1] + chunkn[i - 1]));
    }
    CHECK_MPI(
        MPI_Scatterv(
            myid ? NULL : A->val,
            (int*)chunkn,
            (int*)chunks,
            MPI_BYTE,
            Alocal->val,
            sizeof(vtype) * mycol,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD));

    // shift row pointers
    if (myid > 0) {
        itype shift = Alocal->row[0];
        for (int i = 0; i <= Alocal->n; i++) {
            Alocal->row[i] -= shift;
        }
    }
    return Alocal;
}

CSR* split_matrix_mpi(CSR* A)
{
    CSR* Alocal = split_matrix_mpi_host(A);
    CSR* d_Alocal = CSRm::copyToDevice(Alocal);
    CSRm::free(Alocal);
    return d_Alocal;
}

CSR* join_matrix_mpi(CSR* Alocal)
{
    _MPI_ENV;

    ASSERT(nprocs > 1);
    ASSERT(!Alocal->on_the_device);

    // TRACE("Alocal->n");
    itype row_ns[nprocs];
    CHECK_MPI(MPI_Allgather(&Alocal->n, sizeof(itype), MPI_BYTE,
                            row_ns, sizeof(itype), MPI_BYTE,
                            MPI_COMM_WORLD));

    // TRACE("Alocal->nnz");
    itype nnzs[nprocs];
    CHECK_MPI(MPI_Allgather(&Alocal->nnz, sizeof(itype), MPI_BYTE,
                            nnzs, sizeof(itype), MPI_BYTE,
                            MPI_COMM_WORLD));

    // TRACE("Alocal->col_shifted");
    gsstype col_shifted[nprocs];
    CHECK_MPI(MPI_Allgather(&Alocal->col_shifted, sizeof(gsstype), MPI_BYTE,
                            col_shifted, sizeof(gsstype), MPI_BYTE,
                            MPI_COMM_WORLD));

    itype full_n = 0, full_nnz = 0;
    CSR* A = NULL;
    int chunkn[nprocs], chunks[nprocs];

    if (ISMASTER) {
        // debugArray("row_ns[%d]=%d\n", row_ns, nprocs, false, stderr);
        // debugArray("nnzs[%d]=%d\n", nnzs, nprocs, false, stderr);
        // debugArray("col_shifted[%d]=%d\n", col_shifted, nprocs, false, stderr);


        for (int i = 0; i < nprocs; i++) {
            full_n += row_ns[i];
            full_nnz += nnzs[i];
        }

        ASSERT(full_n == Alocal->full_n);
        A = CSRm::init(full_n, Alocal->m, full_nnz,
                       true, false, false, full_n, 0);

        for (int ii = 0; ii < nprocs; ii++) {
            int i = get_mapped_task(ii);
            chunkn[i] = row_ns[i] * sizeof(itype);
            chunks[i] = (i == 0) ? 0 : chunks[get_mapped_task(ii - 1)] + chunkn[get_mapped_task(ii - 1)];
        }
        chunkn[nprocs - 1] += sizeof(itype); // account for +1 row entry

        // debugArray("chunkn[%d]=%d\n", chunkn, nprocs, false, stderr);
        // debugArray("chunks[%d]=%d\n", chunks, nprocs, false, stderr);
    }

    itype rn = Alocal->n * sizeof(itype);
    if (myid == nprocs - 1) rn += sizeof(itype);

    // TRACE("Alocal->row");
    CHECK_MPI(MPI_Gatherv(Alocal->row, rn, MPI_BYTE,
                          ISMASTER ? A->row : NULL,
                          chunkn, chunks, MPI_BYTE,
                          0, MPI_COMM_WORLD));

    if (ISMASTER) {
        // debugArray("A->row1[%d]=%d\n", A->row, A->n + 1, false, stderr);

        itype rowoffset = 0;
        int pos = 0;
        for (int ii = 0; ii < nprocs; ii++) {
            int i = get_mapped_task(ii);
            for (int j = 0; j < row_ns[i]; j++) {
                A->row[pos + j] += rowoffset;
            }
            pos += row_ns[i];
            rowoffset += nnzs[i];
        }

        // Set last row entry
        A->row[A->full_n] = full_nnz;

        // debugArray("A->row2[%d]=%d\n", A->row, A->n + 1, false, stderr);
    }


    // gather columns
    if (ISMASTER) {
        for (int ii = 0; ii < nprocs; ii++) {
            int i = get_mapped_task(ii);
            chunkn[i] = nnzs[i] * sizeof(itype);
            chunks[i] = (i == 0) ? 0 : chunks[get_mapped_task(ii - 1)] + chunkn[get_mapped_task(ii - 1)];
        }

        // debugArray("chunkn[%d]=%d\n", chunkn, nprocs, false, stderr);
        // debugArray("chunks[%d]=%d\n", chunks, nprocs, false, stderr);
    }

    // TRACE("Alocal->col");
    CHECK_MPI(MPI_Gatherv(Alocal->col, Alocal->nnz * sizeof(itype), MPI_BYTE,
                          ISMASTER ? A->col : NULL,
                          chunkn, chunks, MPI_BYTE,
                          0, MPI_COMM_WORLD));

    // shift column indices
    if (ISMASTER) {
        // debugArray("A->col1[%d]=%d\n", A->col, A->nnz, false, stderr);

        int pos = 0;
        for (int ii = 0; ii < nprocs; ii++) {
            int i = get_mapped_task(ii);
            for (int j = 0; j < nnzs[i]; j++) {
                A->col[pos + j] -= col_shifted[i];
            }
            pos += nnzs[i];
        }

        // debugArray("A->col2[%d]=%d\n", A->col, A->nnz, false, stderr);
    }

    // gather values
    if (ISMASTER) {
        for (int ii = 0; ii < nprocs; ii++) {
            int i = get_mapped_task(ii);
            chunkn[i] = nnzs[i] * sizeof(vtype);
            chunks[i] = (i == 0) ? 0 : chunks[get_mapped_task(ii - 1)] + chunkn[get_mapped_task(ii - 1)];
        }
    }

    // TRACE("Alocal->val");
    CHECK_MPI(MPI_Gatherv(Alocal->val, Alocal->nnz * sizeof(vtype), MPI_BYTE,
                          ISMASTER ? A->val : NULL,
                          chunkn, chunks, MPI_BYTE,
                          0, MPI_COMM_WORLD));

    // TRACE("Done");
    return A;
}


// CSR* join_matrix_mpi_old(CSR* Alocal)
// {
//     _MPI_ENV;

//     ASSERT(nprocs > 1);
//     ASSERT(!Alocal->on_the_device);
//     // ASSERT(Alocal->row_shift == -Alocal->col_shifted);

//     itype row_ns[nprocs];

//     // send rows sizes
//     CHECK_MPI(
//         MPI_Allgather(
//             &Alocal->n,
//             sizeof(itype),
//             MPI_BYTE,
//             row_ns,
//             sizeof(itype),
//             MPI_BYTE,
//             MPI_COMM_WORLD));

//     itype nnzs[nprocs];

//     // send nnz sizes
//     CHECK_MPI(
//         MPI_Allgather(
//             &Alocal->nnz,
//             sizeof(itype),
//             MPI_BYTE,
//             nnzs,
//             sizeof(itype),
//             MPI_BYTE,
//             MPI_COMM_WORLD));

//     gsstype col_shifted[nprocs];
//     CHECK_MPI(
//         MPI_Allgather(
//             &Alocal->col_shifted,
//             sizeof(gsstype),
//             MPI_BYTE,
//             col_shifted,
//             sizeof(gsstype),
//             MPI_BYTE,
//             MPI_COMM_WORLD));

//     itype full_n = 0;
//     itype full_nnz = 0;
//     CSR* A = NULL;
//     int chunkn[nprocs], chunks[nprocs];

//     if (ISMASTER) {

//         for (int i = 0; i < nprocs; i++) {
//             full_n += row_ns[i];
//             full_nnz += nnzs[i];
//             // printf("nnz[%d] = %d\n", i, nnzs[i]);
//             // printf("col_shifted[%d] = %d\n", i, col_shifted[i]);
//         }

//         if (full_n != Alocal->full_n) {
//             TRACE("full_n=%d, A->full_n=%lu", full_n, Alocal->full_n);
//         }
//         ASSERT(full_n == Alocal->full_n);

//         A = CSRm::init(full_n, Alocal->m, full_nnz, true, false, false, full_n, 0);

//         // gather rows
//         for (int i = 0; i < nprocs; i++) {
//             chunkn[i] = row_ns[i] * sizeof(itype);
//             chunks[i] = ((i == 0) ? 0 : (chunks[i - 1] + chunkn[i - 1]));
//         }
//         chunkn[nprocs - 1] += 1 * sizeof(itype);
//     }

//     itype rn = Alocal->n * sizeof(itype);
//     if (myid == nprocs - 1) {
//         rn += 1; // +1 for the last process
//     }

//     CHECK_MPI(
//         MPI_Gatherv(
//             Alocal->row,
//             rn,
//             MPI_BYTE,
//             myid ? NULL : A->row,
//             chunkn,
//             chunks,
//             MPI_BYTE,
//             0,
//             MPI_COMM_WORLD));

//     if (ISMASTER) {
//         /* reset the row number */
//         itype rowoffset = 0;
//         itype th = row_ns[0];
//         int j = 0;
//         for (int i = 0; i < Alocal->full_n; i++) {
//             // next piece
//             if (i >= th && (j < (nprocs))) {
//                 rowoffset += nnzs[j];
//                 j++;
//                 th += row_ns[j];
//             }
//             A->row[i] += rowoffset;
//         }

//         A->row[A->full_n] = nnzs[0];
//         for (int i = 1; i < nprocs; i++) {
//             A->row[A->full_n] += nnzs[i];
//         }
//     }
//     // gather columns
//     for (int i = 0; i < nprocs; i++) {
//         chunkn[i] = nnzs[i] * sizeof(itype);
//         chunks[i] = ((i == 0) ? 0 : (chunks[i - 1] + chunkn[i - 1]));
//     }
//     CHECK_MPI(
//         MPI_Gatherv(
//             Alocal->col,
//             Alocal->nnz * sizeof(itype),
//             MPI_BYTE,
//             myid ? NULL : A->col,
//             chunkn,
//             chunks,
//             MPI_BYTE,
//             0,
//             MPI_COMM_WORLD));

//     // Begin Giacomo 2025-05-21: Fix shift columns
//     if (ISMASTER) {
//         int offset = 0;
//         for (int i = 0; i < nprocs; i++) {
//             for (int j = 0; j < nnzs[i]; j++) {
//                 A->col[offset + j] -= col_shifted[i];
//             }
//             offset += nnzs[i];
//         }
//     }
//     // End Giacomo 2025-05-21: Fix shift columns

//     // gather value
//     for (int i = 0; i < nprocs; i++) {
//         chunkn[i] = nnzs[i] * sizeof(vtype);
//         chunks[i] = ((i == 0) ? 0 : (chunks[i - 1] + chunkn[i - 1]));
//     }
//     CHECK_MPI(
//         MPI_Gatherv(
//             Alocal->val,
//             Alocal->nnz * sizeof(vtype),
//             MPI_BYTE,
//             myid ? NULL : A->val,
//             chunkn,
//             chunks,
//             MPI_BYTE,
//             0,
//             MPI_COMM_WORLD));

//     return A;
// }

int stringCmp(const void* a, const void* b)
{
    return strcmp((const char*)a, (const char*)b);
}

void checkMatrixMPI(CSR* A, bool check_diagonal = true)
{
    _MPI_ENV;
    ASSERT(A->on_the_device);
    CSR* h_Alocal = CSRm::copyToHost(A);
    CSR* h_Afull = join_matrix_mpi(h_Alocal);

    if (ISMASTER) {
        CSRm::checkMatrix(h_Afull, check_diagonal);
    }

    CSRm::free(h_Alocal);
    if (ISMASTER) {
        CSRm::free(h_Afull);
    }
}

bool _check_in_A(CSR* A, int i, int J)
{
    for (int j = A->row[i]; j < A->row[i + 1]; j++) {
        int c = A->col[j];
        if (c == J) {
            return true;
        }
    }
    return false;
}

void check_A_P_MPI(CSR* A_local, CSR* P_)
{
    _MPI_ENV;
    ASSERT(A_local->on_the_device);
    CSR* h_Alocal = CSRm::copyToHost(A_local);
    CSR* A = join_matrix_mpi(h_Alocal);

    if (ISMASTER) {
        CSR* P = CSRm::copyToHost(P_);
        CSRm::checkMatrix(A);

        for (int i = 0; i < P->n; i++) {
            for (int j = P->row[i]; j < P->row[i + 1]; j++) {
                if (!_check_in_A(A, i, P->col[j])) {
                    printf("AP_ERROR %d %d\n", P->col[j], i);
                }
            }
        }
        CSRm::free(A);
        CSRm::free(P);
    }

    CSRm::free(h_Alocal);
}

CSR* broadcast_FullMatrix(CSR* A)
{
    _MPI_ENV;

    if (ISMASTER) {
        ASSERT(!A->on_the_device);
    }

    itype n, m, nnz;
    if (ISMASTER) {
        n = A->n;
        m = A->m;
        nnz = A->nnz;
    }

    CHECK_MPI(
        MPI_Bcast(&n, sizeof(itype), MPI_BYTE, 0, MPI_COMM_WORLD));

    CHECK_MPI(
        MPI_Bcast(&m, sizeof(itype), MPI_BYTE, 0, MPI_COMM_WORLD));

    CHECK_MPI(
        MPI_Bcast(&nnz, sizeof(itype), MPI_BYTE, 0, MPI_COMM_WORLD));

    if (!ISMASTER) {
        A = CSRm::init(n, m, nnz, true, false, false, n, 0);
    }

    CHECK_MPI(
        MPI_Bcast(A->row, sizeof(itype) * (A->n + 1), MPI_BYTE, 0, MPI_COMM_WORLD));

    CHECK_MPI(
        MPI_Bcast(A->col, sizeof(itype) * A->nnz, MPI_BYTE, 0, MPI_COMM_WORLD));

    CHECK_MPI(
        MPI_Bcast(A->val, sizeof(vtype) * A->nnz, MPI_BYTE, 0, MPI_COMM_WORLD));

    CSR* d_A = CSRm::copyToDevice(A);
    CSRm::free(A);

    return d_A;
}
