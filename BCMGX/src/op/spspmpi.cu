#include "spspmpi.h"

#include "datastruct/CSR.h"
#include "halo_communication/local_permutation.h"
#include "op/getmct.h"
#include "utility/bswhichprocess.h"
#include "utility/cudamacro.h"
#include "utility/memory.h"
#include "utility/setting.h"
#include "utility/col8.h"

#include <unistd.h>
#include <unordered_set>

#define BITXBYTE 8

// #include "nsp2.h"
#include "nsparse.h"

SpSpLib SPSP_LIB = SpSpLib::CUSPARSE;

extern int *itaskmap, *taskmap;

itype* iPtemp1 = NULL;
vtype* vPtemp1 = NULL;
itype* iAtemp1 = NULL;
vtype* vAtemp1 = NULL;
itype* idevtemp1 = NULL;
vtype* vdevtemp1 = NULL;
itype* idevtemp2 = NULL;

itype* dev_rcvprow_stat = NULL;
vtype* completedP_stat_val = NULL;
itype* completedP_stat_col = NULL;
gsstype* completedP_stat_col8 = NULL;
itype* completedP_stat_row = NULL;

// =============================================================================

SpSpLib string_to_spsplib(const std::string& str)
{
    if (str == "CUSPARSE") {
        return SpSpLib::CUSPARSE;
    } else if (str == "NSPARSE") {
        return SpSpLib::NSPARSE;
    } else if (str == "NSP") {
        return SpSpLib::NSP;
    // } else if (str == "NSP2") {
    //     return SpSpLib::NSP2;
    } else {
        return SpSpLib::INVALID;
    }
}

std::string spsplib_to_string(const SpSpLib& val)
{
    switch (val) {
    case SpSpLib::CUSPARSE:
        return "CUSPARSE";
    case SpSpLib::NSPARSE:
        return "NSPARSE";
    case SpSpLib::NSP:
        return "NSP";
    // case SpSpLib::NSP2:
    //     return "NSP2";
    case SpSpLib::INVALID:
        return "INVALID";
    default:
        DIE("Unhandled value for enum class SpSpLib\n");
    }
}

// =============================================================================

void check_exact_overflow(
    const int *A_rowptr, const int *A_colidx,
    const int *B_rowptr, const int *B_colidx,
    int n_rows_A,
    int HASH_SIZE)
{
    for (int i = 0; i < n_rows_A; i++) {
        std::unordered_set<int> cols;
        int startA = A_rowptr[i];
        int endA   = A_rowptr[i+1];

        for (int k = startA; k < endA; k++) {
            int rowB = A_colidx[k];
            for (int p = B_rowptr[rowB]; p < B_rowptr[rowB+1]; p++)
                cols.insert(B_colidx[p]);
        }

        if ((int)cols.size() > HASH_SIZE) {
            std::cout << "Overflow certo in riga " << i
                      << " (output=" << cols.size()
                      << ", HASH_SIZE=" << HASH_SIZE << ")\n";
        }
    }
}

/**
 * @brief Computes the number of non-zero elements (NNZ) for each row in a sparse matrix.
 *
 * This kernel calculates the number of non-zero elements for each row in the sparse matrix
 * based on the provided row pointers and stores the result in nnz_to_get_form_prow.
 *
 * @param n The number of rows in the matrix.
 * @param to_get_form_prow Pointer to the array that indicates which rows to get.
 * @param row Pointer to the row pointers of the sparse matrix.
 * @param nnz_to_get_form_prow Pointer to the output array where the NNZ counts will be stored.
 */
__global__ void _getNNZ(itype n, const itype* __restrict__ to_get_form_prow, const itype* __restrict__ row, itype* nnz_to_get_form_prow)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    itype j = to_get_form_prow[i];
    nnz_to_get_form_prow[i] = row[j + 1] - row[j];
}

/**
 * @brief Performs a binary search on a sorted array.
 *
 * This function finds the index of the first element in the array that is greater than
 * the specified value.
 *
 * @param array Pointer to the sorted array.
 * @param size The size of the array.
 * @param value The value to search for.
 * @return int The index of the first element greater than the value.
 */
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

/**
 * @brief Fills the row pointers for a sparse matrix based on received data.
 *
 * This kernel populates the row pointers for a sparse matrix based on the received
 * data and local information.
 *
 * @param n The number of rows in the matrix.
 * @param rows2bereceived The number of rows to be received.
 * @param whichproc Pointer to the array indicating which process owns each row.
 * @param p_nnz_map Pointer to the array mapping rows to their non-zero counts.
 * @param mypfirstrow The first row owned by the current process.
 * @param myplastrow The last row owned by the current process.
 * @param nzz_pre_local The number of non-zero elements before local rows.
 * @param Plocalnnz The number of non-zero elements in the local matrix.
 * @param local_row Pointer to the local row pointers.
 * @param row Pointer to the output row pointers.
 */
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

/**
 * @brief Fills the row pointers for a sparse matrix without communication.
 *
 * This kernel populates the row pointers for a sparse matrix based on local data
 * without any inter-process communication.
 *
 * @param n The number of rows in the matrix.
 * @param mypfirstrow The first row owned by the current process.
 * @param myplastrow The last row owned by the current process.
 * @param Plocalnnz The number of non-zero elements in the local matrix.
 * @param local_row Pointer to the local row pointers.
 * @param row Pointer to the output row pointers.
 */
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

/**
 * @brief Marks columns that are missing from the local process.
 *
 * This kernel updates a mask to indicate which columns are missing from the local
 * process based on the column indices.
 *
 * @param nnz The number of non-zero elements in the matrix.
 * @param mypfirstrow The first row owned by the current process.
 * @param myplastrow The last row owned by the current process.
 * @param col Pointer to the column indices of the non-zero elements.
 * @param mask Pointer to the mask array that will be updated.
 */
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

/**
 * @brief Retrieves column values and indices for a given row.
 *
 * This kernel retrieves the column indices and values for a specified row and
 * stores them in the provided output arrays.
 *
 * @param n The number of rows to process.
 * @param rcvprow Pointer to the array of row indices to receive.
 * @param nnz_per_row Pointer to the array of non-zero counts per row.
 * @param row Pointer to the row pointers of the sparse matrix.
 * @param col Pointer to the column indices of the non-zero elements.
 * @param val Pointer to the values of the non-zero elements.
 * @param col2get Pointer to the output array for column indices.
 * @param val2get Pointer to the output array for values.
 * @param row_shift The shift to apply to the row index.
 */
__global__ void _getColVal(
    itype n,
    itype* rcvprow,
    itype* nnz_per_row,
    const itype* __restrict__ row,
    const itype* __restrict__ col,
    const vtype* __restrict__ val,
    itype* col2get,
    vtype* val2get,
    itype row_shift)
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

/**
 * @brief Merges two sorted arrays into a third array.
 *
 * This function merges two sorted arrays into a single sorted array, removing duplicates.
 *
 * @param a Pointer to the first sorted array.
 * @param b Pointer to the second sorted array.
 * @param c Pointer to the output array where the merged result will be stored.
 * @param n1 * The size of the first array.
 * @param n2 The size of the second array.
 * @return itype The size of the merged array.
 */
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

int dumpP = 0;
extern char idstring[];

/**
 * @brief Main function for sparse matrix multiplication using GPU.
 *
 * This function performs sparse matrix multiplication on the GPU, handling
 * communication between processes and managing memory allocation.
 *
 * @param Alocal Pointer to the local sparse matrix.
 * @param Pfull Pointer to the full sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param used_by_solver A boolean indicating if the result is used by a solver.
 * @return CSR* Pointer to the resulting sparse matrix after multiplication.
 */
CSR* nsparseMGPU(CSR* Alocal, CSR* Pfull, csrlocinfo* Plocal /*, bool used_by_solver*/)
{
    static int count = 0;

    // ASSERT(CSRm::checkUniqueIndeces(Alocal));
    // ASSERT(CSRm::checkUniqueIndeces(Pfull));
    
    _MPI_ENV;

    // fprintf(stderr, "myid = %d, count = %d\n", myid, count);
    // if (count > 21)
    // {
    //     {
    //         char fname[1024] = {0};
    //         snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "Alocal", count, myid, nprocs);
    //         CSRm::printMMsimple(Alocal, fname, false);
    //     }
    //     {
    //         char fname[1024] = {0};
    //         snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "Pfull", count, myid, nprocs);
    //         CSRm::printMMsimple(Pfull, fname, false);
    //     }
    //     // DIE("Forced shutdown\n");
    // }

    #if 0
    ASSERT(Alocal->m == Pfull->n);
    ASSERT(Alocal->nnz <= Alocal->n * Alocal->m);
    ASSERT(Pfull->nnz <= Pfull->n * Pfull->m);

    {
        int m = Alocal->m;
        deviceForEach(Alocal->col, Alocal->nnz, [m]__device__(const int &index, const int &val) {
            if (val < 0 || val >= m) {
                printf("Alocal->col[%d] = %d, not in [%d, %d)\n", index, val, 0, m);
                assert(!(val < 0 || val >= m));
            }
        });
    }

    {
        int m = Pfull->m;
        deviceForEach(Pfull->col, Pfull->nnz, [m]__device__(const int &index, const int &val) {
            if (val < 0 || val >= m) {
                printf("Pfull->col[%d] = %d, not in [%d, %d)\n", index, val, 0, m);
                assert(!(val < 0 || val >= m));
            }
        });
    }

    {
        int *Arow = copyArrayToHost(Alocal->row, Alocal->n + 1);
        int *Acol = copyArrayToHost(Alocal->col, Alocal->nnz);
        int *Prow = copyArrayToHost(Pfull->row, Pfull->n + 1);
        int *Pcol = copyArrayToHost(Pfull->col, Pfull->nnz);
        check_exact_overflow(Arow, Acol, Prow, Pcol, A->n, 512);
        FREE(Arow);
        FREE(Acol);
        FREE(Prow);
        FREE(Pcol);
    }
    #endif

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

    switch (SPSP_LIB) {
        case SpSpLib::CUSPARSE: {
            int spgemmcusparse(int A_num_rows, int A_num_cols, int A_nnz, int *dA_csrOffsets, int *dA_columns, double *dA_values,
                            int B_num_rows, int B_num_cols, int B_nnz, int *dB_csrOffsets, int *dB_columns, double *dB_values,
                            int *p2C_nnz, int **p2dC_csrOffsets, int **p2dC_columns, double **p2dC_values);

            spgemmcusparse(mat_a.M, mat_a.N, mat_a.nnz, mat_a.d_rpt, mat_a.d_col, mat_a.d_val,
                        mat_p.M, mat_p.N, mat_p.nnz, mat_p.d_rpt, mat_p.d_col, mat_p.d_val,
                        &mat_c.nnz, &(mat_c.d_rpt), &(mat_c.d_col), &(mat_c.d_val));

            break;
        }
        case SpSpLib::NSPARSE: {
            spgemm_csrseg_kernel_hash(&mat_a, &mat_p, &mat_c, Plocal, true /* used_by_solver */);
            break;
        }
        case SpSpLib::NSP: {
            void nsp_spgemm_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c);
            nsp_spgemm_kernel_hash(&mat_a, &mat_p, &mat_c);
            break;
        }
        // case SpSpLib::NSP2: {
        //     nsp2_spgemm_kernel_hash(&mat_a, &mat_p, &mat_c, true /* used_by_solver */);
        //     break;
        // }
        default: {
            DIE("Unsupported SpSpLib");
        }
    }

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
    assert(C);

    C->row = mat_c.d_rpt;
    C->col = mat_c.d_col;
    C->val = mat_c.d_val;
    C->custom_alloced = true;

    // debugArray("C->col[%d] = %d\n", C->col, C->nnz, true, stderr);
    if (!C->col8) {
        C->col8 = CUDA_MALLOC(gsstype, C->nnz);
        col2col8(C->col, C->col8, C->nnz);
    }
    assert(C->col8);

    #if 0
    cudaDeviceSynchronize();
    // if (count == 32) {
    if (CSRm::checkUniqueIndeces(Alocal)
            && CSRm::checkUniqueIndeces(Pfull)
            && !CSRm::checkUniqueIndeces(C)) {
        {
            char fname[1024] = {0};
            snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "Alocal", count, myid, nprocs);
            CSRm::printMMsimple(Alocal, fname, false);
        }
        {
            char fname[1024] = {0};
            snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "Pfull", count, myid, nprocs);
            CSRm::printMMsimple(Pfull, fname, false);
        }
        {
            char fname[1024] = {0};
            snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "C", count, myid, nprocs);
            CSRm::printMMsimple(C, fname, false);
        }
        const int NON_UNIQUE_INDECES = 0;
        ASSERT(NON_UNIQUE_INDECES);
    }
    #endif

    count++;

    return C;
}

/**
 * @brief Retrieves missing columns from a sparse matrix.
 *
 * This function identifies and returns the columns that are missing from the local
 * sparse matrix.
 *
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @return vector<int>* Pointer to a vector containing the missing column indices.
 */
vector<gsstype>* get_missing_col(CSR* Alocal, CSR* Plocal)
{
    _MPI_ENV;
    stype myplastrow;
    if (nprocs == 1) {
        vector<gsstype>* _bitcol = Vector::init<long>(1, true, false);
        return _bitcol;
    }

    // GridBlock gb; - FIX

    if (Plocal != NULL) {
        myplastrow = Plocal->n - 1;
    } else {
        myplastrow = Alocal->n - 1;
    } // P_n_per_process[i]: number of rows that process i have of matrix P

    if (Alocal->nnz == 0) {
        return NULL;
    }
    int uvs;
    long* ptr = getmct(Alocal->col8, Alocal->nnz, 0, myplastrow, &uvs, &(Alocal->bitcol), &(Alocal->bitcolsize), &(Alocal->nonuniquesize), NUM_THR); // FIX
    if (uvs == 0) {
        vector<gsstype>* _bitcol = Vector::init<long>(1, true, false); // FIX
        return _bitcol;
    } else {
        vector<gsstype>* _bitcol = Vector::init<long>(uvs, false, false); // FIX
        _bitcol->val = ptr;
        return _bitcol;
    }
}

/**
 * @brief Retrieves and shrinks the columns of a sparse matrix.
 *
 * This function identifies and returns the shrunk columns from the local sparse matrix.
 *
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @return vector<int>* Pointer to a vector containing the shrunk column indices.
 */
vector<long>* get_shrinked_col(CSR* Alocal, CSR* Plocal)
{
    _MPI_ENV;
    stype myplastrow;

    // GridBlock gb;
    // long* getmct_4shrink(gsstype*, itype, itype, itype, int, int*, long**, int*, int*, int);

    if (Plocal != NULL) {
        myplastrow = Plocal->n - 1;
    } else {
        stype nr_of_cols = Alocal->n;
        // ===============================================================
        // if (Alocal->full_n != Alocal->m) {
        //     nr_of_cols = Alocal->m / nprocs;
        //     if (myid == nprocs - 1) {
        //         nr_of_cols += Alocal->m % nprocs;
        //     }
        // }
        // ===============================================================
        myplastrow = nr_of_cols - 1;
    } // P_n_per_process[i]: number of rows that process i have of matrix P

    // stype nr_of_cols;
    // if (Plocal == NULL) {
    //     if (Alocal->full_n == Alocal->m) {
    //         nr_of_cols = Alocal->n;
    //     } else {
    //         nr_of_cols = Alocal->m / nprocs;
    //         if (myid == nprocs - 1) {
    //             nr_of_cols += Alocal->m % nprocs;
    //         }
    //     }
    //     } else {
    //     nr_of_cols = Plocal->n;
    // }
    // myplastrow = nr_of_cols - 1;

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

    long* ptr = getmct_4shrink(Alocal->col8, Alocal->nnz, 0, myplastrow, first_or_last, &uvs, &(Alocal->bitcol), &(Alocal->bitcolsize), &(Alocal->nonuniquesize), &(Alocal->post_local), NUM_THR);
    vector<long>* _bitcol = Vector::init<long>(uvs, false, true);
    _bitcol->val = ptr;
    return _bitcol;
}

vector<long>* get_shrinked_col(CSR* Alocal, stype firstlocal, stype lastlocal)
{
    _MPI_ENV;
    stype myplastrow;

    // GridBlock gb;
    // long* getmct_4shrink(gsstype*, itype, itype, itype, int, int*, long**, int*, int*, int);

    myplastrow = lastlocal;

    // Alocal->shrinked_lastrow = myplastrow;

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

    long* ptr = getmct_4shrink(Alocal->col8, Alocal->nnz, 0, myplastrow, first_or_last, &uvs, &(Alocal->bitcol), &(Alocal->bitcolsize), &(Alocal->nonuniquesize), &(Alocal->post_local), NUM_THR);
    vector<long>* _bitcol = Vector::init<long>(uvs, false, true);
    _bitcol->val = ptr;
    return _bitcol;
}

/**
 * @brief Computes the rows to receive from other processes.
 *
 * This function calculates which rows need to be received from other processes
 * based on the local sparse matrix and the information about missing columns.
 *
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param _bitcol Pointer to the vector containing missing column indices.
 */
void compute_rows_to_rcv_CPU(CSR* Alocal, CSR* Plocal, vector<long>* _bitcol)
{
    _MPI_ENV;
    
    if (nprocs == 1) {
        return;
    }

    rows_to_get_info* ret = MALLOC(rows_to_get_info, 1, true);

    static int cnt = 0;

    // TRACE("Alocal->n: %d\n"         , Alocal->n);
    // TRACE("Alocal->row_shift: %lu\n", Alocal->row_shift);
    // TRACE("Alocal->m: %lu\n"        , Alocal->m);
    // TRACE("Alocal->full_n: %lu\n"   , Alocal->full_n);
    // if (Plocal) {
    //     TRACE("Plocal->n: %d\n"         , Plocal->n);
    //     TRACE("Plocal->row_shift: %lu\n", Plocal->row_shift);
    //     TRACE("Plocal->m: %lu\n"        , Plocal->m);
    //     TRACE("Plocal->full_n: %lu\n"   , Plocal->full_n);
    // }

    if (Plocal == NULL) {
        Plocal = Alocal;
    }

    // -----------------------------------------------------------------------------------

    TRACE("Exchanging n");
    itype* P_n_per_process = MALLOC(itype, nprocs);

    // send rows numbers to each process, Plocal->n local number of rows
    CHECK_MPI(MPI_Allgather(
        &Plocal->n, sizeof(itype), MPI_BYTE,
        P_n_per_process, sizeof(itype), MPI_BYTE,
        MPI_COMM_WORLD
    ));
    // P_n_per_process[i]: number of rows that process i owns of matrix P

    // debugArray("P_n_per_process[%d] = %d\n", P_n_per_process, nprocs, false, stderr);

    TRACE("Exchanging row_shift");
    gstype row_shift[nprocs];
    CHECK_MPI(MPI_Allgather(
        &Plocal->row_shift, sizeof(gstype), MPI_BYTE,
        row_shift, sizeof(gstype), MPI_BYTE,
        MPI_COMM_WORLD
    ));

    // debugArray("row_shift[%d] = %ld\n", row_shift, nprocs, false, stderr);

    gstype ends[nprocs];
    gsstype cum_p_n_per_process[nprocs];
    ends[0] = P_n_per_process[0];
    cum_p_n_per_process[0] = P_n_per_process[0] - 1;

    TRACE("Computing cum_p_n_per_process, ends");
    for (int i = 1; i < nprocs; i++) {
        cum_p_n_per_process[i] = cum_p_n_per_process[i - 1] + (gstype)P_n_per_process[taskmap ? taskmap[i] : i];
        ends[i] = ends[i - 1] + ((gstype)P_n_per_process[i]);
    }

    // debugArray("ends[%d] = %ld\n", ends, nprocs, false, stderr);
    // debugArray("cum_p_n_per_process[%d] = %ld\n", ends, nprocs, false, stderr);

    ASSERT(ends[nprocs - 1] == Plocal->full_n);

    // -----------------------------------------------------------------------------------

    unsigned int countp[nprocs] = { 0 };
    for (int i = 0; i < nprocs; i++) {
        countp[i] = 0;
    }
    itype countall = 0;

    itype* aofwhichproc = MALLOC(itype, _bitcol->n);
    for (int j = 0; j < _bitcol->n; j++) {
        long search_index = Alocal->row_shift + _bitcol->val[j];

        int whichproc = bswhichprocess(cum_p_n_per_process, nprocs, search_index);
        if (whichproc > (nprocs - 1)) {
            whichproc = nprocs - 1;
        }

        int pwp = whichproc;
        if (taskmap) {
            whichproc = taskmap[whichproc];
        }

#if DEBUG
        if (log_file) {
            fprintf(log_file, "I will request row %d to process %d\n", search_index, whichproc);
        }
#endif

        if (whichproc == myid) {
            fprintf(stderr, "Task %d, unexpected whichproc for col %ld (was %d), line=%d\n", myid, search_index, pwp, __LINE__);
            fprintf(stderr, "Task %d, A->row_shift=%ld, bitcol->val[%d]=%ld\n", myid, Alocal->row_shift, j, _bitcol->val[j]);

            for (itype i = 0; i < nprocs; i++) {
                fprintf(stderr, "cnt=%d, row_shift[%d]=%ld, end[%d]=%ld, cum_p_n_per_process[%d]=%ld,\n", cnt, i, row_shift[i], i, ends[i], i, cum_p_n_per_process[i]);
            }

            DIE("whichproc == myid\n");
        }
        
        countp[whichproc]++;
        aofwhichproc[countall] = whichproc;
        countall++;
    }

    // countp[i]       = num. di righe da richiedere al processo i
    // aofwhichproc[i] = processo che ha in gestione la riga i

    ret->rows2bereceived = countall;

#if DEBUG
    if (log_file) {
        fprintf(log_file, "I will ask for %d rows:\n", countall);
        for (int p = 0; p < nprocs; p++) {
            if (p != myid) {
                fprintf(log_file, "\tI will ask proc %d for %d rows\n", p, countp[p]);
            }
        }
    }
#endif

    // -----------------------------------------------------------------------------------
    
    unsigned int offset[nprocs] = { 0 };
    offset[0] = 0;
    for (int i = 1; i < nprocs; i++) {
        offset[i] = offset[i - 1] + countp[i - 1];
        countp[i - 1] = 0;
    }
    countp[nprocs - 1] = 0;

    // offset[i] = displacement per il vettore che contiene
    //             le righe da richiedere al processo i
    // countp riazzerato per essere riutilizzato

    itype* whichprow = NULL;
    itype* rcvpcolxrow = NULL;
    if (countall > 0) {
        whichprow = MALLOC(itype, countall);
        ret->whichprow = MALLOC(gstype, countall);
        rcvpcolxrow = MALLOC(itype, countall);
    }

    for (int j = 0; j < _bitcol->n; j++) {
        int whichproc = aofwhichproc[j];
        whichprow[offset[whichproc] + countp[whichproc]] = _bitcol->val[j] + (Alocal->row_shift - row_shift[whichproc]);
        countp[whichproc]++;
    }
    FREE(aofwhichproc);

    // countp[i] = num. di righe da richiedere al processo i

    // -----------------------------------------------------------------------------------

    if (countp[myid] != 0) {
        DIE("self countp should be zero! %d\n", myid);
    }

    unsigned int* rcvcntp = MALLOC(unsigned int, nprocs);
    CHECK_MPI(MPI_Alltoall(
        countp, sizeof(itype), MPI_BYTE,
        rcvcntp, sizeof(itype), MPI_BYTE,
        MPI_COMM_WORLD
    ));

    if (rcvcntp[myid] != 0) {
        DIE("self rcvcntp should be zero! %d\n", myid);
    }

#if DEBUG
    if (log_file) {
        // for (int p = 0; p < nprocs; p++) {
        //     fprintf(log_file, "countp[%d] = %d\n", p, countp[p]);
        // }
        // for (int p = 0; p < nprocs; p++) {
        //     fprintf(log_file, "rcvcntp[%d] = %d\n", p, rcvcntp[p]);
        // }
        for (int p = 0; p < nprocs; p++) {
            if (p != myid) {
                fprintf(log_file, "Proc %d asked me for %d rows\n", p, rcvcntp[p]);
            }
        }
    }
#endif

    // rcvcntp[i] = num. di righe da inviare al processo i

    // -----------------------------------------------------------------------------------

    int* displr = MALLOC(int, nprocs);
    int* rcounts2 = MALLOC(int, nprocs);
    int* scounts2 = MALLOC(int, nprocs);
    int* displs2 = MALLOC(int, nprocs);
    int* displr2 = MALLOC(int, nprocs);
    int* displs = MALLOC(int, nprocs);
    int* scounts = MALLOC(int, nprocs);
    
    countall = 0;
    int rcounts[nprocs];
    for (int i = 0; i < nprocs; i++) {
        rcounts2[i] = scounts[i] = countp[i] * sizeof(itype);
        displr2[i] = displs[i] = ((i == 0) ? 0 : (displs[i - 1] + scounts[i - 1]));

        scounts2[i] = rcounts[i] = rcvcntp[i] * sizeof(itype);
        displs2[i] = displr[i] = ((i == 0) ? 0 : (displr[i - 1] + rcounts[i - 1]));
        countall += rcvcntp[i];
    }

    if (countall > 0) {
        int k = 0;
        for (int p = 0; p < nprocs; p++) {
            if (p != myid) {
#if DEBUG
                if (log_file) {
                    // fprintf(log_file, "Proc %d scounts %d\n", p, scounts[p] / sizeof(itype));
                    fprintf(log_file, "I will ask proc %d for %d rows\n", p, scounts[p] / sizeof(itype));
                }
#endif
                for (int j = 0; j < scounts[p] / sizeof(itype); j++) {
                    ret->whichprow[k] = whichprow[k] + row_shift[p];
#if DEBUG
                    if (log_file) {
                        // fprintf(log_file, "\twhichprow[%d]=%d (p=%d, j=%d)\n", k, whichprow[k] + row_shift[p], p, j);
                        fprintf(log_file, "\t%d (shifted: %d)\n", ret->whichprow[k], whichprow[k]);
                    }
#endif
                    k++;
                }
            }
        }
    }

    itype* rcvprow = NULL;
    if (countall > 0) {
        rcvprow = MALLOC(itype, countall);
    }

    CHECK_MPI(MPI_Alltoallv(
        whichprow, scounts, displs, MPI_BYTE,
        rcvprow, rcounts, displr, MPI_BYTE,
        MPI_COMM_WORLD
    ));

#if DEBUG
    if (log_file) {
        int k = 0;
        for (int p = 0; p < nprocs; p++) {
            if (p != myid) {
                fprintf(log_file, "Proc %d asked me for %d rows\n", p, rcounts[p] / sizeof(itype));
                // fprintf(log_file, "Proc %d will send me %d rows\n", p, rcounts[p] / sizeof(itype));
                for (int j = 0; j < rcounts[p] / sizeof(itype); j++) {
                    // fprintf(log_file, "\trcvprow[%d]=%d (p=%d, j=%d)\n", k, rcvprow[k] /*+ row_shift[p]*/, p, j);
                    fprintf(log_file, "\t%d (shifted: %d)\n", rcvprow[k] + row_shift[myid], rcvprow[k]);
                    k++;
                }
            }
        }
    }
#endif

    int k = 0;
    for (int i = 0; i < nprocs; i++) {
        for (int j = 0; j < (scounts[i] / sizeof(itype)); j++) {
            ret->whichprow[k] = whichprow[k] + row_shift[i];
            k++;
        }
    }
    FREE(whichprow);

    // -----------------------------------------------------------------------------------
    // Business logic to handle the number of elements per row, previously coded here,
    // moved to nsparseMGPU_commu_new
    // -----------------------------------------------------------------------------------

    ret->countall = countall;
    ret->rcvprow = rcvprow;
    ret->rcvpcolxrow = rcvpcolxrow;
    ret->displr = displr;
    ret->displs = displs;
    ret->scounts = scounts;
    ret->rcounts2 = rcounts2;
    ret->scounts2 = scounts2;
    ret->displs2 = displs2;
    ret->displr2 = displr2;
    ret->rcvcntp = rcvcntp;
    ret->P_n_per_process = P_n_per_process;
    cnt++;

#if DEBUG
    if (log_file) {
        fprintf(log_file, "Returning from %s\n\n\n", __func__);
    }
#endif

    Alocal->rows_to_get = ret;
}

/**
 * @brief Initializes the completed rows for a sparse matrix.
 *
 * This kernel initializes the completed rows for a sparse matrix based on the
 * number of completed rows.
 *
 * @param completedP_n The number of completed rows.
 * @param new_rows Pointer to the output array where the new row indices will be stored.
 */
__global__ void _completedP_rows(itype completedP_n, itype* new_rows)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < completedP_n) {
        new_rows[id] = id;
    }
}

/**
 * @brief Initializes the completed rows for a sparse matrix with local information.
 *
 * This kernel initializes the completed rows for a sparse matrix based on the
 * number of completed rows and local non-zero counts.
 *
 * @param completedP_n The number of completed rows.
 * @param rows_pre_local The number of rows that are pre-local.
 * @param local_rows The number of local rows.
 * @param nzz_pre_local The number of non-zero elements before local rows.
 * @param Plocal_nnz The number of non-zero elements in the local matrix.
 * @param Plocal_row Pointer to the local row pointers.
 * @param P_nnz_map Pointer to the mapping of rows to their non-zero counts.
 * @param completedP_row Pointer to the output array where the completed row indices will be stored.
 */
__global__ void _completedP_rows2(itype completedP_n, itype rows_pre_local, itype local_rows, itype nzz_pre_local, itype Plocal_nnz, itype tot_nnz, itype* Plocal_row, itype* P_nnz_map, itype* completedP_row)
{
    itype id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id == 0) {
        completedP_row[id] = 0;
    } else if (id < rows_pre_local) {
        completedP_row[id] = P_nnz_map[id - 1];
    } else if (id <= rows_pre_local + local_rows) {
        completedP_row[id] = nzz_pre_local + Plocal_row[id - rows_pre_local]; // id - rows_pre_local
    } else if (id < completedP_n) {
        // completedP_row[id] = nzz_pre_local + Plocal_nnz + P_nnz_map[id - rows_pre_local - local_rows - 1];
        // completedP_row[id] = nzz_pre_local + Plocal_nnz + P_nnz_map[id - local_rows - 1] - nzz_pre_local;
	    completedP_row[id] = Plocal_nnz + P_nnz_map[id - local_rows - 1];
    } else if (id == completedP_n) {
        completedP_row[id] = tot_nnz;
    }
}

/**
 * @brief Main function for sparse matrix multiplication without communication.
 *
 * This function performs sparse matrix multiplication on the GPU without
 * inter-process communication, handling local data only.
 *
 * @param h Pointer to the handles for managing GPU resources.
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param used_by_solver A boolean indicating if the result is used by a solver.
 * @return CSR* Pointer to the resulting sparse matrix after multiplication.
 */
CSR* nsparseMGPU_noCommu_new(handles* h, CSR* Alocal, CSR* Plocal /*, bool used_by_solver*/)
{
    _MPI_ENV;

    TRACE("Before get_shrinked_matrix(A = %p, P = %p)", Alocal, Plocal);
    CSR* Alocal_ = get_shrinked_matrix(Alocal, Plocal);

    cudaDeviceSynchronize();

    csrlocinfo Pinfo1p;
    Pinfo1p.fr = 0;
    Pinfo1p.lr = Plocal->n;
    Pinfo1p.row = Plocal->row;
    Pinfo1p.col = NULL;
    Pinfo1p.val = Plocal->val;

    cudaDeviceSynchronize();

    // printf("Before nsparseMGPU in %s\n", __func__);
    CSR* C = nsparseMGPU(Alocal_, Plocal, &Pinfo1p /*, used_by_solver*/);
    // printf("After nsparseMGPU in %s\n", __func__);
    Alocal_->col = NULL;
    Alocal_->col8 = NULL;
    Alocal_->row = NULL;
    Alocal_->val = NULL;
    FREE(Alocal_);

    return C;
}

// =========================================================================
// BEGIN (sort)

template <typename K, typename V>
struct KeyValue {
    K k;
    V v;
};

template <typename K, typename V>
int compare_by_key_and_val(const void* a, const void* b) {
    const auto &ak = ((KeyValue<K, V>*)a)->k;
    const auto &bk = ((KeyValue<K, V>*)b)->k;
    if (ak != bk) {
        return ak - bk;
    }

    const auto &av = ((KeyValue<K, V>*)a)->v;
    const auto &bv = ((KeyValue<K, V>*)b)->v;
    return av - bv;
}

template <typename K, typename V>
void my_sort(size_t n, K *keys, V *values) {
    typedef KeyValue<K, V> KeyValueImpl;

    KeyValueImpl* temp = MALLOC(KeyValueImpl, n);
    for (int i = 0; i < n; ++i) {
        temp[i].k = keys[i];
        temp[i].v = values[i];
    }

    qsort(temp, n, sizeof(KeyValueImpl), compare_by_key_and_val<K, V>);

    for (int i = 0; i < n; ++i) {
        keys[i] = temp[i].k;
        values[i] = temp[i].v;
    }
    FREE(temp);
}

// END (sort)
// =========================================================================

/**
 * @brief Main function for sparse matrix multiplication with communication.
 *
 * This function performs sparse matrix multiplication on the GPU, handling
 * communication between processes and managing memory allocation.
 *
 * @param h Pointer to the handles for managing GPU resources.
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param used_by_solver A boolean indicating if the result is used by a solver.
 * @return CSR* Pointer to the resulting sparse matrix after multiplication.
 */
CSR* nsparseMGPU_commu_new(handles* h, CSR* Alocal, CSR* Plocal /*, bool used_by_solver*/)
{
    _MPI_ENV;

    csrlocinfo Pinfo1p;
    Pinfo1p.fr = 0;
    Pinfo1p.lr = Plocal->n;
    Pinfo1p.row = Plocal->row;
#if !defined(CSRSEG)
    Pinfo1p.col = NULL;
#else
    Pinfo1p.col = Plocal->col;
    Pinfo1p.col8 = Plocal->col8;
#endif
    Pinfo1p.val = Plocal->val;

    if (nprocs == 1) {
        return nsparseMGPU(Alocal, Plocal, &Pinfo1p /*, used_by_solver*/);
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
    itype countall;
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
    countall = Alocal->rows_to_get->countall;

    gstype* whichprow = NULL;
    itype *rcvpcolxrow = NULL, *rcvprow = NULL;
    rcvprow = Alocal->rows_to_get->rcvprow;
    whichprow = Alocal->rows_to_get->whichprow;
    rcvpcolxrow = Alocal->rows_to_get->rcvpcolxrow;

    // -----------------------------------------------------------------------------------
    // Moved from compute_rows_to_rcv_CPU (BEGIN)
    // -----------------------------------------------------------------------------------

    memset(scounts, 0, nprocs * sizeof(int));
    memset(displs, 0, nprocs * sizeof(int));

    vector<itype>* nnz_per_row_shift = NULL;
    // total_row_to_rec actually store the total rows to send, the sum of the number of rows we must send to each process i
    // rcvcntp[i] = number of rows to send to process i
    itype total_row_to_rec = countall;
    countall = 0;

    TRACE("Before if(total_row_to_rec)");
    itype tot_shift = 0;
    if (total_row_to_rec) {
        nnz_per_row_shift = Vector::init<itype>(total_row_to_rec, true, false); // no temp buff is used on the HOST only;
        itype q = 0;
        itype* hRow = copyArrayToHost(Plocal->row, Plocal->n + 1);
        
        for (int i = 0; i < nprocs; i++) {
// displs[i] = (i == 0) ? 0 : (displs[i - 1] + scounts[i - 1]);
#if DEBUG
            if (log_file && i != myid) {
                // fprintf(log_file, "rcvcntp[%d]=%d\n", i, rcvcntp[i]);
                fprintf(log_file, "I will send %d rows to process %d\n", rcvcntp[i], i);
            }
#endif
            for (int j = 0; j < rcvcntp[i]; j++) {
                itype irow = rcvprow[q];
                itype number_nnz_per_rec_row = hRow[irow + 1] - hRow[irow];
                // itype number_nnz_per_rec_row = 1;
                scounts[i] += number_nnz_per_rec_row;
                nnz_per_row_shift->val[q] = tot_shift;
#if DEBUG
                if (log_file && i != myid) {
                    fprintf(log_file, "\tnnz_per_row_shift->val[%d]=%d\n", q, nnz_per_row_shift->val[q]);
                }
#endif
                tot_shift += number_nnz_per_rec_row;
                q++;
            }
            countall += scounts[i];
#if DEBUG
            if (log_file && i != myid) {
                fprintf(log_file, "\t***scounts[%d]=%d\n", i, scounts[i]);
            }
#endif
            scounts[i] *= sizeof(itype);
            displs[i] = ((i == 0) ? 0 : (displs[i - 1] + scounts[i - 1]));
        }

        // nnz_per_row_shift->val[q] = tot_shift;

        FREE(hRow);
    }
    TRACE("After if(total_row_to_rec)");

    // #if DEBUG
    // if (log_file) {
    //     itype q = 0;
    //     for (int i = 0; i < nprocs; i++) {
    //         if (i == myid) {
    //             continue;
    //         }
    //         for (int j = 0; j < rcvcntp[i]; j++) {
    //             fprintf(log_file, "\tnnz_per_req_row->val[%d]=%d\n", q, nnz_per_req_row->val[q]);
    //             q++;
    //         }
    //     }
    // }
    // #endif

#if DEBUG
    if (log_file) {
        for (int i = 0; i < nprocs; i++) {
            if (i != myid) {
                fprintf(log_file, "scounts[%d]=%d, displs[%d]=%d\n", i, scounts[i] / sizeof(itype), i, displs[i] / sizeof(itype));
            }
        }
    }
#endif

    // -----------------------------------------------------------------------------------
    // Moved from compute_rows_to_rcv_CPU (END)
    // -----------------------------------------------------------------------------------

    itype mycolp = Plocal->nnz; // number of nnz stored by the process
    gstype Pm = (unsigned long)Plocal->m; // number of columns in P
    gstype mypfirstrow = Plocal->row_shift;
    gstype myplastrow = Plocal->n + Plocal->row_shift - 1;
    // itype Pn = Plocal->full_n;

    // vector<itype>* nnz_per_row_shift = NULL;
    // nnz_per_row_shift = Alocal->rows_to_get->nnz_per_row_shift;

    // itype* p2rcvprow;
    // itype countall = 0;
    // countall = Alocal->rows_to_get->countall;
    itype q = 0;

    memset(rcounts, 0, nprocs * sizeof(int));

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
                DIE("Task %d, n=%d\n", myid, nnz_per_row_shift->n);
            }
            dev_nnz_per_row_shift = Vector::init<itype>(nnz_per_row_shift->n, false, true);
            dev_nnz_per_row_shift->val = idevtemp2;

            CHECK_DEVICE(cudaMemcpyAsync(dev_nnz_per_row_shift->val, nnz_per_row_shift->val, dev_nnz_per_row_shift->n * sizeof(itype), cudaMemcpyHostToDevice, h->stream1));
            if (dev_nnz_per_row_shift->n > nnz_per_row_shift_n_stat) {
                if (nnz_per_row_shift_n_stat > 0) {
                    CUDA_FREE(dev_rcvprow_stat);
                }
                nnz_per_row_shift_n_stat = dev_nnz_per_row_shift->n;
                dev_rcvprow_stat = CUDA_MALLOC(itype, nnz_per_row_shift_n_stat, true);
            }
            // for(int icheck=0; icheck<nnz_per_row_shift->n-1; icheck++) { if(rcvprow[icheck]>Plocal->n) { printf("ORRORE 2: %d %d %d %d\n",myid,icheck,rcvprow[icheck],nnz_per_row_shift->n); } }
            CHECK_DEVICE(cudaMemcpyAsync(dev_rcvprow_stat, rcvprow, (dev_nnz_per_row_shift->n) * sizeof(itype), cudaMemcpyHostToDevice, h->stream1));
            // ------------------------------------------------------------------------------------------
        }
        if (nnz_per_row_shift->n > 0) {
            gb = gb1d(dev_nnz_per_row_shift->n, NUM_THR);
            // -------- TEST ---------
            _getColVal<<<gb.g, gb.b, 0, h->stream1>>>(
                dev_nnz_per_row_shift->n,
                dev_rcvprow_stat,
                dev_nnz_per_row_shift->val,

                Plocal->row,
                Plocal->col,
                Plocal->val,
                
                dev_col2send,
                dev_val2send,
                
                0 /* mypfirstrow */
            );
            // -----------------------
            CHECK_DEVICE(cudaStreamSynchronize(h->stream1));
        }

        // ASSERT(col2send);
        // ASSERT(val2send);
        // ASSERT(dev_col2send);
        // ASSERT(dev_val2send);
        // ASSERT(countall);
        // ASSERT(h->stream1);
        // ASSERT(h->stream2);
        // CHECK_DEVICE(cudaStreamSynchronize(h->stream1));
        // CHECK_DEVICE(cudaStreamSynchronize(h->stream2));
        CHECK_DEVICE(cudaMemcpyAsync(col2send, dev_col2send, countall * sizeof(itype), cudaMemcpyDeviceToHost, h->stream1));
        CHECK_DEVICE(cudaMemcpyAsync(val2send, dev_val2send, countall * sizeof(vtype), cudaMemcpyDeviceToHost, h->stream2));

        if (nnz_per_row_shift->n > 0) {
            dev_nnz_per_row_shift->val = NULL;
            FREE(dev_nnz_per_row_shift);
        }
    }
    q = 0;
    for (i = 0; i < nprocs; i++) {
        if (i == myid) {
            continue;
        }
        // shift
        // p2rcvprow = &rcvprow[displr[i] / sizeof(itype)]; // recvprow will be modified
        for (j = 0; j < rcvcntp[i]; j++) {
            rcvprow[q] = (q + 1 == nnz_per_row_shift->n ? tot_shift : nnz_per_row_shift->val[q + 1]) - nnz_per_row_shift->val[q]; // recycle rcvprow to send the number of columns in each row
            // p2rcvprow[j] = 1; // recycle rcvprow to send the number of columns in each row
            q++;
        }
    }

    CHECK_MPI(MPI_Alltoallv(
        rcvprow, scounts2, displs2, MPI_BYTE,
        rcvpcolxrow, rcounts2, displr2, MPI_BYTE,
        MPI_COMM_WORLD
    ));

    #if DEBUG
    if (log_file) {
        debugArray("Number of columns to be received per row rcvpcolxrow[%d]=%d\n", rcvpcolxrow, countall, false, log_file);
    }
    #endif

    itype nzz_pre_local = 0;
    itype rows2bereceived = Alocal->rows_to_get->rows2bereceived;

    vector<itype>* P_nnz_map = NULL;
    itype rows_pre_local = 0;
    if (rows2bereceived) {
        gsstype r = 0;

        P_nnz_map = Vector::init<itype>(rows2bereceived, true, false);
        k = 0;
        int whichproc;

        gsstype cum_p_n_per_process[nprocs];
        cum_p_n_per_process[0] = P_n_per_process[0] - 1;
        for (int j = 1; j < nprocs; j++) {
            cum_p_n_per_process[j] = cum_p_n_per_process[j - 1] + P_n_per_process[taskmap ? taskmap[j] : j];
        }

        // =========================================================================
        // BEGIN (sort)

        if (taskmap) {
            my_sort(rows2bereceived, whichprow, rcvpcolxrow);
        }

        // END (sort)
        // =========================================================================
        
        for (i = 0; i < rows2bereceived; i++) {
            r = whichprow[i];
            // count nnz per process for comunication
            int pwp;
            whichproc = bswhichprocess(cum_p_n_per_process, nprocs, r);
            if (whichproc > (nprocs - 1)) {
                whichproc = nprocs - 1;
            }
            pwp = whichproc;
            if (taskmap) {
                whichproc = taskmap[whichproc];
                // fprintf(stderr, "MPI[%d] pwp %d whichproc %d\n", myid, pwp, whichproc);
            }
            if (whichproc == myid) {
                DIE("Task %d, unexpected whichproc for col %ld (was %d), line=%d\n", myid, r + Alocal->row_shift, pwp, __LINE__);
            }
            rcounts[whichproc] += rcvpcolxrow[i];
            if (r < mypfirstrow) {
                nzz_pre_local += rcvpcolxrow[i];
                rows_pre_local++;
            }
            k += rcvpcolxrow[i];
            P_nnz_map->val[i] = k;

            #if DEBUG
            if (log_file) {
                fprintf(
                    log_file,
                    "%d) I should receive row %d from process %d. It contains %d elements. k is now %d.\n",
                    i, r, whichproc, rcvpcolxrow[i], k
                );
            }
            #endif
        }
    }

    if (rcounts[myid] != 0) {
        DIE("task: %d, unexpected rcount[%d]=%d. It should be zero\n", myid, myid, rcounts[myid]);
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
                int left = itaskmap ? itaskmap[whichdisplr[j]] : whichdisplr[j];
                int right = itaskmap ? itaskmap[whichdisplr[j + 1]] : whichdisplr[j + 1];
                if (left > right) {
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
        int itm_i = itaskmap ? itaskmap[i] : i;
        int itm_myid = itaskmap ? itaskmap[myid] : myid;
        displr_target[i] = (j == 0)
            ? 0
            : (displr_target[previous_index] + (((itm_i > itm_myid) && flagaddmycolp) ? mycolp : rcounts_src[previous_index]));
        if (itm_i > itm_myid) {
            flagaddmycolp = 0;
        }
        rcounts[i] *= sizeof(itype);
        displr[i] = (j == 0) ? 0 : (displr[previous_index] + rcounts[previous_index]);
        displr_src[i] = displr[i] / sizeof(itype);
        previous_index = i;
    }
    if (iPtemp1 == NULL && totcell > 0) { // first allocation
        iPtemp1 = CUDA_MALLOC_HOST(itype, totcell, true);
        vPtemp1 = MALLOC(vtype, totcell, true);
        s_totcell_new = totcell;
    }
    if (totcell > s_totcell_new) { // not enough space
        CUDA_FREE_HOST(iPtemp1);
        #if DEBUG
        printf("[Realloc] --- totcell: %d s_totcell_new: %d\n", totcell, s_totcell_new);
        #endif
        iPtemp1 = CUDA_MALLOC_HOST(itype, totcell, true);
        vPtemp1 = REALLOC(vtype, vPtemp1, s_totcell_new, totcell);
        s_totcell_new = totcell;
    }
    Pcol = iPtemp1;
    Pval = vPtemp1;
    if (countall > 0) {
        cudaStreamSynchronize(h->stream1);
        dev_col2send = NULL;
    }
    CHECK_MPI(MPI_Alltoallv(
        col2send, scounts, displs, MPI_BYTE,
        Pcol, rcounts, displr, MPI_BYTE,
        MPI_COMM_WORLD
    ));

    #if DEBUG
    if (log_file) {
        debugArray("Pcol[%d]=%d\n", Pcol, totcell, false, log_file);
    }
    #endif

    CSR* completedP = NULL;
    itype completedP_n = Plocal->n + rows2bereceived; // Alocal->rows_to_get->total_row_to_rec; (?NOTE?)
    itype completedP_nnz = Plocal->nnz + totcell;
    #if DEBUG
    if (log_file) {
        fprintf(log_file, "Plocal->nnz=%d\n", Plocal->nnz);
        fprintf(log_file, "totcell=%d\n", totcell);
        fprintf(log_file, "completedP_nnz=%d\n", completedP_nnz);
    }
    #endif

    // ------------------------------------- TEST -------------------------------------------
    static int completedP_stat_nnz = 0, completedP_stat_n = 0;

    completedP = CSRm::init(completedP_n, Pm, completedP_nnz, false, true, false, completedP_n,
        Alocal->row_shift
    );
    if (completedP_n > completedP_stat_n || completedP_nnz > completedP_stat_nnz) {
        // cudaError_t err;
        if (completedP_n > completedP_stat_n) {
            if (completedP_stat_n > 0) {
                CUDA_FREE(completedP_stat_row);
            }
            completedP_stat_n = completedP_n;
            completedP_stat_row = CUDA_MALLOC(itype, completedP_stat_n + 1, true);
        }
        if (completedP_nnz > completedP_stat_nnz) {
            if (completedP_stat_nnz > 0) {
                CUDA_FREE(completedP_stat_col);
                CUDA_FREE(completedP_stat_col8);
                CUDA_FREE(completedP_stat_val);
            }
            completedP_stat_nnz = completedP_nnz;
            completedP_stat_val = CUDA_MALLOC(vtype, completedP_stat_nnz, true);
            completedP_stat_col = CUDA_MALLOC(itype, completedP_stat_nnz, true);
            completedP_stat_col8 = CUDA_MALLOC(gsstype, completedP_stat_nnz, true);
        }
    }
    completedP->val = completedP_stat_val;
    completedP->col = completedP_stat_col;
    completedP->col8 = completedP_stat_col8;
    completedP->row = completedP_stat_row;
    // --------------------------------------------------------------------------------------

    vector<itype>* P_nnz_map_dev = NULL;
    if (P_nnz_map) {
        #if DEBUG
        if (log_file) {
            fprintf(log_file, "rows_pre_local = %d\n", rows_pre_local);
            fprintf(log_file, "nzz_pre_local = %d\n", nzz_pre_local);
            fprintf(log_file, "Plocal->n = %d\n", Plocal->n);
            fprintf(log_file, "Plocal->nnz = %d\n", Plocal->nnz);
            debugArray("P_nnz_map[%d]=%d\n", P_nnz_map->val, P_nnz_map->n, false, log_file);
        }
        #endif
        P_nnz_map_dev = Vector::copyToDevice(P_nnz_map);
        Vector::free(P_nnz_map);
    }

    gb = gb1d(completedP_n + 1, NUM_THR);
    // _completedP_rows<<<gb.g, gb.b>>>(completedP_n + 1, completedP->row);
    _completedP_rows2<<<gb.g, gb.b>>>(
        completedP_n,
        rows_pre_local,
        Plocal->n,
        nzz_pre_local,
        Plocal->nnz,
        completedP_nnz,
        Plocal->row,
        P_nnz_map_dev->val,
        completedP->row
    );
    // debugArray("completedP->row[%d]=%d\n", completedP->row, completedP->n + 1, true, stderr);
    #if DEBUG
    if (log_file) {
        fprintf(log_file, "mypfirstrow=%d\n", mypfirstrow);
        fprintf(log_file, "myplastrow=%d\n", myplastrow);
        fprintf(log_file, "rows_pre_local=%d\n", rows_pre_local);
        fprintf(log_file, "nzz_pre_local=%d\n", nzz_pre_local);
        fprintf(log_file, "completedP->n=%d\n", completedP->n);
        debugArray("P_nnz_map[%d]=%d\n", P_nnz_map_dev->val, P_nnz_map_dev->n, true, log_file);
        debugArray("Plocal->row[%d]=%d\n", Plocal->row, Plocal->n + 1, true, log_file);
        debugArray("completedP->row[%d]=%d\n", completedP->row, completedP->n + 1, true, log_file);
    }
    #endif

    if (P_nnz_map_dev) {
        Vector::free(P_nnz_map_dev);
    }

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
    CHECK_MPI(MPI_Alltoallv(
        val2send, scounts, displs, MPI_BYTE,
        Pval, rcounts, displr, MPI_BYTE,
        MPI_COMM_WORLD
    ));
    val2send = NULL;

    #if DEBUG
    if (log_file) {
        debugArray("Pval[%d]=%lf\n", Pval, totcell, false, log_file);
    }
    #endif

    for (j = 0; j < ndisplr; j++) {
        i = whichdisplr[j];
        if (rcounts_src[i] > 0) {
            CHECK_DEVICE(cudaMemcpy(completedP->val + displr_target[i], Pval + displr_src[i], rcounts_src[i] * sizeof(vtype), cudaMemcpyHostToDevice));
        }
    }

#if !defined(CSRSEG)
    CHECK_DEVICE(cudaMemcpy(completedP->val + nzz_pre_local, Plocal->val, Plocal->nnz * sizeof(vtype), cudaMemcpyDeviceToDevice));
    CHECK_DEVICE(cudaMemcpy(completedP->col + nzz_pre_local, Plocal->col, Plocal->nnz * sizeof(itype), cudaMemcpyDeviceToDevice));
    CHECK_DEVICE(cudaMemcpy(completedP->col8 + nzz_pre_local, Plocal->col8, Plocal->nnz * sizeof(gsstype), cudaMemcpyDeviceToDevice));
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

    // trace_enabled = true;
    TRACE("Before get_shrinked_matrix(A = %p, P = %p)", Alocal, Plocal);
    CSR* Alocal_ = get_shrinked_matrix(Alocal, Plocal);
    if (!Alocal_->col8) {
        Alocal_->col8 = CUDA_MALLOC(gsstype, Alocal_->nnz);
    }
    ASSERT(Alocal_->col8);
    col2col8(Alocal_->col, Alocal_->col8, Alocal_->nnz);
    TRACE("After get_shrinked_matrix(A = %p, P = %p)", Alocal, Plocal);

    #if 0
    if (1) {
        char fname[1024] = {0};

        snprintf(fname, 1024, "%s/%s%s_myid%d_nprocs%d.mtx", output_dir.c_str(), "" /*output_prefix.c_str()*/, "Alocal", myid, nprocs);
        printf("Dumping %s\n", fname);
        CSRm::printMM(Alocal, fname, false);

        snprintf(fname, 1024, "%s/%s%s_myid%d_nprocs%d.mtx", output_dir.c_str(), "" /*output_prefix.c_str()*/, "Plocal", myid, nprocs);
        printf("Dumping %s\n", fname);
        CSRm::printMM(Plocal, fname, false);
        
        snprintf(fname, 1024, "%s/%s%s_myid%d_nprocs%d.mtx", output_dir.c_str(), "" /*output_prefix.c_str()*/, "Alocal_shrinked", myid, nprocs);
        printf("Dumping %s\n", fname);
        CSRm::printMM(Alocal_, fname, false);
        
        snprintf(fname, 1024, "%s/%s%s_myid%d_nprocs%d.mtx", output_dir.c_str(), "" /*output_prefix.c_str()*/, "completedP", myid, nprocs);
        printf("Dumping %s\n", fname);
        CSRm::printMM(completedP, fname, false);
    }
    #endif

    cudaDeviceSynchronize();
    if (Alocal_->m != completedP->n) {
        fprintf(stderr, "[%d] Alocal_->m = %lu != %d = completedP->n (totcell = %d, Plocal->n = %d, countall = %d, rows2berecived = %d, nzz_pre_local=%d)\n", myid, Alocal_->m, completedP->n, totcell, Plocal->n, countall, Alocal->rows_to_get->rows2bereceived, nzz_pre_local);
    }
    ASSERT(Alocal_->m == completedP->n);

    #if DEBUG
    if (log_file) {
        fprintf(log_file, "\nAlocal:\n");
        CSRm::printMM(Alocal, log_file);

        // fprintf(log_file, "\nAlocal_ (shrinked):\n");
        // CSRm::print(Alocal_, 0, 0, log_file);
        // CSRm::print(Alocal_, 1, 0, log_file);
        // CSRm::print(Alocal_, 2, 0, log_file);
        // CSRm::print(Alocal_, 3, -1, log_file);
        
        // CSRm::printMM(Alocal_, log_file);
    }

    if (log_file) {
        fprintf(log_file, "completedP:\n");
        // CSRm::print(completedP, 0, 0, log_file);
        // CSRm::print(completedP, 1, 0, log_file);
        // CSRm::print(completedP, 2, 0, log_file);
        // CSRm::print(completedP, 3, -1, log_file);

        CSRm::printMM(completedP, log_file);
    }
    #endif

    CSR* C = nsparseMGPU(Alocal_, completedP, &Plocalinfo /*, used_by_solver*/);

    Pcol = NULL;
    Pval = NULL; // memory will free up in AMG
    CSRm::free_rows_to_get(Alocal);
    FREE(completedP);

    Alocal_->col = NULL;
    Alocal_->col8 = NULL;
    Alocal_->row = NULL;
    Alocal_->val = NULL;
    FREE(Alocal_);

    cnt++;

    return C;
}

/**
 * @brief Performs sparse matrix multiplication.
 *
 * This function orchestrates the sparse matrix multiplication process, handling
 * memory allocation and communication between processes.
 *
 * @param Alocal Pointer to the local sparse matrix.
 * @param Plocal Pointer to the local information of the sparse matrix.
 * @param mem_alloc_size The size of memory to allocate for temporary storage.
 * @return CSR* Pointer to the resulting sparse matrix after multiplication.
 */
CSR* SpMM(CSR* Alocal, CSR* Plocal, int mem_alloc_size)
{
    _MPI_ENV;
    handles* h = Handles::init();

    iPtemp1 = NULL;
    vPtemp1 = NULL;
    iAtemp1 = CUDA_MALLOC_HOST(itype, mem_alloc_size, true);
    vAtemp1 = CUDA_MALLOC_HOST(vtype, mem_alloc_size, true);
    idevtemp1 = CUDA_MALLOC(itype, mem_alloc_size, true);
    vdevtemp1 = CUDA_MALLOC(vtype, mem_alloc_size, true);
    idevtemp2 = CUDA_MALLOC(itype, mem_alloc_size, true);

    if (nprocs != 1) {
        vector<gsstype>* _bitcol = get_missing_col(Alocal, NULL);
        compute_rows_to_rcv_CPU(Alocal, NULL, _bitcol);
        Vector::free(_bitcol);
    }

    TRACE("Before nsparseMGPU_commu_new");
    CSR* APlocal = nsparseMGPU_commu_new(h, Alocal, Plocal /*, false*/);
    TRACE("After nsparseMGPU_commu_new");

    return APlocal;
}
