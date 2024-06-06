#pragma once

#include "utility/globals.h"
#include "utility/mpi.h"
#include "utility/setting.h"
#include "utility/utils.h"

#include "utility/ColumnIndexSender.h"
#include "utility/cudamacro.h"
#include "utility/deviceForEach.h"
#include "utility/deviceMap.h"
#include "utility/devicePartition.h"
#include "utility/devicePrefixSum.h"
#include "utility/deviceSort.h"
#include "utility/deviceUnique.h"

#include "datastruct/matrixItem.h"
#include "datastruct/vector.h"

#include "basic_kernel/halo_communication/newoverlap.h"

#include <mpi.h>

struct halo_info {
    vector<gstype>* to_receive = NULL;
    vector<gstype>* to_receive_d = NULL;
    int to_receive_n = 0;
    int* to_receive_counts = NULL;
    int* to_receive_spls = NULL;
    vtype* what_to_receive = NULL;
    vtype* what_to_receive_d = NULL;

    vector<itype>* to_send = NULL;
    vector<itype>* to_send_d = NULL;
    int to_send_n = 0;
    int* to_send_counts = NULL;
    int* to_send_spls = NULL;
    vtype* what_to_send = NULL;
    vtype* what_to_send_d = NULL;

    bool init = false;
};

typedef struct rows_info {
    vector<itype>* nnz_per_row_shift = NULL;
    itype rows2bereceived = 0;
    itype countall = 0;
    itype* P_n_per_process = NULL;
    int* scounts = NULL;
    int* displr = NULL;
    int* displs = NULL;
    int* rcounts2 = NULL;
    int* scounts2 = NULL;
    int* displr2 = NULL;
    int* displs2 = NULL;
    unsigned int* rcvcntp = NULL;
    itype* rcvprow = NULL;
    gstype* whichprow = NULL;
    itype* rcvpcolxrow = NULL;
} rows_to_get_info;

#define PRINTM(x)         \
    CSRm::print(x, 0, 0); \
    CSRm::print(x, 1, 0); \
    CSRm::print(x, 2, 0);

struct overlappedList {
    cudaStream_t* local_stream = NULL;
    cudaStream_t* comm_stream = NULL;
};

struct overlapped {
    vector<itype>* loc_rows = NULL;
    itype loc_n = 0;

    vector<itype>* needy_rows = NULL;
    itype needy_n = 0;

    struct overlappedList* streams = NULL;
};

struct CSR {
    stype nnz = 0; // number of non-zero
    stype n = 0; // rows number
    gstype m = 0; // columns number
    stype shrinked_m = 0; // columns number for the shrinked matrix

    gstype full_n = 0;
    gstype full_m = 0;

    bool on_the_device = false;
    bool is_symmetric = false;
    bool shrinked_flag = false;
    bool custom_alloced = false;
    gsstype col_shifted = 0;
    gstype shrinked_firstrow = 0;
    gstype shrinked_lastrow = 0;

    vtype* val = NULL; // array of nnz values
    itype* col = NULL; // array of the column index
    itype* row = NULL; // array of the pointer of the first nnz element of the rows
    itype* shrinked_col = NULL; // array of the shrinked column indexes

    gstype row_shift = 0;
    int* bitcol = NULL;
    int bitcolsize = 0;
    int post_local = 0;

    halo_info halo;

    rows_to_get_info* rows_to_get = NULL;

    struct overlapped os;
};

struct matrixItem_t;

struct NnzColumnSelector {
    itype* operator()(matrixItem_t* d_nnzItemsToBeRequested, size_t nnzItemsToBeRequestedSize, size_t* columnsToBeRequestedSize)
    {
        *columnsToBeRequestedSize = nnzItemsToBeRequestedSize;
        if (!nnzItemsToBeRequestedSize) {
            return NULL;
        }
        return deviceMap<matrixItem_t, itype, MatrixItemColumnMapper>(
            d_nnzItemsToBeRequested,
            nnzItemsToBeRequestedSize,
            MatrixItemColumnMapper());
    }
};

struct EveryColumnSelector {
    gstype row_shift;

    EveryColumnSelector(gstype row_shift)
        : row_shift(row_shift)
    {
    }

    itype* operator()(matrixItem_t* d_nnzItemsToBeRequested, size_t nnzItemsToBeRequestedSize, size_t* columnsToBeRequestedSize)
    {
        *columnsToBeRequestedSize = row_shift;

        if (!row_shift) {
            return NULL;
        }

        itype* ret = NULL;
        CHECK_DEVICE(cudaMalloc(&ret, row_shift * sizeof(itype)));

        deviceForEach(ret, row_shift, FillWithIndexOperator<itype>());

        return ret;
    }
};

namespace CSRm {
CSR* init(stype n, gstype m, stype nnz, bool allocate_mem, bool on_the_device, bool is_symmetric, gstype full_n, gstype row_shift = 0);
void free(CSR* A);
void printInfo(CSR* A, FILE* fp = stdout);

int choose_mini_warp_size(CSR* A);
void free_rows_to_get(CSR* A);
void print(CSR* A, int type, int limit = 0, FILE* fp = stdout);
void printMM(CSR* A, char* name);
CSR* copyToDevice(CSR* A);
CSR* copyToHost(CSR* A_d);

// CHECK op/mydiag
vector<vtype>* diag(CSR* A);

// Transpose_local is used only in matchingAggregation/matchingPairAggragation.cu
// BE CAREFUL: the transposed matrix has row_shift = 0 and column_shifted = 0
//             even if dlA is row/col shifted.
CSR* Transpose_local(CSR* dlA, FILE* f);

CSR* transpose(CSR* A, FILE* f);

vector<vtype>* absoluteRowSum(CSR* A, vector<vtype>* sum);

matrixItem_t* collectMatrixItems(CSR* dlA, FILE* debug, bool useColShift = false);

vector<vtype>* CSRVector_product_adaptive_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, vtype alpha = 1., vtype beta = 0.);
vector<vtype>* CSRVector_product_adaptive_indirect_row_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, itype n, itype* rows, cudaStream_t stream, unsigned int offset, vtype alpha = 1., vtype beta = 0.);
vector<vtype>* shifted_CSRVector_product_adaptive_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, itype shift, vtype alpha = 1., vtype beta = 0.);
vector<vtype>* shifted_CSRVector_product_adaptive_miniwarp2(CSR* A, vector<vtype>* x, vector<vtype>* y, itype shift, vtype alpha = 1., vtype beta = 0.);
vector<vtype>* CSRVector_product_prolungator(CSR* A, vector<vtype>* x, vector<vtype>* y);
vector<vtype>* CSRVector_product_adaptive_miniwarp_new(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha = 1., vtype beta = 0.);
vector<vtype>* CSRVector_product_adaptive_miniwarp_witho(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha = 1., vtype beta = 0.);
// PICO line
vector<vtype>* CSRVector_product_MPI(CSR* Alocal, vector<vtype>* x, int type);

vector<vtype>* CSRscale_adaptive_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, vtype alpha = 1., vtype beta = 0.);
vector<vtype>* CSRscaleA_0(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha = 1., vtype beta = 0.);

void checkMatrix(CSR* A_, bool check_diagonal = false);
void checkColumnsOrder(CSR* A);
void shift_cols(CSR* A, gsstype shift);

// kernel
__global__ void _CSR_vector_mul_mini_indexed_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y, itype* to_comp, itype shift, itype op_type);

// kernels
template <int OP_TYPE>
__global__ void _CSR_vector_mul_mini_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y);

// kernels
template <int OP_TYPE>
__global__ void _CSR_vector_mul_mini_warp_indirect(itype n, itype*, unsigned, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y);

// kernels
template <int OP_TYPE>
__global__ void _CSR_scale_mini_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype* A_val, itype* A_row, itype* A_col, vtype* x, vtype* y);

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
__global__ void countNnzPerRow(
    itype* row,
    itype row_shift,
    itype* requestedRowIndexes,
    itype requestedRowIndexesSize,
    itype* ret);

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
__global__ void collectNnzPerRow(
    int warpSize,
    itype* row,
    itype* col,
    vtype* val,
    itype row_shift,
    itype n,
    itype* requestedRowIndexes,
    itype* counter,
    itype* offset,
    matrixItem_t* ret);

/**
 * Given the local portion of a matrix (dlA), asks the other MPI processes
 * the (whole) rows whose (global) index satisfies the supplied operator.
 * Result is returned as an array of matrixItem_t(s), host-allocated.
 *
 * Operators:
 *   MatrixItemColumnIndexLessThanSelector selector(
 *     dlA->row_shift
 *   );
 *
 *   MatrixItemColumnIndexGreaterThanSelector selector(
 *     dlA->row_shift + dlA->n - 1
 *   );
 *
 *   MatrixItemColumnIndexOutOfBoundsSelector selector(
 *     dlA->row_shift,
 *     dlA->row_shift + dlA->n - 1
 *   );
 *
 * @param dlA device local A
 * @param f process-specific log file
 * @param retSize returned array length
 * @param d_nnzItemsRet returned array of nnz matrixItem_t(s);
 *                      can be NULL, otherwise must be released.
 * @param rowsToBeRequestedSizeRet returned d_rowsToBeRequestedRet's length
 * @param d_rowsToBeRequestedRet returned array of ordered unique row indexes to be requested;
 *                               can be NULL, otherwise must be released.
 * @param matrixItemSelector operator to be used in order to partition (local) elements
 * @param columnSelector operator to be used in order to choose which column indexes must be requested
 * @return array of matrix items.
 */
template <typename MatrixItemSelector, typename ColumnSelector>
matrixItem_t* requestMissingRows(CSR* dlA, FILE* f, size_t* retSize,
    matrixItem_t** d_nnzItemsRet,
    size_t* rowsToBeRequestedSizeRet,
    itype** d_rowsToBeRequestedRet,
    MatrixItemSelector matrixItemSelector,
    ColumnSelector columnSelector,
    bool useRowShift);
}

void check_and_fix_order(CSR* A);
void bubbleSort(itype arr[], vtype val[], itype n);

CSR* read_matrix_from_file(const char* matrix_path, int m_type, bool loadOnDevice = true);

CSR* readMTXDouble(const char* file_name);

CSR* readMTX2Double(const char* file_name);

void CSRMatrixPrintMM(CSR* A_, const char* file_name);

// =============================================================================

template <typename MatrixItemSelector, typename ColumnSelector>
matrixItem_t* CSRm::requestMissingRows(CSR* dlA, FILE* f, size_t* retSize,
    matrixItem_t** d_nnzItemsRet,
    size_t* rowsToBeRequestedSizeRet,
    itype** d_rowsToBeRequestedRet,
    MatrixItemSelector matrixItemSelector,
    ColumnSelector columnSelector,
    bool useRowShift)
{
    _MPI_ENV;

    assert(dlA->on_the_device);

    if (f) {
        fprintf(f, "n (rows) : %d\n", dlA->n);
        fprintf(f, "m (cols) : %d\n", dlA->m);
        fprintf(f, "nnz      : %d\n", dlA->nnz);
        fprintf(f, "row shift: %d\n", dlA->row_shift);
    }

    // Register custom MPI datatypes
    // ---------------------------------------------------------------------------
    registerMatrixItemMpiDatatypes();

    // Collect non-zero items in dlA
    // ---------------------------------------------------------------------------
    matrixItem_t* d_nnzItems = collectMatrixItems(dlA, f);

    // Identify the items to be requested: they are the ones whose column
    // index is before the first row index assigned to the process or
    // after the last row assigned to the process.
    // ---------------------------------------------------------------------------

    size_t nnzItemsToBeRequestedSize = 0;
    matrixItem_t* d_nnzItemsToBeRequested = devicePartition(
        d_nnzItems,
        dlA->nnz,
        matrixItemSelector,
        &nnzItemsToBeRequestedSize);

    if (f) {
        fprintf(f, "nnzItemsToBeRequested effective size: %d\n", nnzItemsToBeRequestedSize);
        debugMatrixItems("nnzItemsToBeRequested", d_nnzItemsToBeRequested, nnzItemsToBeRequestedSize, true, f);
    }

    // // Identify the items not to be requested: they are the ones whose column
    // // index is between the first row index and the last row index
    // // assigned to the process.
    // // ---------------------------------------------------------------------------
    // matrixItem_t *d_nnzItemsNotToBeRequested = d_nnzItemsToBeRequested + nnzItemsToBeRequestedSize;
    // size_t nnzItemsNotToBeRequestedSize = dlA->nnz - nnzItemsToBeRequestedSize;

    // if (f) {
    //   fprintf(f, "nnzItemsNotToBeRequested effective size: %d\n", nnzItemsNotToBeRequestedSize);
    //   debugMatrixItems("nnzItemsNotToBeRequested", d_nnzItemsNotToBeRequested, nnzItemsNotToBeRequestedSize, true, f);
    // }

    // Release memory or return nnzItems
    // ---------------------------------------------------------------------------
    if (d_nnzItemsRet) {
        *d_nnzItemsRet = d_nnzItems;
    } else {
        CudaFree(d_nnzItems);
    }

    // Extract column indexes (** REQUESTS **)
    // ---------------------------------------------------------------------------
    // size_t columnsToBeRequestedSize = nnzItemsToBeRequestedSize;
    // itype *d_columnsToBeRequested = deviceMap<matrixItem_t, itype, MatrixItemColumnMapper>(
    //   d_nnzItemsToBeRequested,
    //   nnzItemsToBeRequestedSize,
    //   MatrixItemColumnMapper()
    // );

    size_t columnsToBeRequestedSize = 0;
    itype* d_columnsToBeRequested = columnSelector(
        d_nnzItemsToBeRequested, nnzItemsToBeRequestedSize, &columnsToBeRequestedSize);

    // Release memory
    // ---------------------------------------------------------------------------
    CudaFree(d_nnzItemsToBeRequested);
    // cudaFree(d_nnzItemsNotToBeRequested);

    // Sort column indexes
    // ---------------------------------------------------------------------------
    deviceSort<itype, itype, ColumnIndexComparator>(d_columnsToBeRequested, columnsToBeRequestedSize, ColumnIndexComparator());

    if (f) {
        debugArray("d_columnsToBeRequested[%d] = %d\n", d_columnsToBeRequested, columnsToBeRequestedSize, true, f);
    }

    // Remove duplicates
    // ---------------------------------------------------------------------------
    size_t columnsToBeRequestedUniqueSize;
    itype* d_columnsToBeRequestedUnique = deviceUnique<itype>(d_columnsToBeRequested, columnsToBeRequestedSize, &columnsToBeRequestedUniqueSize);

    if (rowsToBeRequestedSizeRet) {
        *rowsToBeRequestedSizeRet = columnsToBeRequestedUniqueSize;
    }

    if (d_rowsToBeRequestedRet) {
        *d_rowsToBeRequestedRet = d_columnsToBeRequestedUnique;
    }

    if (f) {
        debugArray("d_columnsToBeRequestedUnique[%d] = %d\n", d_columnsToBeRequestedUnique, columnsToBeRequestedUniqueSize, true, f);
    }

    // Copy unique column indexes to host in order to perform MPI communication
    // ---------------------------------------------------------------------------
    itype* h_columnsToBeRequestedUnique = copyArrayToHost(d_columnsToBeRequestedUnique, columnsToBeRequestedUniqueSize);

    // Release memory
    // ---------------------------------------------------------------------------
    CudaFree(d_columnsToBeRequested);
    if (!d_rowsToBeRequestedRet) {
        CudaFree(d_columnsToBeRequestedUnique);
    }

    // Exchange data with other processes.
    // After this step, rcvIndexBuffer will contain the indexes requested by
    // other processes.
    // ---------------------------------------------------------------------------
    ProcessSelector processSelector(dlA, f);
    processSelector.setUseRowShift(useRowShift);

    ColumnIndexSender columnIndexSender(&processSelector, f);
    columnIndexSender.setUseRowShift(useRowShift);

    MpiBuffer<itype> sendIndexBuffer;
    MpiBuffer<itype> rcvIndexBuffer;
    columnIndexSender.send(h_columnsToBeRequestedUnique, columnsToBeRequestedUniqueSize,
        &sendIndexBuffer, &rcvIndexBuffer);

    // Release memory
    // ---------------------------------------------------------------------------
    ::Free(h_columnsToBeRequestedUnique);

    // Move data to device
    // ---------------------------------------------------------------------------
    itype* d_requestedRowIndexes = copyArrayToDevice(rcvIndexBuffer.buffer, rcvIndexBuffer.size);

    // Sort row indexes
    // ---------------------------------------------------------------------------
    size_t requestedRowIndexesSize = rcvIndexBuffer.size;
    deviceSort<itype, itype, ColumnIndexComparator>(d_requestedRowIndexes, requestedRowIndexesSize, ColumnIndexComparator());

    // Remove duplicates
    // ---------------------------------------------------------------------------
    size_t requestedRowIndexesUniqueSize;
    itype* d_requestedRowIndexesUnique = deviceUnique<itype>(d_requestedRowIndexes, requestedRowIndexesSize, &requestedRowIndexesUniqueSize);

    if (f) {
        debugArray("d_requestedRowIndexesUnique[%d] = %d\n", d_requestedRowIndexesUnique, requestedRowIndexesUniqueSize, true, f);
    }

    // Count nnz per row
    // ---------------------------------------------------------------------------
    itype* d_nnzPerRowCounter = NULL;
    if (requestedRowIndexesUniqueSize) {
        GridBlock gb = getKernelParams(requestedRowIndexesUniqueSize); // One thread per requested row index
        CHECK_DEVICE(cudaMalloc(&d_nnzPerRowCounter, requestedRowIndexesUniqueSize * sizeof(itype)));
        countNnzPerRow<<<gb.g, gb.b>>>(
            dlA->row,
            dlA->row_shift,
            d_requestedRowIndexesUnique,
            requestedRowIndexesUniqueSize,
            d_nnzPerRowCounter);
        cudaError_t err = cudaDeviceSynchronize();
        CHECK_DEVICE(err);
    }

    if (f) {
        debugArray("d_nnzPerRowCounter[%d] = %d\n", d_nnzPerRowCounter, requestedRowIndexesUniqueSize, true, f);
    }

    // Compute offsets for requested rows
    // ---------------------------------------------------------------------------
    itype* d_nnzPerRowOffset = NULL;
    CHECK_DEVICE(cudaMalloc(&d_nnzPerRowOffset, (requestedRowIndexesUniqueSize + 1) * sizeof(itype)));
    CHECK_DEVICE(cudaMemset(d_nnzPerRowOffset, 0, (requestedRowIndexesUniqueSize + 1) * sizeof(itype)));
    CHECK_DEVICE(cudaMemcpy(
        d_nnzPerRowOffset + 1,
        d_nnzPerRowCounter,
        requestedRowIndexesUniqueSize * sizeof(itype),
        cudaMemcpyDeviceToDevice));
    devicePrefixSum(d_nnzPerRowOffset, requestedRowIndexesUniqueSize + 1);

    if (f) {
        debugArray("d_nnzPerRowOffset[%d] = %d\n", d_nnzPerRowOffset, requestedRowIndexesUniqueSize + 1, true, f);
    }

    // Collect nnz per row.
    // Buffer size must be equal to the last element in offset[]
    // ---------------------------------------------------------------------------
    itype nnzPerRowBufferSize = 0;
    CHECK_DEVICE(cudaMemcpy(
        &nnzPerRowBufferSize,
        d_nnzPerRowOffset + requestedRowIndexesUniqueSize,
        sizeof(itype),
        cudaMemcpyDeviceToHost));
    if (f) {
        fprintf(f, "nnzPerRowBufferSize: %d\n", nnzPerRowBufferSize);
    }
    matrixItem_t* d_nnzPerRowBuffer = NULL;
    CHECK_DEVICE(cudaMalloc(&d_nnzPerRowBuffer, nnzPerRowBufferSize * sizeof(matrixItem_t)));
    CHECK_DEVICE(cudaMemset(d_nnzPerRowBuffer, 0, nnzPerRowBufferSize * sizeof(matrixItem_t)));

    // One mini warp per requested row
    if (requestedRowIndexesUniqueSize) {
        int warpSize = CSRm::choose_mini_warp_size(dlA);
        GridBlock gb = getKernelParams(requestedRowIndexesUniqueSize * warpSize);
        collectNnzPerRow<<<gb.g, gb.b>>>(
            warpSize,
            dlA->row,
            dlA->col,
            dlA->val,
            dlA->row_shift,
            requestedRowIndexesUniqueSize,
            d_requestedRowIndexesUnique,
            d_nnzPerRowCounter,
            d_nnzPerRowOffset,
            d_nnzPerRowBuffer);
        cudaError_t err = cudaDeviceSynchronize();
        CHECK_DEVICE(err);
    }

    if (f) {
        debugMatrixItems("d_nnzPerRowBuffer", d_nnzPerRowBuffer, nnzPerRowBufferSize, true, f);
    }

    // Map requested row indexes to actual indexes
    // ---------------------------------------------------------------------------
    itype* h_requestedRowIndexesUnique = copyArrayToHost(
        d_requestedRowIndexesUnique, requestedRowIndexesUniqueSize);

    itype requestedRowIndex2actualIndex[dlA->n];
    memset(requestedRowIndex2actualIndex, -1, dlA->n * sizeof(int));
    for (int i = 0; i < requestedRowIndexesUniqueSize; i++) {
        requestedRowIndex2actualIndex[h_requestedRowIndexesUnique[i] - dlA->row_shift] = i;
    }
    ::free(h_requestedRowIndexesUnique);

    if (f) {
        debugArray("requestedRowIndex2actualIndex[%d] = %d\n", requestedRowIndex2actualIndex, dlA->n, false, f);
    }

    // Count the number of elements to be sent to other processes
    // ---------------------------------------------------------------------------
    itype* h_nnzPerRowCounter = copyArrayToHost(
        d_nnzPerRowCounter, requestedRowIndexesUniqueSize);

    MpiBuffer<itype> sendTotNnzBuffer;
    sendTotNnzBuffer.size = sendTotNnzBuffer.nprocs - 1;
    sendTotNnzBuffer.init();

    MpiBuffer<itype> rcvTotNnzBuffer;
    rcvTotNnzBuffer.size = rcvTotNnzBuffer.nprocs - 1;
    rcvTotNnzBuffer.init();

    for (int i = 0; i < nprocs; i++) {
        sendTotNnzBuffer.counter[i] = (i == myid) ? 0 : 1;
        sendTotNnzBuffer.offset[i] = (i == 0)
            ? 0
            : sendTotNnzBuffer.offset[i - 1] + sendTotNnzBuffer.counter[i - 1];

        rcvTotNnzBuffer.counter[i] = sendTotNnzBuffer.counter[i];
        rcvTotNnzBuffer.offset[i] = sendTotNnzBuffer.offset[i];

        if (i == myid) {
            continue;
        }

        for (int j = 0; j < rcvIndexBuffer.counter[i]; j++) {
            itype requestedRowIndex = rcvIndexBuffer.buffer[rcvIndexBuffer.offset[i] + j];
            itype actualIndex = requestedRowIndex2actualIndex[requestedRowIndex - dlA->row_shift];
            sendTotNnzBuffer.buffer[i < myid ? i : i - 1] += h_nnzPerRowCounter[actualIndex];
        }
    }

    if (f) {
        debugArray("sendTotNnzBuffer.buffer[%d] = %d\n", sendTotNnzBuffer.buffer, sendTotNnzBuffer.size, false, f);
        debugArray("sendTotNnzBuffer.counter[%d] = %d\n", sendTotNnzBuffer.counter, sendTotNnzBuffer.nprocs, false, f);
        debugArray("sendTotNnzBuffer.offset[%d] = %d\n", sendTotNnzBuffer.offset, sendTotNnzBuffer.nprocs, false, f);
    }

    // Exchange the number of elements to be received with other processes
    // ---------------------------------------------------------------------------
    CHECK_MPI(
        MPI_Alltoallv(
            sendTotNnzBuffer.buffer,
            sendTotNnzBuffer.counter,
            sendTotNnzBuffer.offset,
            MPI_ITYPE,
            rcvTotNnzBuffer.buffer,
            rcvTotNnzBuffer.counter,
            rcvTotNnzBuffer.offset,
            MPI_ITYPE,
            MPI_COMM_WORLD));

    if (f) {
        debugArray("rcvTotNnzBuffer.buffer[%d] = %d\n", rcvTotNnzBuffer.buffer, rcvTotNnzBuffer.size, false, f);
        debugArray("rcvTotNnzBuffer.counter[%d] = %d\n", rcvTotNnzBuffer.counter, rcvTotNnzBuffer.nprocs, false, f);
        debugArray("rcvTotNnzBuffer.offset[%d] = %d\n", rcvTotNnzBuffer.offset, rcvTotNnzBuffer.nprocs, false, f);
    }

    // Count the number of elements to be sent to other processes,
    // and prepare sending MpiBuffer (set offsets, counters, and allocate memory)
    // ---------------------------------------------------------------------------
    MpiBuffer<matrixItem_t> sendItemsBuffer;
    for (int i = 0; i < nprocs; i++) {
        sendItemsBuffer.offset[i] = (i == 0)
            ? 0
            : sendItemsBuffer.offset[i - 1] + sendItemsBuffer.counter[i - 1];

        if (i == myid) {
            continue;
        }

        for (int j = 0; j < rcvIndexBuffer.counter[i]; j++) {
            itype requestedRowIndex = rcvIndexBuffer.buffer[rcvIndexBuffer.offset[i] + j];
            itype actualIndex = requestedRowIndex2actualIndex[requestedRowIndex - dlA->row_shift];
            sendItemsBuffer.counter[i] += h_nnzPerRowCounter[actualIndex];
        }

        sendItemsBuffer.size += sendItemsBuffer.counter[i];
    }
    sendItemsBuffer.init();

    // Fill sending MpiBuffer with data
    // ---------------------------------------------------------------------------
    itype* h_nnzPerRowOffset = copyArrayToHost(
        d_nnzPerRowOffset, requestedRowIndexesUniqueSize);

    for (int i = 0; i < nprocs; i++) {
        if (i == myid) {
            continue;
        }

        matrixItem_t* currentBuffer = sendItemsBuffer.buffer + sendItemsBuffer.offset[i];
        for (int j = 0; j < rcvIndexBuffer.counter[i]; j++) {
            itype requestedRowIndex = rcvIndexBuffer.buffer[rcvIndexBuffer.offset[i] + j];
            itype actualIndex = requestedRowIndex2actualIndex[requestedRowIndex - dlA->row_shift];

            CHECK_DEVICE(
                cudaMemcpy(
                    currentBuffer,
                    d_nnzPerRowBuffer + h_nnzPerRowOffset[actualIndex],
                    h_nnzPerRowCounter[actualIndex] * sizeof(matrixItem_t),
                    cudaMemcpyDeviceToHost));

            currentBuffer += h_nnzPerRowCounter[actualIndex];
        }
    }

    if (f) {
        debugMatrixItems("sendItemsBuffer.buffer", sendItemsBuffer.buffer, sendItemsBuffer.size, false, f);
        debugArray("sendItemsBuffer.counter[%d] = %d\n", sendItemsBuffer.counter, sendItemsBuffer.nprocs, false, f);
        debugArray("sendItemsBuffer.offset[%d] = %d\n", sendItemsBuffer.offset, sendItemsBuffer.nprocs, false, f);
    }

    // Count the number of elements to be received by other processes and
    // prepare receiving MpiBuffer (set offsets, counters, and allocate memory)
    // ---------------------------------------------------------------------------
    MpiBuffer<matrixItem_t> rcvItemsBuffer;
    for (int i = 0; i < nprocs; i++) {
        rcvItemsBuffer.counter[i] = (i == myid)
            ? 0
            : rcvTotNnzBuffer.buffer[rcvTotNnzBuffer.offset[i]];

        rcvItemsBuffer.offset[i] = (i == 0)
            ? 0
            : rcvItemsBuffer.offset[i - 1] + rcvItemsBuffer.counter[i - 1];

        rcvItemsBuffer.size += rcvItemsBuffer.counter[i];
    }
    rcvItemsBuffer.init();

    // Exchange elements with other processes
    // ---------------------------------------------------------------------------
    CHECK_MPI(
        MPI_Alltoallv(
            sendItemsBuffer.buffer,
            sendItemsBuffer.counter,
            sendItemsBuffer.offset,
            MPI_MATRIX_ITEM_T,
            rcvItemsBuffer.buffer,
            rcvItemsBuffer.counter,
            rcvItemsBuffer.offset,
            MPI_MATRIX_ITEM_T,
            MPI_COMM_WORLD));

    if (f) {
        debugMatrixItems("rcvItemsBuffer.buffer", rcvItemsBuffer.buffer, rcvItemsBuffer.size, false, f);
        debugArray("rcvItemsBuffer.counter[%d] = %d\n", rcvItemsBuffer.counter, rcvItemsBuffer.nprocs, false, f);
        debugArray("rcvItemsBuffer.offset[%d] = %d\n", rcvItemsBuffer.offset, rcvItemsBuffer.nprocs, false, f);
    }

    // Release memory
    // ---------------------------------------------------------------------------
    Free(h_nnzPerRowCounter);
    Free(h_nnzPerRowOffset);
    CudaFree(d_nnzPerRowBuffer);
    CudaFree(d_nnzPerRowOffset);
    CudaFree(d_nnzPerRowCounter);
    CudaFree(d_requestedRowIndexesUnique);
    CudaFree(d_requestedRowIndexes);

    matrixItem_t* ret = rcvItemsBuffer.buffer;
    rcvItemsBuffer.buffer = NULL; // Avoid buffer to be free(d)
    *retSize = rcvItemsBuffer.size;
    return ret;
}
