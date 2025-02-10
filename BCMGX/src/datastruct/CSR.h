/**
 * @file
 */
#pragma once

#include "utility/globals.h"
#include "utility/memory.h"
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

#include "halo_communication/newoverlap.h"

#include <mpi.h>

 /**
 * @struct halo_info
 * @brief Structure to hold information about halo communication in a distributed matrix.
 */
struct halo_info {
    vector<gstype>* to_receive = NULL; ///< Pointer to vector of values to receive.
    vector<gstype>* to_receive_d = NULL; ///< Pointer to device vector of values to receive.
    int to_receive_n = 0; ///< Number of values to receive.
    int* to_receive_counts = NULL; ///< Array of counts of values to receive from each process.
    int* to_receive_spls = NULL; ///< Array of starting indices for received values.
    vtype* what_to_receive = NULL; ///< Pointer to values to receive.
    vtype* what_to_receive_d = NULL; ///< Pointer to device values to receive.

    vector<itype>* to_send = NULL; ///< Pointer to vector of values to send.
    vector<itype>* to_send_d = NULL; ///< Pointer to device vector of values to send.
    int to_send_n = 0; ///< Number of values to send.
    int* to_send_counts = NULL; ///< Array of counts of values to send to each process.
    int* to_send_spls = NULL; ///< Array of starting indices for sent values.
    vtype* what_to_send = NULL; ///< Pointer to values to send.
    vtype* what_to_send_d = NULL; ///< Pointer to device values to send.

    bool init = false; ///< Flag indicating if halo_info has been initialized.
};

/**
 * @struct rows_info
 * @brief Structure to hold information about rows to be received in a distributed matrix.
 */
typedef struct rows_info {
    vector<itype>* nnz_per_row_shift = NULL; ///< Pointer to vector of non-zero counts per row.
    itype rows2bereceived = 0; ///< Number of rows to be received.
    itype countall = 0; ///< Total count of elements.
    itype* P_n_per_process = NULL; ///< Array of counts of rows per process.
    int* scounts = NULL; ///< Array of send counts.
    int* displr = NULL; ///< Array of displacements for received rows.
    int* displs = NULL; ///< Array of displacements for sent rows.
    int* rcounts2 = NULL; ///< Array of counts of received rows.
    int* scounts2 = NULL; ///< Array of send counts for second communication.
    int* displr2 = NULL; ///< Array of displacements for received rows in second communication.
    int* displs2 = NULL; ///< Array of displacements for sent rows in second communication.
    unsigned int* rcvcntp = NULL; ///< Pointer to receive counts.
    itype* rcvprow = NULL; ///< Pointer to received row indices.
    gstype* whichprow = NULL; ///< Pointer to row indices.
    itype* rcvpcolxrow = NULL; ///< Pointer to received column indices per row.
} rows_to_get_info;

/**
 * @def PRINTM(x)
 * @brief Macro to print the CSR matrix.
 * @param x The CSR matrix to print.
 */
#define PRINTM(x)         \
    CSRm::print(x, 0, 0); \
    CSRm::print(x, 1, 0); \
    CSRm::print(x, 2, 0);

/**
 * @struct overlappedList
 * @brief Structure to hold CUDA streams for overlapped communication.
 */
struct overlappedList {
    cudaStream_t* local_stream = NULL; ///< Pointer to local CUDA stream.
    cudaStream_t* comm_stream = NULL; ///< Pointer to communication CUDA stream.
};

/**
 * @struct overlapped
 * @brief Structure to hold information about overlapped communication.
 */
struct overlapped {
    vector<itype>* loc_rows = NULL; ///< Pointer to local rows.
    itype loc_n = 0; ///< Number of local rows 

    vector<itype>* needy_rows = NULL; ///< Pointer to rows that need to be received.
    itype needy_n = 0; ///< Number of needy rows.

    struct overlappedList* streams = NULL; ///< Pointer to the list of CUDA streams for communication.
};

/**
 * @struct CSR
 * @brief Structure representing a custom, distributed, compressed sparse row (CSR) matrix.
 */
struct CSR {
    /**
     * @var nnz
     * @brief Number of non-zero elements in the matrix.
     */
    stype nnz = 0; 

    /**
     * @var n
     * @brief Number of rows in the local portion of the matrix.
     */
    stype n = 0;

    /**
     * @var m
     * @brief Number of columns in the local portion of the matrix.
     */
    gstype m = 0;

    /**
     * @var shrinked_m
     * @brief Number of columns in the shrunk matrix.
     */
    stype shrinked_m = 0;

    /**
     * @var full_n
     * @brief Total number of rows in the global matrix.
     */
    gstype full_n = 0;

    /**
     * @var full_m
     * @brief Total number of columns in the global matrix.
     */
    gstype full_m = 0;

    /**
     * @var on_the_device
     * @brief Flag indicating if the CSR matrix is allocated on the device (GPU).
     */
    bool on_the_device = false;

    /**
     * @var is_symmetric
     * @brief Flag indicating if the CSR matrix is symmetric.
     */
    bool is_symmetric = false;

    /**
     * @var shrinked_flag
     * @brief Flag indicating if the CSR matrix is shrunk.
     */
    bool shrinked_flag = false;

    /**
     * @var custom_alloced
     * @brief Flag indicating if the CSR matrix was manually allocated.
     */
    bool custom_alloced = false;

    /**
     * @var row_shift
     * @brief Row shift for identifying the portion of the matrix assigned to a process.
     */
    gstype row_shift = 0;

    /**
     * @var col_shifted
     * @brief Column shift for the CSR matrix.
     */
    gsstype col_shifted = 0;

    /**
     * @var shrinked_firstrow
     * @brief First row index of the shrunk matrix.
     */
    gstype shrinked_firstrow = 0;

    /**
     * @var shrinked_lastrow
     * @brief Last row index of the shrunk matrix.
     */
    gstype shrinked_lastrow = 0;

    /**
     * @var val
     * @brief Pointer to the non-zero values of the matrix.
     */
    vtype* val = NULL;

    /**
     * @var col
     * @brief Pointer to the column indices of the non-zero values.
     */
    itype* col = NULL;

    /**
     * @var row
     * @brief Pointer to the row pointers for the CSR matrix.
     */
    itype* row = NULL;

    /**
     * @var shrinked_col
     * @brief Pointer to the column indices of the shrunk matrix.
     */
    itype* shrinked_col = NULL;

    int* bitcol = NULL; ///< Pointer to bit column data.
    int bitcolsize = 0; ///< Size of the bit column data.
    int post_local = 0; ///< Post-local data.

    halo_info halo; ///< Halo information for the CSR matrix.

    rows_to_get_info* rows_to_get = NULL; ///< Information about rows to be received.

    struct overlapped os; ///< Overlapped communication structure.
};

struct matrixItem_t;

/**
 * Function object to be used when calling CSRm::requestMissingRows to select only
 * the indexes of the columns related to non-zero values in the local CSR.
 */
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

/**
 * Function object to be used when calling CSRm::requestMissingRows to select
 * all the column indexes.
 */
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

        itype* ret = CUDA_MALLOC(itype, row_shift);

        deviceForEach(ret, row_shift, FillWithIndexOperator<itype>());

        return ret;
    }
};

/**
 * @namespace CSRm
 * @brief Namespace containing functions for CSR matrix operations.
 */
namespace CSRm {

/**
 * @brief Initializes a CSR matrix.
 * @param n Number of rows in the local matrix.
 * @param m Number of columns in the local matrix.
 * @param nnz Number of non-zero elements in the matrix.
 * @param allocate_mem Flag indicating whether to allocate memory.
 * @param on_the_device Flag indicating if the matrix should be allocated on the device.
 * @param is_symmetric Flag indicating if the matrix is symmetric.
 * @param full_n Total number of rows in the global matrix.
 * @param row_shift Row shift for the CSR matrix.
 * @return Pointer to the initialized CSR matrix.
 */
CSR* init(stype n, gstype m, stype nnz, bool allocate_mem, bool on_the_device, bool is_symmetric, gstype full_n, gstype row_shift = 0);

/**
 * @brief Frees the memory allocated for a CSR matrix.
 * @param A Pointer to the CSR matrix to be freed.
 */
void free(CSR* A);

/**
 * @brief Prints information about a CSR matrix.
 * @param A Pointer to the CSR matrix.
 * @param fp File pointer for output (default is stdout).
 */
void printInfo(CSR* A, FILE* fp = stdout);

/**
 * @brief Chooses an appropriate mini-warp size based on the matrix density.
 * @param A Pointer to the CSR matrix.
 * @return Selected mini-warp size.
 */
int choose_mini_warp_size(CSR* A);

/**
 * @brief Frees the rows_to_get_info structure associated with a CSR matrix.
 * @param A Pointer to the CSR matrix.
 */
void free_rows_to_get(CSR* A);

/**
 * @brief Prints the CSR matrix.
 * @param A Pointer to the CSR matrix.
 * @param type Type of print (0, 1, or 2).
 * @param limit Limit for printing.
 * @param fp File pointer for output (default is stdout).
 */
void print(CSR* A, int type, int limit = 0, FILE* fp = stdout);

/**
 * @brief Prints the CSR matrix in Matrix Market format.
 * @param A Pointer to the CSR matrix.
 * @param name Name of the output file.
 * @param appendMyIdAndNprocs Flag indicating whether to append process ID and number of processes.
 */
void printMM(CSR* A, char* name, bool appendMyIdAndNprocs = true);

/**
 * @brief Copies a CSR matrix from host to device.
 * @param A Pointer to the CSR matrix on the host.
 * @return Pointer to the CSR matrix on the device.
 */
CSR* copyToDevice(CSR* A);

/**
 * @brief Copies a CSR matrix from device to host.
 * @param A_d Pointer to the CSR matrix on the device.
 * @return Pointer to the CSR matrix on the host.
 */
CSR* copyToHost(CSR* A_d);

/**
 * @brief Computes the diagonal of a CSR matrix.
 * @param A Pointer to the CSR matrix.
 * @return Pointer to a vector containing the diagonal elements.
 */
vector<vtype>* diag(CSR* A);

/**
 * @brief Transposes a local CSR matrix.
 * @param dlA Pointer to the local CSR matrix.
 * @param f File pointer for logging.
 * @return Pointer to the transposed CSR matrix.
 * @note Transpose_local is used only in matchingAggregation/matchingPairAggragation.cu
 * @note The transposed matrix has row_shift = 0 and column_shifted = 0
 *          even if dlA is row/col shifted.
 */
CSR* Transpose_local(CSR* dlA, FILE* f);

/**
 * @brief Transposes a CSR matrix.
 * @param A Pointer to the CSR matrix.
 * @param f File pointer for logging.
 * @param shape Shape of the matrix (default is "Q").
 * @return Pointer to the transposed CSR matrix.
 */
CSR* transpose(CSR* A, FILE* f, const char* shape = "Q");

/**
 * @brief Computes the absolute row sum of a CSR matrix.
 * @param A Pointer to the CSR matrix.
 * @param sum Pointer to a vector to store the row sums.
 * @return Pointer to the vector containing the absolute row sums.
 */
vector<vtype>* absoluteRowSum(CSR* A, vector<vtype>* sum);

/**
 * @brief Collects matrix items from a local CSR matrix.
 * @param dlA Pointer to the local CSR matrix.
 * @param debug File pointer for logging.
 * @param useColShift Flag indicating whether to use column shift.
 * @return Pointer to an array of matrix items.
 */
matrixItem_t* collectMatrixItems(CSR* dlA, FILE* debug, bool useColShift = false);

/**
 * @brief Collects matrix items from a local CSR matrix without using GPU.
 * @param dlA Pointer to the local CSR matrix.
 * @param debug File pointer for logging.
 * @param useColShift Flag indicating whether to use column shift.
 * @return Pointer to an array of matrix items.
 */
matrixItem_t* collectMatrixItems_nogpu(CSR* dlA, FILE* debug, bool useColShift = false);

/**
 * @brief Performs a matrix-vector product using adaptive mini-warp strategy.
 * @param A Pointer to the CSR matrix.
 * @param x Pointer to the input vector.
 * @param y Pointer to the output vector.
 * @param alpha Scalar multiplier for the matrix.
 * @param beta Scalar multiplier for the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* CSRVector_product_adaptive_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, vtype alpha = 1., vtype beta = 0.);

/**
 * @brief Performs a matrix-vector product using adaptive indirect row mini-warp strategy.
 * @param A Pointer to the CSR matrix.
 * @param x Pointer to the input vector.
 * @param y Pointer to the output vector.
 * @param n Number of rows.
 * @param rows Pointer to the rows to be processed.
 * @param stream CUDA stream for asynchronous execution.
 * @param offset Offset for the output vector.
 * @param alpha Scalar multiplier for the matrix.
 * @param beta Scalar multiplier for the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* CSRVector_product_adaptive_indirect_row_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, itype n, itype* rows, cudaStream_t stream, unsigned int offset, vtype alpha = 1., vtype beta = 0.);

/**
 * @brief Performs a shifted matrix-vector product using adaptive mini-warp strategy.
 * @param A Pointer to the CSR matrix.
 * @param x Pointer to the input vector.
 * @param y Pointer to the output vector.
 * @param shift Shift value for the matrix.
 * @param alpha Scalar multiplier for the matrix.
 * @param beta Scalar multiplier for the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* shifted_CSRVector_product_adaptive_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, itype shift, vtype alpha = 1., vtype beta = 0.);

/**
 * @brief Performs a shifted matrix-vector product using adaptive mini-warp strategy (alternative).
 * @param A Pointer to the CSR matrix.
 * @param x Pointer to the input vector.
 * @param y Pointer to the output vector.
 * @param shift Shift value for the matrix.
 * @param alpha Scalar multiplier for the matrix.
 * @param beta Scalar multiplier for the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* shifted_CSRVector_product_adaptive_miniwarp2(CSR* A, vector<vtype>* x, vector<vtype>* y, itype shift, vtype alpha = 1., vtype beta = 0.);

/**
 * @brief Performs a matrix-vector product using a prolongator.
 * @param A Pointer to the CSR matrix.
 * @param x Pointer to the input vector.
 * @param y Pointer to the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* CSRVector_product_prolungator(CSR* A, vector<vtype>* x, vector<vtype>* y);

/**
 * @brief Performs a matrix-vector product using adaptive mini-warp strategy (new version).
 * @param A Pointer to the CSR matrix.
 * @param local_x Pointer to the local input vector.
 * @param w Pointer to the output vector.
 * @param alpha Scalar multiplier for the matrix.
 * @param beta Scalar multiplier for the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* CSRVector_product_adaptive_miniwarp_new(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha = 1., vtype beta = 0.);

/**
 * @brief Performs a matrix-vector product using adaptive mini-warp strategy without certain optimizations.
 * @param A Pointer to the CSR matrix.
 * @param local_x Pointer to the local input vector.
 * @param w Pointer to the output vector.
 * @param alpha Scalar multiplier for the matrix.
 * @param beta Scalar multiplier for the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* CSRVector_product_adaptive_miniwarp_witho(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha = 1., vtype beta = 0.);

/**
 * @brief Performs a matrix-vector product using MPI.
 * @param Alocal Pointer to the local CSR matrix.
 * @param x Pointer to the input vector.
 * @param type Type of operation.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* CSRVector_product_MPI(CSR* Alocal, vector<vtype>* x, int type);

/**
 * @brief Scales a CSR matrix using adaptive mini-warp strategy.
 * @param A Pointer to the CSR matrix.
 * @param x Pointer to the input vector.
 * @param y Pointer to the output vector.
 * @param alpha Scalar multiplier for the matrix.
 * @param beta Scalar multiplier for the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* CSRscale_adaptive_miniwarp(CSR* A, vector<vtype>* x, vector<vtype>* y, vtype alpha = 1., vtype beta = 0.);

/**
 * @brief Scales a CSR matrix with a specific operation.
 * @param A Pointer to the CSR matrix.
 * @param local_x Pointer to the local input vector.
 * @param w Pointer to the output vector.
 * @param alpha Scalar multiplier for the matrix.
 * @param beta Scalar multiplier for the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* CSRscaleA_0(CSR* A, vector<vtype>* local_x, vector<vtype>* w, vtype alpha = 1., vtype beta = 0.);

/**
 * @brief Scales a CSR matrix with an in-place operation.
 * @param A Pointer to the CSR matrix.
 * @param local_x Pointer to the local input vector.
 * @param alpha Scalar multiplier for the matrix.
 * @param beta Scalar multiplier for the output vector.
 * @return Pointer to the resulting vector.
 */
vector<vtype>* CSRscaleA_0IP(CSR* A, vector<vtype>* local_x, vtype alpha = 1., vtype beta = 0.);

/**
 * @brief Checks the integrity of a CSR matrix.
 * @param A_ Pointer to the CSR matrix.
 * @param check_diagonal Flag indicating whether to check the diagonal.
 */
void checkMatrix(CSR* A_, bool check_diagonal = false);

/**
 * @brief Checks the order of columns in a CSR matrix.
 * @param A Pointer to the CSR matrix.
 */
void checkColumnsOrder(CSR* A);

/**
 * @brief Shifts the columns of a CSR matrix.
 * @param A Pointer to the CSR matrix.
 * @param shift Amount to shift the columns.
 */
void shift_cols(CSR* A, gsstype shift);

/**
 * @brief Shifts the columns of a CSR matrix without using GPU.
 * @param A Pointer to the CSR matrix.
 * @param shift Amount to shift the columns.
 */
void shift_cols_nogpu(CSR* A, gsstype shift);

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

/**
 * @brief Requests missing rows from other MPI processes based on specified criteria.
 * @tparam MatrixItemSelector Type of the selector for matrix items.
 * @tparam ColumnSelector Type of the selector for column indexes.
 * @param dlA Pointer to the local CSR matrix.
 * @param f File pointer for logging.
 * @param retSize Pointer to store the size of the returned array.
 * @param d_nnzItemsRet Pointer to store the returned array of non-zero matrix items.
 * @param rowsToBeRequestedSizeRet Pointer to store the size of requested rows.
 * @param d_rowsToBeRequestedRet Pointer to store the array of requested row indexes.
 * @param matrixItemSelector Selector for matrix items.
 * @param columnSelector Selector for column indexes.
 * @param useRowShift Flag indicating whether to use row shift.
 * @return Pointer to the array of matrix items.
 */
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
        fprintf(f, "m (cols) : %lu\n", dlA->m);
        fprintf(f, "nnz      : %d\n", dlA->nnz);
        fprintf(f, "row shift: %lu\n", dlA->row_shift);
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
        fprintf(f, "nnzItemsToBeRequested effective size: %zu\n", nnzItemsToBeRequestedSize);
        debugMatrixItems("nnzItemsToBeRequested", d_nnzItemsToBeRequested, nnzItemsToBeRequestedSize, true, f);
    }

    // Release memory or return nnzItems
    // ---------------------------------------------------------------------------
    if (d_nnzItemsRet) {
        *d_nnzItemsRet = d_nnzItems;
    } else {
        CUDA_FREE(d_nnzItems);
    }

    // Extract column indexes (** REQUESTS **)
    // ---------------------------------------------------------------------------
    size_t columnsToBeRequestedSize = 0;
    itype* d_columnsToBeRequested = columnSelector(
        d_nnzItemsToBeRequested, nnzItemsToBeRequestedSize, &columnsToBeRequestedSize);

    // Release memory
    // ---------------------------------------------------------------------------
    CUDA_FREE(d_nnzItemsToBeRequested);

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
    CUDA_FREE(d_columnsToBeRequested);
    if (!d_rowsToBeRequestedRet) {
        CUDA_FREE(d_columnsToBeRequestedUnique);
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
    FREE(h_columnsToBeRequestedUnique);

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
        d_nnzPerRowCounter = CUDA_MALLOC(itype, requestedRowIndexesUniqueSize, true);
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
    itype* d_nnzPerRowOffset = CUDA_MALLOC(itype, requestedRowIndexesUniqueSize + 1, true);
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
    if (nnzPerRowBufferSize) {
        d_nnzPerRowBuffer = CUDA_MALLOC(matrixItem_t, nnzPerRowBufferSize, true);
    }

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
    FREE(h_requestedRowIndexesUnique);

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
    FREE(h_nnzPerRowCounter);
    FREE(h_nnzPerRowOffset);
    CUDA_FREE(d_nnzPerRowBuffer);
    CUDA_FREE(d_nnzPerRowOffset);
    CUDA_FREE(d_nnzPerRowCounter);
    CUDA_FREE(d_requestedRowIndexesUnique);
    CUDA_FREE(d_requestedRowIndexes);

    matrixItem_t* ret = rcvItemsBuffer.buffer;
    rcvItemsBuffer.buffer = NULL; // Avoid buffer to be free(d)
    *retSize = rcvItemsBuffer.size;
    return ret;
}
