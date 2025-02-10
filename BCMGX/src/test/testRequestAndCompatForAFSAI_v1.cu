#include <assert.h>
#include <getopt.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#include "datastruct/CSR.h"
#include "datastruct/matrixItem.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/distribuite.h"
#include "utility/globals.h"
#include "utility/handles.h"
#include "utility/memory.h"
#include "utility/mpi.h"

using namespace std;

// =============================================================================

#define USAGE                                                                                   \
    "\nUsage: %s --matrix <FILE_NAME> \n\n"                                                     \
    "\t-m, --matrix <FILE_NAME> Read the matrix from file <FILE_NAME>.\n"                       \
    "\t-l, --log <FILE_NAME> Write log files (one per MPI process) to <FILE_NAME>_<MPI_PID>.\n" \
    "\n"

// =============================================================================

struct RowIndexShifter {
    const itype shift;

    __host__ __device__ __forceinline__ explicit RowIndexShifter(const itype shift)
        : shift(shift)
    {
    }

    __host__ __device__ __forceinline__
        itype
        operator()(itype a) const
    {
        return a + shift;
    }
};

int main(int argc, char** argv)
{
    char* mtx_file = NULL;
    char* log_file = NULL;
    int opt;
    int verbose = 0;

    static struct option long_options[] = {
        { "matrix", required_argument, NULL, 'm' },
        { "log", required_argument, NULL, 'l' },
        { "verbose", no_argument, NULL, 'v' },
        { "help", no_argument, NULL, 'h' }
    };

    while ((opt = getopt_long(argc, argv, "m:l:s:v:h", long_options, NULL)) != -1) {
        switch (opt) {
        case 'm':
            mtx_file = strdup(optarg);
            break;
        case 'l':
            log_file = strdup(optarg);
            break;
        case 'v':
            verbose = 1;
            break;
        case 'h':
        default:
            printf(USAGE, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (mtx_file == NULL || log_file == NULL) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    // Start MPI
    int myid, nprocs, device_id;
    StartMpi(&myid, &nprocs, &argc, &argv);

    // Set device
    int deviceCount = 0;
    CHECK_DEVICE(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount);
    device_id = assignDeviceToProcess();
    int assigned_device_id = device_id % deviceCount;
    fprintf(stderr, "Trying to set device %d. Total devices: %d. Assigned device: %d\n", device_id, deviceCount, assigned_device_id);
    CHECK_DEVICE(cudaSetDevice(assigned_device_id));

    handles* h = Handles::init();

    CSR* hmA = NULL; // Host master A
    if (ISMASTER) {
        fprintf(stderr, "Read matrix: %s\n", mtx_file);
        hmA = read_matrix_from_file(mtx_file, 0, false);
        // check_and_fix_order(Alocal_master);
    }

    // Device local A
    CSR* dlA = split_matrix_mpi(hmA);
    // TODO check for errors

    FILE* f = NULL;
    if (verbose) {
        char filename[255] = { 0 };
        sprintf(filename, "%s_%d", log_file, myid);
        f = fopen(filename, "w");
        if (f == NULL) {
            fprintf(stderr, "Error opening file <%s>\n", filename);
            exit(EXIT_FAILURE);
        }
        CSRm::print(dlA, 3, 0, f);
    }

    size_t missingItemsSize = 0;
    matrixItem_t* h_missingItems = NULL;

    size_t rowsToBeRequestedSize = 0;
    itype* d_rowsToBeRequested = NULL;
    MatrixItemColumnIndexLessThanSelector matrixItemSelector(
        dlA->row_shift);
    h_missingItems = CSRm::requestMissingRows(
        dlA,
        f,
        &missingItemsSize,
        NULL,
        &rowsToBeRequestedSize,
        &d_rowsToBeRequested,
        matrixItemSelector,
        NnzColumnSelector(),
        false);
    /*
    h_missingItems = CSRm::requestMissingRows(
      dlA,
      f,
      &missingItemsSize,
      NULL,
      &rowsToBeRequestedSize,
      &d_rowsToBeRequested,
      matrixItemSelector,
      EveryColumnSelector(dlA->row_shift)
    );
    */
    if (f) {
        debugArray("d_rowsToBeRequested[%d] = %d\n", d_rowsToBeRequested, rowsToBeRequestedSize, true, f);
    }
    CUDA_FREE(d_rowsToBeRequested);

    // ---------------------------------------------------------------------------

    matrixItem_t* d_missingItems = copyArrayToDevice(h_missingItems, missingItemsSize);
    FREE(h_missingItems);

    itype missingItemsN = dlA->row_shift;
    itype* d_missingItemsRow = NULL;
    itype* d_missingItemsCol = NULL;
    vtype* d_missingItemsVal = NULL;

    size_t missingItemsRowUniqueSize;
    itype* d_missingItemsRowUnique = NULL;

    if (missingItemsN > 0) {
        fillCsrFromMatrixItems(
            d_missingItems,
            missingItemsSize,
            missingItemsN,
            0, // row_shift
            &d_missingItemsRow,
            &d_missingItemsCol,
            &d_missingItemsVal,
            false, // Transposed,
            true // Allocate memory
        );

        if (f) {
            debugArray("d_missingItemsRow[%d] = %d\n", d_missingItemsRow, missingItemsN + 1, true, f);
            debugArray("d_missingItemsCol[%d] = %d\n", d_missingItemsCol, missingItemsSize, true, f);
            debugArray("d_missingItemsVal[%d] = %lf\n", d_missingItemsVal, missingItemsSize, true, f);
        }

        d_missingItemsRowUnique = deviceUnique(d_missingItemsRow, missingItemsN + 1, &missingItemsRowUniqueSize);

        if (f) {
            debugArray("d_missingItemsRowUnique[%d] = %d\n", d_missingItemsRowUnique, missingItemsRowUniqueSize, true, f);
        }
    }

    CUDA_FREE(d_missingItems);

    itype concatenatedNnz = missingItemsSize + dlA->nnz;
    itype concatenatedN = dlA->n;
    itype* d_concatenatedRow = dlA->row;
    itype* d_concatenatedCol = dlA->col;
    vtype* d_concatenatedVal = dlA->val;

    if (missingItemsN > 0) {
        concatenatedN = missingItemsRowUniqueSize - 1 + dlA->n;

        itype shift = 0;
        CHECK_DEVICE(cudaMemcpy(
            &shift,
            d_missingItemsRowUnique + missingItemsRowUniqueSize - 1,
            sizeof(itype),
            cudaMemcpyDeviceToHost));

        itype* d_shiftedRow = deviceMap<itype, itype, RowIndexShifter>(
            dlA->row,
            dlA->n + 1,
            RowIndexShifter(shift));

        d_concatenatedRow = concatArrays(
            d_missingItemsRowUnique, // First array
            missingItemsRowUniqueSize, // First array: len
            true, // First array: onDevice
            d_shiftedRow + 1, // Second array
            dlA->n, // Second array: len
            true, // Second array: onDevice
            true // Returned array: onDevice
        );

        CUDA_FREE(d_shiftedRow);

        d_concatenatedCol = concatArrays(
            d_missingItemsCol, // First array
            missingItemsSize, // First array: len
            true, // First array: onDevice
            dlA->col, // Second array
            dlA->nnz, // Second array: len
            true, // Second array: onDevice
            true // Returned array: onDevice
        );

        d_concatenatedVal = concatArrays(
            d_missingItemsVal, // First array
            missingItemsSize, // First array: len
            true, // First array: onDevice
            dlA->val, // Second array
            dlA->nnz, // Second array: len
            true, // Second array: onDevice
            true // Returned array: onDevice
        );

        if (f) {
            debugArray("d_concatenatedRow[%d] = %d\n", d_concatenatedRow, concatenatedN + 1, true, f);
            debugArray("d_concatenatedCol[%d] = %d\n", d_concatenatedCol, concatenatedNnz, true, f);
            debugArray("d_concatenatedVal[%d] = %lf\n", d_concatenatedVal, concatenatedNnz, true, f);
        }

        CUDA_FREE(d_concatenatedRow);
        CUDA_FREE(d_concatenatedCol);
        CUDA_FREE(d_concatenatedVal);
        CUDA_FREE(d_missingItemsRowUnique);
        CUDA_FREE(d_missingItemsRow);
        CUDA_FREE(d_missingItemsCol);
        CUDA_FREE(d_missingItemsVal);
    }

    // ---------------------------------------------------------------------------

    CSRm::free(dlA);

    if (f) {
        fflush(f);
        fclose(f);
    }

    Handles::free(h);
    MPI_Finalize();
    return 0;
}
