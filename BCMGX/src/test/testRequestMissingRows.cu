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
    "\t-s, --selector [lower|upper|both] Specify which columns to consider.\n"                  \
    "\t-c, --col-shift Use column shift.\n"                                                     \
    "\n"

// =============================================================================

enum selector_t {
    INVALID_SELECTOR,
    LOWER_SELECTOR,
    UPPER_SELECTOR,
    BOTH_SELECTOR
};

selector_t getSelector(const char* str)
{
    if (!strcmp(str, "lower")) {
        return LOWER_SELECTOR;
    } else if (!strcmp(str, "upper")) {
        return UPPER_SELECTOR;
    }
    if (!strcmp(str, "both")) {
        return BOTH_SELECTOR;
    } else {
        return INVALID_SELECTOR;
    }
}

int main(int argc, char** argv)
{
    char* mtx_file = NULL;
    char* log_file = NULL;
    selector_t selector = INVALID_SELECTOR;
    int opt;
    int verbose = 0;
    bool use_column_shift = false;

    static struct option long_options[] = {
        { "matrix", required_argument, NULL, 'm' },
        { "log", required_argument, NULL, 'l' },
        { "selector", required_argument, NULL, 's' },
        { "col-shift", no_argument, NULL, 'c' },
        { "verbose", no_argument, NULL, 'v' },
        { "help", no_argument, NULL, 'h' }
    };

    while ((opt = getopt_long(argc, argv, "m:l:s:c:v:h", long_options, NULL)) != -1) {
        switch (opt) {
        case 'm':
            mtx_file = strdup(optarg);
            break;
        case 'l':
            log_file = strdup(optarg);
            break;
        case 's':
            selector = getSelector(optarg);
            break;
        case 'c':
            use_column_shift = true;
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

    if (selector == INVALID_SELECTOR) {
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

    if (use_column_shift && dlA->row_shift) {
        CSRm::shift_cols(dlA, -dlA->row_shift);
        dlA->col_shifted = -dlA->row_shift;
    }

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
    matrixItem_t* d_missingItems = NULL;
    matrixItem_t* d_nnzItems = NULL;

    switch (selector) {
    case LOWER_SELECTOR: {
        MatrixItemColumnIndexLessThanSelector matrixItemSelector(
            use_column_shift ? 0 : dlA->row_shift);
        h_missingItems = CSRm::requestMissingRows(dlA, f, &missingItemsSize, &d_nnzItems, NULL, NULL, matrixItemSelector, NnzColumnSelector(), use_column_shift);
        break;
    }

    case UPPER_SELECTOR: {
        MatrixItemColumnIndexGreaterThanSelector matrixItemSelector(
            use_column_shift ? dlA->n - 1 : dlA->row_shift + dlA->n - 1);
        h_missingItems = CSRm::requestMissingRows(dlA, f, &missingItemsSize, &d_nnzItems, NULL, NULL, matrixItemSelector, NnzColumnSelector(), use_column_shift);
        break;
    }

    default: {
        MatrixItemColumnIndexOutOfBoundsSelector matrixItemSelector(
            use_column_shift ? 0 : dlA->row_shift,
            use_column_shift ? dlA->n - 1 : dlA->row_shift + dlA->n - 1);
        h_missingItems = CSRm::requestMissingRows(dlA, f, &missingItemsSize, &d_nnzItems, NULL, NULL, matrixItemSelector, NnzColumnSelector(), use_column_shift);
    }
    }

    if (use_column_shift) {
        d_missingItems = copyArrayToDevice(h_missingItems, missingItemsSize);
        /*
        deviceForEach<matrixItem_t, ReceivedMatrixItemAdjuster>(
            d_missingItems,
            missingItemsSize,
            ReceivedMatrixItemAdjuster(dlA->row_shift, dlA->full_n, nprocs)
        );
        */
    }

    size_t concatenatedSize = dlA->nnz + missingItemsSize;
    matrixItem_t* d_concatenated = NULL;

    switch (selector) {
    // Previously missing items belongs to rows with index < dlA->row_shift,
    // hence prefixing them to the ones locally available is enough in order
    // not to sort them.
    case LOWER_SELECTOR: {
        d_concatenated = concatArrays(
            d_missingItems ? d_missingItems : h_missingItems, // First array
            missingItemsSize, // First array: len
            d_missingItems ? true : false, // First array: onDevice
            d_nnzItems, // Second array
            dlA->nnz, // Second array: len
            true, // Second array: onDevice
            true // Returned array: onDevice
        );
        break;
    }

    // Previously missing items belongs to rows with
    // index > dlA->row_shift + dlA->n - 1,
    // hence suffixing them to the ones locally available is enough in order
    // not to sort them.
    case UPPER_SELECTOR: {
        d_concatenated = concatArrays(
            d_nnzItems, // First array
            dlA->nnz, // First array: len
            true, // First array: onDevice
            d_missingItems ? d_missingItems : h_missingItems, // Second array
            missingItemsSize, // Second array: len
            d_missingItems ? true : false, // Second array: onDevice
            true // Returned array: onDevice
        );
        break;
    }

    // Previously missing items belongs to rows with either
    // index < dlA->row_shift or index > dlA->row_shift + dlA->n - 1,
    // hence items must be sorted.
    case BOTH_SELECTOR: {
        d_concatenated = concatArrays(
            d_nnzItems, // First array
            dlA->nnz, // First array: len
            true, // First array: onDevice
            d_missingItems ? d_missingItems : h_missingItems, // Second array
            missingItemsSize, // Second array: len
            d_missingItems ? true : false, // Second array: onDevice
            true // Returned array: onDevice
        );

        deviceSort<matrixItem_t, gstype, MatrixItemComparator>(d_concatenated, concatenatedSize, MatrixItemComparator(dlA->m));
        break;
    }
    }

    if (verbose) {
        debugMatrixItems("h_missingItems", h_missingItems, missingItemsSize, false, f);
        debugMatrixItems("nnzItems (returned)", d_nnzItems, dlA->nnz, true, f);
        debugMatrixItems("d_concatenated", d_concatenated, concatenatedSize, true, f);
    }

    // Release memory
    CUDA_FREE(d_concatenated);
    CUDA_FREE(d_nnzItems);

    CSRm::free(dlA);
    FREE(h_missingItems);
    CUDA_FREE(d_missingItems);

    if (f) {
        fflush(f);
        fclose(f);
    }

    Handles::free(h);
    MPI_Finalize();
    return 0;
}
