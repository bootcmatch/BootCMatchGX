#include <assert.h>
#include <chrono>
#include <getopt.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#include "datastruct/CSR.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/distribuite.h"
#include "utility/globals.h"
#include "utility/handles.h"
#include "utility/mpi.h"

using namespace std;

// =============================================================================

#define USAGE                                                                                       \
    "\nUsage: %s --matrix <FILE_NAME> --out <FILE_NAME>\n\n"                                        \
    "\t-m, --matrix <FILE_NAME> Read the matrix from file <FILE_NAME>.\n"                           \
    "\t-l, --log <FILE_NAME> Write log files (one per MPI process) to <FILE_NAME>_<MPI_PID>.\n"     \
    "\t-o, --out <FILE_NAME> Write out files (one per MPI process) to <FILE_NAME>_<MPI_PID>.mtx.\n" \
    "\t-c, --col-shift Use column shift.\n"                                                         \
    "\t-f, --force-multiproc Force multiproc version even with 1 process.\n"                        \
    "\n"

// =============================================================================

int main(int argc, char** argv)
{
    char* mtx_file_name = NULL;
    char* log_file_name = NULL;
    char* out_file_name = NULL;
    bool use_column_shift = false;
    bool force_multiproc = false;
    int opt;

    static struct option long_options[] = {
        { "matrix", required_argument, NULL, 'm' },
        { "log", required_argument, NULL, 'l' },
        { "out", required_argument, NULL, 'o' },
        { "col-shift", no_argument, NULL, 'c' },
        { "force-multiproc", no_argument, NULL, 'f' },
        { "help", no_argument, NULL, 'h' },
    };

    while ((opt = getopt_long(argc, argv, "m:l:o:c:f:h", long_options, NULL)) != -1) {
        switch (opt) {
        case 'm':
            mtx_file_name = strdup(optarg);
            break;
        case 'l':
            log_file_name = strdup(optarg);
            break;
        case 'o':
            out_file_name = strdup(optarg);
            break;
        case 'c':
            use_column_shift = true;
            break;
        case 'f':
            force_multiproc = true;
            break;
        case 'h':
        default:
            printf(USAGE, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (mtx_file_name == NULL || out_file_name == NULL) {
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
        fprintf(stderr, "Read matrix: %s\n", mtx_file_name);
        hmA = read_matrix_from_file(mtx_file_name, 0, false);
        // check_and_fix_order(Alocal_master);
    }

    // Device local A
    CSR* dlA = split_matrix_mpi(hmA);
    // TODO check for errors

    if (ISMASTER) {
        CSRm::free(hmA);
    }

    if (use_column_shift && dlA->row_shift) {
        CSRm::shift_cols(dlA, -dlA->row_shift);
        dlA->col_shifted = -dlA->row_shift;
    }

    if (log_file_name) {
        char filename[255] = { 0 };
        sprintf(filename, "%s_%d", log_file_name, myid);
        log_file = fopen(filename, "w");
        if (log_file == NULL) {
            fprintf(stderr, "Error opening file <%s>\n", filename);
            exit(EXIT_FAILURE);
        }
        if (atexit(close_log_file)) {
            fprintf(stderr, "Error registering atexit\n");
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Transposing matrix... (force_multiproc: " << force_multiproc << ")\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    CSR* dlAt = (nprocs == 1 && !force_multiproc) ? CSRm::Transpose_local(dlA, log_file) : CSRm::transpose(dlA, log_file);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Operation took " << ms_double.count() << " ms.\n";
    // printf("dlAt->col_shifted: %d\n", dlAt->col_shifted);

    if (log_file) {
        CSRm::print(dlAt, 3, 0, log_file);
    }

    CSRm::printMM(dlAt, out_file_name);

    CSRm::free(dlA);
    CSRm::free(dlAt);

    if (log_file) {
        fflush(log_file);
        fclose(log_file);
        log_file = NULL;
    }

    Handles::free(h);
    MPI_Finalize();
    return 0;
}
