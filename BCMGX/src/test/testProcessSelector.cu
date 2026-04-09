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
#include "utility/distribute.h"
#include "utility/globals.h"
#include "utility/handles.h"
#include "utility/input.h"
#include "utility/mpi.h"

using namespace std;

// =============================================================================

#define USAGE                                                                                                               \
    "\nUsage: %s [--matrix <FILE_NAME> | --laplacian <SIZE> | --laplacian-3d <FILE_NAME>] --out <FILE_NAME>\n\n"            \
    "\t-e, --log <FILE_NAME>                       Write log files (one per MPI process) to <FILE_NAME>_<MPI_PID>.\n"       \
    "\t-g, --laplacian-3d-generator [ 7p | 27p ]   Choose laplacian 3d generator (7 points or 27 points).\n"                \
    "\t-l, --laplacian-3d <FILE_NAME>              Read generation parameters from file <FILE_NAME>.\n"                     \
    "\t-m, --matrix <FILE_NAME>                    Read the matrix from file <FILE_NAME>.\n"                                \
    "\n"

// =============================================================================

int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Command line options
    // -------------------------------------------------------------------------

    enum opts { MTX, LAP_3D, NONE } opt = NONE;

    char* mtx_file_name = NULL;
    char* log_file_name = NULL;
    char* lap_3d_file = NULL;
    generator_t generator = LAP_27P;
    signed char ch = 0;

    static struct option long_options[] = {
        { "log", required_argument, NULL, 'e' },
        { "laplacian-3d-generator", required_argument, NULL, 'g' },
        { "help", no_argument, NULL, 'h' },
        { "laplacian-3d", required_argument, NULL, 'l' },
        { "matrix", required_argument, NULL, 'm' },
    };

    while ((ch = getopt_long(argc, argv, "e:g:hl:m:", long_options, NULL)) != -1) {
        switch (ch) {
        case 'e':
            log_file_name = strdup(optarg);
            break;
        case 'g':
            generator = get_generator(optarg);
            break;
        case 'l':
            lap_3d_file = strdup(optarg);
            opt = LAP_3D;
            break;
        case 'm':
            mtx_file_name = strdup(optarg);
            opt = MTX;
            break;
        case 'h':
        default:
            DIE(USAGE, argv[0]);
        }
    }

    if (opt == NONE || generator == INVALIG_GEN) {
        DIE(USAGE, argv[0]);
    }

    // -------------------------------------------------------------------------
    // Initialize MPI
    // -------------------------------------------------------------------------

    // Start MPI
    int myid, nprocs, device_id;
    StartMpi(&myid, &nprocs, &argc, &argv);

    // -------------------------------------------------------------------------
    // Initialize logging
    // -------------------------------------------------------------------------

    if (log_file_name) {
        open_log_file(myid, log_file_name);
    }

    // -------------------------------------------------------------------------
    // Assign GPU
    // -------------------------------------------------------------------------

    int deviceCount = 0;
    CHECK_DEVICE(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount);
    device_id = assignDeviceToProcess();
    int assigned_device_id = device_id % deviceCount;
    fprintf(stderr, "Trying to set device %d. Total devices: %d. Assigned device: %d\n", device_id, deviceCount, assigned_device_id);
    CHECK_DEVICE(cudaSetDevice(assigned_device_id));

    // -------------------------------------------------------------------------
    // Read/Generate input matrix
    // -------------------------------------------------------------------------

    CSR* dlA = NULL;
    if (opt == MTX) { // The master reads the matrix and distributes it.
        dlA = read_local_matrix_from_mtx(mtx_file_name);
    } else if (opt == LAP_3D) {
        dlA = generate_lap3d_local_matrix(generator, lap_3d_file);
    }

    handles* h = Handles::init();

    ProcessSelector processSelector(dlA, log_file);
    ProcessSelector processSelectorNew(dlA->full_n, log_file);

    CSRm::free(dlA);

    if (log_file) {
        fflush(log_file);
        fclose(log_file);
        log_file = NULL;
    }

    Handles::free(h);
    MPI_Finalize();
    return 0;
}
