#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <getopt.h>

#include "datastruct/CSR.h"
#include "halo_communication/halo_communication.h"
#include "halo_communication/local_permutation.h"
#include "op/spspmpi.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/col8.h"
#include "utility/distribute.h"
#include "utility/globals.h"
#include "utility/handles.h"
#include "utility/input.h"
#include "utility/mpi.h"

#define DEBUG 1

#define USAGE                                                                                                \
    "\nUsage: %s --log <FILE_NAME>\n"                                                                        \
    "\t-e, --log <FILE_NAME>            Write log files (one per MPI process) to <FILE_NAME>_<MPI_PID>.\n"   \
    "\n"

#define GLOB_MEM_ALLOC_SIZE 2000000

extern itype *iAtemp1;
extern vtype *vAtemp1;
extern itype *idevtemp1;
extern vtype *vdevtemp1;
extern itype *idevtemp2;

int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Command line parameters
    // -------------------------------------------------------------------------

    char* log_file_name = NULL;

    static struct option long_options[] = {
        { "log", required_argument, NULL, 'e' },
    };

    char ch = 0;
    while ((ch = getopt_long(argc, argv, "e:h", long_options, NULL)) != -1) {
        switch (ch) {
        case 'e':
            log_file_name = strdup(optarg);
            break;
        case 'h':
        default:
            DIE(USAGE, argv[0]);
        }
    }

    // -------------------------------------------------------------------------
    // Initialize
    // -------------------------------------------------------------------------

    BCM::init(&argc, &argv, log_file_name);
    trace_enabled = true;
    _MPI_ENV;

    // -------------------------------------------------------------------------
    // Preparation
    // -------------------------------------------------------------------------

    handles* h = Handles::init();

    // -------------------------------------------------------------------------
    // Init data
    // -------------------------------------------------------------------------

    if (nprocs == 4) {
        taskmap = MALLOC(int, 4);
        itaskmap = MALLOC(int, 4);
        taskmap[0] = 0; itaskmap[0] = 0;
        taskmap[1] = 2; itaskmap[1] = 2;
        taskmap[2] = 1; itaskmap[2] = 1;
        taskmap[3] = 3; itaskmap[3] = 3;
    }

    TRACE("A");
    CSR *A = CSRm::createTestMatrix(5, {
        1, 1, 1, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 1, 1, 0,
    });

    fprintf(log_file, "Alocal:\n");
    CSRm::print(A, 3, -1, log_file);

    if (ISMASTER) fprintf(stderr, "A:\n");
    CSRm::debug(A, stderr);

    TRACE("P");
    CSR *P = CSRm::createTestMatrix(4, {
        1,  0,  0,  0,
        2,  3,  4,  0,
        0,  0,  5,  6,
        7,  0,  8,  9,
        0, 10, 11, 12
    });

    fprintf(log_file, "Plocal:\n");
    CSRm::print(P, 3, -1, log_file);

    if (ISMASTER) fprintf(stderr, "P:\n");
    CSRm::debug(P, stderr);

    // -------------------------------------------------------------------------

    iAtemp1 = CUDA_MALLOC_HOST(itype, GLOB_MEM_ALLOC_SIZE, true);
    vAtemp1 = CUDA_MALLOC_HOST(vtype, GLOB_MEM_ALLOC_SIZE, true);
    idevtemp1 = CUDA_MALLOC(itype, GLOB_MEM_ALLOC_SIZE, true);
    vdevtemp1 = CUDA_MALLOC(vtype, GLOB_MEM_ALLOC_SIZE, true);
    idevtemp2 = CUDA_MALLOC(itype, GLOB_MEM_ALLOC_SIZE, true);

    // -------------------------------------------------------------------------

    CSR *AP = CSRm::product(h, A, P);    

    // -------------------------------------------------------------------------

    CUDA_FREE_HOST(iAtemp1);
    CUDA_FREE_HOST(vAtemp1);
    CUDA_FREE(idevtemp1);
    CUDA_FREE(vdevtemp1);
    CUDA_FREE(idevtemp2);

    // -------------------------------------------------------------------------
    
    if (ISMASTER) {
        fprintf(stderr, "Computed result:\n");
    }
    CSRm::debug(AP, stderr);

    ASSERT(CSRm::checkProduct(h, A, P, AP));

    // if (!CSRm::eqGlobal(AP, expectedAP, "AP", "expectedAP")) {
    //     if (ISMASTER) {
    //         fprintf(stderr, "Ops, it doesn't seem to work properly.\n");
    //         // fprintf(stderr, "Expected result:\n");
    //         // CSRm::debug(expectedAP, stderr);
    //     }
    // } else {
    //     if (ISMASTER) {
    //         fprintf(stderr, "Wow, it seems to work properly.\n");
    //     }
    // }

    CSRm::free(AP);

    // -------------------------------------------------------------------------
    // Finalize
    // -------------------------------------------------------------------------

    if (nprocs == 4) {
        FREE(taskmap);
        FREE(itaskmap);
    }

    if (SPSP_LIB == SpSpLib::CUSPARSE) {
        void spgemmcusparseFree();
        spgemmcusparseFree();
    }

    // CSRm::free(expectedAP);
    CSRm::free(P);
    CSRm::free(A);
    Handles::free(h);
    FREE(log_file_name);
    BCM::shutdown();
    return 0;
}
