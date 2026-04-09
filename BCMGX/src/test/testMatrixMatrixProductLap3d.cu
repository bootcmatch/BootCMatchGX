#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <getopt.h>

#include "datastruct/CSR.h"
#include "generator/laplacian.h"
#include "op/spspmpi.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/distribute.h"
#include "utility/globals.h"
#include "utility/handles.h"
#include "utility/input.h"
#include "utility/mpi.h"
#include "utility/utils.h"

#define DEBUG 0

#define USAGE                                                                                                \
    "\nUsage: %s --log <FILE_NAME>\n"                                                                        \
    "\t-e, --log <FILE_NAME>            Write log files (one per MPI process) to <FILE_NAME>_<MPI_PID>.\n"   \
    "\t-O, --out-dir <DIR>              Write additional files to <DIR>.\n"                                  \
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

    itype nx1 = 0;
    itype ny1 = 0;
    itype nz1 = 0;
    itype P1 = 0;
    itype Q1 = 0;
    itype R1 = 0;

    itype nx2 = 0;
    itype ny2 = 0;
    itype nz2 = 0;
    itype P2 = 0;
    itype Q2 = 0;
    itype R2 = 0;

    static struct option long_options[] = {
        { "log", required_argument, NULL, 'e' },
        { "nx1", required_argument, NULL, '1' },
        { "ny1", required_argument, NULL, '2' },
        { "nz1", required_argument, NULL, '3' },
        { "P1" , required_argument, NULL, '4' },
        { "Q1" , required_argument, NULL, '5' },
        { "R1" , required_argument, NULL, '6' },
        { "nx2", required_argument, NULL, '7' },
        { "ny2", required_argument, NULL, '8' },
        { "nz2", required_argument, NULL, '9' },
        { "P2" , required_argument, NULL, '0' },
        { "Q2" , required_argument, NULL, 'A' },
        { "R2" , required_argument, NULL, 'B' },
        { "out-dir", required_argument, NULL, 'O' },
    };

    char ch = 0;
    while ((ch = getopt_long(argc, argv, "e:1:2:3:4:5:6:7:8:9:0:A:B:O:h", long_options, NULL)) != -1) {
        switch (ch) {
        case 'e':
            log_file_name = strdup(optarg);
            break;
        case '1':
            nx1 = atoi(optarg);
            break;
        case '2':
            ny1 = atoi(optarg);
            break;
        case '3':
            nz1 = atoi(optarg);
            break;
        case '4':
            P1 = atoi(optarg);
            break;
        case '5':
            Q1 = atoi(optarg);
            break;
        case '6':
            R1 = atoi(optarg);
            break;
        case '7':
            nx2 = atoi(optarg);
            break;
        case '8':
            ny2 = atoi(optarg);
            break;
        case '9':
            nz2 = atoi(optarg);
            break;
        case '0':
            P2 = atoi(optarg);
            break;
        case 'A':
            Q2 = atoi(optarg);
            break;
        case 'B':
            R2 = atoi(optarg);
            break;
        case 'O':
            output_dir = optarg;
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
    // trace_enabled = true;
    _MPI_ENV;

    // -------------------------------------------------------------------------
    // Preparation
    // -------------------------------------------------------------------------

    handles* h = Handles::init();

    // -------------------------------------------------------------------------
    // Init data
    // -------------------------------------------------------------------------

    CSR *hA = generate_lap3d_local_matrix_host(LAP_7P, nx1, ny1, nz1, P1, Q1, R1);
#if DEBUG
    if (ISMASTER) fprintf(stderr, "A:\n");
    CSRm::debug(hA, stderr);
#endif

    CSR *hP = generate_lap3d_local_matrix_host(LAP_7P, nx2, ny2, nz2, P2, Q2, R2);
#if DEBUG
    if (ISMASTER) fprintf(stderr, "P:\n");
    CSRm::debug(hP, stderr);
#endif
    
    // -------------------------------------------------------------------------

    TRACE("copyToDevice dA");
    CSR* dA = CSRm::copyToDevice(hA);

    TRACE("copyToDevice dP");
    CSR* dP = CSRm::copyToDevice(hP);

    CSRm::free(hA);
    CSRm::free(hP);

    // -------------------------------------------------------------------------

    iAtemp1 = CUDA_MALLOC_HOST(itype, GLOB_MEM_ALLOC_SIZE, true);
    vAtemp1 = CUDA_MALLOC_HOST(vtype, GLOB_MEM_ALLOC_SIZE, true);
    idevtemp1 = CUDA_MALLOC(itype, GLOB_MEM_ALLOC_SIZE, true);
    vdevtemp1 = CUDA_MALLOC(vtype, GLOB_MEM_ALLOC_SIZE, true);
    idevtemp2 = CUDA_MALLOC(itype, GLOB_MEM_ALLOC_SIZE, true);

    // -------------------------------------------------------------------------

    CSR *dAP = CSRm::product(h, dA, dP);
    
    ASSERT(dAP->on_the_device);
    ASSERT(dA->row != dAP->row);
    ASSERT(dP->row != dAP->row);
    ASSERT(dA->col != dAP->col);
    ASSERT(dP->col != dAP->col);
    ASSERT(dA->col8 != dAP->col8);
    ASSERT(dP->col8 != dAP->col8);
    ASSERT(dA->val != dAP->val);
    ASSERT(dP->val != dAP->val);

    // CSRm::free(dP);
    // CSRm::free(dA);
    
    // -------------------------------------------------------------------------

    CUDA_FREE_HOST(iAtemp1);
    CUDA_FREE_HOST(vAtemp1);
    CUDA_FREE(idevtemp1);
    CUDA_FREE(vdevtemp1);
    CUDA_FREE(idevtemp2);
    
    // -------------------------------------------------------------------------

// #if DEBUG
//     if (ISMASTER) {
//         fprintf(stderr, "Computed result:\n");
//     }
//     CSRm::debug(dAP, stderr);
// #endif

    // -------------------------------------------------------------------------

    ASSERT(CSRm::checkProduct(h, dA, dP, dAP));

    // -------------------------------------------------------------------------

    CSRm::free(dP);
    CSRm::free(dA);
    CSRm::free(dAP);

    if (SPSP_LIB == SpSpLib::CUSPARSE) {
        void spgemmcusparseFree();
        spgemmcusparseFree();
    }

    // -------------------------------------------------------------------------
    // Finalize
    // -------------------------------------------------------------------------
    
    Handles::free(h);
    FREE(log_file_name);
    BCM::shutdown();
    return 0;
}
