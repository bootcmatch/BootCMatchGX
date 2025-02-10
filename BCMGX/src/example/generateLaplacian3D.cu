#include "datastruct/CSR.h"
#include "generator/laplacian.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/globals.h"
#include "utility/memory.h"
#include "utility/mpi.h"
#include <assert.h>
#include <getopt.h>

#define USAGE                                                                                          \
    "Usage: %s -x <NUM> -y <NUM> -z <NUM> -P <NUM> -Q <NUM> -R <NUM> -o <FILE_NAME>\n\n"               \
    "\t-g, --generator [ 7p | 27p ]   Laplacian 3D generator.\n"                                       \
    "\t-x, --nx        <NUM>          Laplacian 3D nx parameter.\n"                                    \
    "\t-y, --ny        <NUM>          Laplacian 3D ny parameter.\n"                                    \
    "\t-z, --nz        <NUM>          Laplacian 3D nz parameter.\n"                                    \
    "\t-P              <NUM>          Laplacian 3D nx parameter.\n"                                    \
    "\t-Q              <NUM>          Laplacian 3D ny parameter.\n"                                    \
    "\t-R              <NUM>          Laplacian 3D nz parameter.\n"                                    \
    "\t-l, --log       <FILE_NAME>    Write process-specific log to <FILE_NAME><PROC_ID>.\n"           \
    "\t-o, --out       <FILE_NAME>    Write process-specific mtx to <FILE_NAME>_<PROC_ID>_<NPROCS>.\n" \
    "\n"

enum generator_t {
    LAP_7P,
    LAP_27P,
    INVALIG_GEN
};

generator_t getGenerator(const char* str)
{
    if (!strcmp(str, "7p")) {
        return LAP_7P;
    } else if (!strcmp(str, "27p")) {
        return LAP_27P;
    } else {
        return INVALIG_GEN;
    }
}

int main(int argc, char** argv)
{
    char* log_file_name = NULL;
    char* out_file_name = NULL;

    generator_t generator = LAP_27P;

    signed char ch;
    itype nx = 0;
    itype ny = 0;
    itype nz = 0;
    itype P = 0;
    itype Q = 0;
    itype R = 0;

    static struct option long_options[] = {
        { "generator", required_argument, NULL, 'g' },
        { "log", required_argument, NULL, 'l' },
        { "out", required_argument, NULL, 'o' },
        { "nx", required_argument, NULL, 'x' },
        { "ny", required_argument, NULL, 'y' },
        { "nz", required_argument, NULL, 'z' },
        { "P", required_argument, NULL, 'P' },
        { "Q", required_argument, NULL, 'Q' },
        { "R", required_argument, NULL, 'R' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    while ((ch = getopt_long(argc, argv, "g:l:o:x:y:z:P:Q:R:h", long_options, NULL)) != -1) {
        switch (ch) {
        case 'g':
            generator = getGenerator(optarg);
            break;
        case 'l':
            log_file_name = strdup(optarg);
            break;
        case 'o':
            out_file_name = strdup(optarg);
            break;
        case 'x':
            nx = atoi(optarg);
            break;
        case 'y':
            ny = atoi(optarg);
            break;
        case 'z':
            nz = atoi(optarg);
            break;
        case 'P':
            P = atoi(optarg);
            break;
        case 'Q':
            Q = atoi(optarg);
            break;
        case 'R':
            R = atoi(optarg);
            break;
        case 'h':
        default:
            printf(USAGE, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (nx == 0 || ny == 0 || nz == 0
        || P == 0 || Q == 0 || R == 0
        || out_file_name == NULL
        || generator == INVALIG_GEN) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    int myid, nprocs, device_id;
    StartMpi(&myid, &nprocs, &argc, &argv);

    // SetDevice
    int deviceCount = 0;
    CHECK_DEVICE(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount);
    device_id = assignDeviceToProcess();
    int assigned_device_id = device_id % deviceCount;
    fprintf(stderr, "Trying to set device %d. Total devices: %d. Assigned device: %d\n", device_id, deviceCount, assigned_device_id);
    CHECK_DEVICE(cudaSetDevice(assigned_device_id));

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

    CSR* Alocal = NULL;
    switch (generator) {
    case LAP_7P:
        Alocal = generateLocalLaplacian3D_7p(nx, ny, nz, P, Q, R);
        break;
    case LAP_27P:
        Alocal = generateLocalLaplacian3D_27p(nx, ny, nz, P, Q, R);
        break;
    default:
        printf("Invalid generator\n");
        exit(1);
    }
    check_and_fix_order(Alocal);
    Alocal->col_shifted = -Alocal->row_shift;

    if (log_file) {
        CSRm::print(Alocal, 3, 0, log_file);
    }

    {
        printf("Saving matrix to <%s_%d_%d>.\n", out_file_name, myid, nprocs);
        CSRm::printMM(Alocal, out_file_name);
        printf("Done\n");
    }

    CSRm::free(Alocal);

    FREE(log_file_name);
    FREE(out_file_name);
    MPI_Finalize();
    return 0;
}
