#include "halo_communication/halo_communication.h"
#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "generator/laplacian.h"
#include "op/spspmpi.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/distribuite.h"
#include "utility/globals.h"
#include "utility/handles.h"
#include "utility/memory.h"
#include "utility/mpi.h"
#include "utility/utils.h"

#include <assert.h>
#include <getopt.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#define GLOB_MEM_ALLOC_SIZE 2000000

using namespace std;

#define USAGE                                                                                                                                                   \
    "Usage: %s [--matrix <FILE_NAME> | --laplacian-3d <FILE_NAME> |  --laplacian-3d-generator [ 7p | 27p ] | --laplacian <SIZE>] [--time]\n\n"              \
    "\tYou can specify only one out of the three available options: --matrix, --laplacian-3d and --laplacian.\n\n"                                              \
    "\t-m, --matrix <FILE_NAME>                    Read the matrix from file <FILE_NAME>. Please note that this option works only in a mono-process setting.\n" \
    "\t-l, --laplacian-3d <FILE_NAME>              Read generation parameters from file <FILE_NAME>.\n"                                                         \
    "\t-g, --laplacian-3d-generator [ 7p | 27p ]   Choose laplacian 3d generator (7 points or 27 points).\n"                                                    \
    "\t-a, --laplacian <SIZE>                      Generate a matrix whose size is <SIZE>^3.\n"                                                                 \
    "\t-t, --time                                  Output the execution time of the application\n\n"

extern vtype* d_temp_storage_max_min;
extern vtype* min_max;

enum generator_t {
    LAP_7P,
    LAP_27P,
    INVALIG_GEN
};

generator_t get_generator(const char* str)
{
    if (!strcmp(str, "7p")) {
        return LAP_7P;
    } else if (!strcmp(str, "27p")) {
        return LAP_27P;
    } else {
        return INVALIG_GEN;
    }
}

CSR* read_local_matrix_from_mtx(const char* mtx_file)
{
    _MPI_ENV;

    CSR* Alocal_master = NULL;
    if (ISMASTER) {
        Alocal_master = read_matrix_from_file(mtx_file, 0, false);
        check_and_fix_order(Alocal_master);
    }

    taskmap = MALLOC(int, nprocs);
    itaskmap = MALLOC(int, nprocs);

    for (int i = 0; i < nprocs; i++) {
        taskmap[i] = i;
        itaskmap[i] = i;
    }

    CSR* Alocal = split_matrix_mpi(Alocal_master);
    if (ISMASTER) {
        CSRm::free(Alocal_master);
    }

    snprintf(idstring, sizeof(idstring), "1_1_1");
    CSRm::shift_cols(Alocal, -Alocal->row_shift);
    Alocal->col_shifted = -Alocal->row_shift;

    return Alocal;
}

CSR* generate_lap_local_matrix(itype n)
{
    CSR* Alocal_host = generateLocalLaplacian3D(n);
    check_and_fix_order(Alocal_host);
    CSR* Alocal = CSRm::copyToDevice(Alocal_host);
    Alocal->col_shifted = -Alocal->row_shift;
    CSRm::free(Alocal_host);
    return Alocal;
}

CSR* generate_lap3d_local_matrix(generator_t generator, const char* lap_3d_file)
{
    _MPI_ENV;
    enum lap_params { nx = 0,
        ny = 1,
        nz = 2,
        P = 3,
        Q = 4,
        R = 5 };
    int* parms = read_laplacian_file(lap_3d_file);
    if (nprocs != (parms[P] * parms[Q] * parms[R])) {
        fprintf(stderr, "Nproc must be equal to P*Q*R\n");
        exit(EXIT_FAILURE);
    }
    CSR* Alocal_host = Alocal_host = NULL;
    switch (generator) {
    case LAP_7P:
        fprintf(stderr, "Using laplacian 3d 7 points generator.\n");
        Alocal_host = generateLocalLaplacian3D_7p(parms[nx], parms[ny], parms[nz], parms[P], parms[Q], parms[R]);
        break;
    case LAP_27P:
        fprintf(stderr, "Using laplacian 3d 27 points generator.\n");
        Alocal_host = generateLocalLaplacian3D_27p(parms[nx], parms[ny], parms[nz], parms[P], parms[Q], parms[R]);
        break;
    default:
        printf("Invalid generator\n");
        exit(1);
    }
    snprintf(idstring, sizeof(idstring), "%dx%dx%d", parms[P], parms[Q], parms[R]);
    FREE(parms);
    check_and_fix_order(Alocal_host);
    CSR* Alocal = CSRm::copyToDevice(Alocal_host);
    Alocal->col_shifted = -Alocal->row_shift;
    CSRm::free(Alocal_host);
    return Alocal;
}

int main(int argc, char** argv)
{
    enum opts { MTX,
        LAP_3D,
        LAP,
        NONE } opt
        = NONE;
    char* mtx_file = NULL;
    char* lap_3d_file = NULL;
    signed char ch;
    itype n = 0;
    int time = 0;
    generator_t generator = LAP_27P;
    int mem_alloc_size = GLOB_MEM_ALLOC_SIZE;

    static struct option long_options[] = {
        { "matrix", required_argument, NULL, 'm' },
        { "laplacian-3d", required_argument, NULL, 'l' },
        { "laplacian-3d-generator", required_argument, NULL, 'g' },
        { "laplacian", required_argument, NULL, 'a' },
        { "time", no_argument, NULL, 't' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    while ((ch = getopt_long(argc, argv, "m:l:g:a:ht", long_options, NULL)) != -1) {
        switch (ch) {
        case 't':
            time = 1;
            break;
        case 'm':
            mtx_file = strdup(optarg);
            opt = MTX;
            break;
        case 'l':
            lap_3d_file = strdup(optarg);
            opt = LAP_3D;
            break;
        case 'g':
            generator = get_generator(optarg);
            break;
        case 'a':
            n = atoi(optarg);
            opt = LAP;
            break;
        case 'h':
        default:
            printf(USAGE, argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (opt == NONE) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    int myid, nprocs, device_id;
    StartMpi(&myid, &nprocs, &argc, &argv);

    if (getenv("SCALENNZMISSING")) {
        scalennzmiss = atoi(getenv("SCALENNZMISSING"));
    }

    // SetDevice
    int deviceCount = 0;
    CHECK_DEVICE(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount);
    device_id = assignDeviceToProcess();
    int assigned_device_id = device_id % deviceCount;
    fprintf(stderr, "Trying to set device %d. Total devices: %d. Assigned device: %d\n", device_id, deviceCount, assigned_device_id);
    CHECK_DEVICE(cudaSetDevice(assigned_device_id));

    CSR* Alocal = NULL;
    CSR* Plocal = NULL;
    if (opt == MTX) { // The master reads the matrix and distributes it.
        Alocal = read_local_matrix_from_mtx(mtx_file);
        Plocal = read_local_matrix_from_mtx(mtx_file);
    } else if (opt == LAP_3D) {
        Alocal = generate_lap3d_local_matrix(generator, lap_3d_file);
        Plocal = generate_lap3d_local_matrix(generator, lap_3d_file);
    } else if (opt == LAP) {
        Alocal = generate_lap_local_matrix(n);
        Plocal = generate_lap_local_matrix(n);
    }

    // *********************
    // TIME
    // *********************
    double TOT_TIMEM;
    if (time == 1 && ISMASTER) {
        TOT_TIMEM = MPI_Wtime();
    }

    // CSR* APlocal = sparsemat_sparsemat ( Alocal, Plocal, mem_alloc_size);
    CSR* APlocal = SpMM(Alocal, Plocal, mem_alloc_size);
    cudaDeviceSynchronize();

    // *********************
    // TIME
    // *********************
    if (time == 1 && ISMASTER) {
        printf("TOTAL_TIME: %f\n", (MPI_Wtime() - TOT_TIMEM));
    }
    CSRm::free(Alocal);
    CSRm::free(Plocal);
    CSRm::free(APlocal);

    // if (xsize > 0) {
    CUDA_FREE(xvalstat);
    //}
    CUDA_FREE(d_temp_storage_max_min);
    CUDA_FREE(min_max);

    MPI_Finalize();
    return 0;
}
