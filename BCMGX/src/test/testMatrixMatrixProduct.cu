#include "basic_kernel/halo_communication/halo_communication.h"
#include "custom_cudamalloc/custom_cudamalloc.h"
#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "generator/laplacian.h"
#include "gpoweru/GPowerU.hpp"
#include "op/spspmpi.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/distribuite.h"
#include "utility/globals.h"
#include "utility/handles.h"
#include "utility/memoryPools.h"
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
    "Usage: %s [--matrix <FILE_NAME> | --laplacian <SIZE> | --laplacian-3d <FILE_NAME>] [--time] [--energy] [--processes-per-node <N_PROCS>]\n\n"              \
    "\tYou can specify only one out of the three available options: --matrix, --laplacian-3d and --laplacian.\n\n"                                              \
    "\t-m, --matrix <FILE_NAME>                    Read the matrix from file <FILE_NAME>. Please note that this option works only in a mono-process setting.\n" \
    "\t-l, --laplacian-3d <FILE_NAME>              Read generation parameters from file <FILE_NAME>.\n"                                                         \
    "\t-g, --laplacian-3d-generator [ 7p | 27p ]   Choose laplacian 3d generator (7 points or 27 points).\n"                                                    \
    "\t-a, --laplacian <SIZE>                      Generate a matrix whose size is <SIZE>^3.\n"                                                                 \
    "\t-n, --processes-per-node <N_PROCS>          Number of MPI processes per node  \n"                                                                        \
    "\t-t, --time                                  Output the execution time of the application\n"                                                              \
    "\t-e, --energy                                If set measure the energy consumption\n\n"

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

    taskmap = (int*)Malloc(nprocs * sizeof(*taskmap));
    if (taskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for taskmap\n", nprocs * sizeof(*taskmap));
        exit(1);
    }

    itaskmap = (int*)Malloc(nprocs * sizeof(*itaskmap));
    if (itaskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for itaskmap\n", nprocs * sizeof(*itaskmap));
        exit(1);
    }

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
    free(parms);
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
    int energy = 0;
    generator_t generator = LAP_27P;
    FILE* fp_time;
    int procs_per_node = 1;
    struct timeval matrix_t, setup_t, start_kernel_t, stop_kernel_t, teardown_t;
    int mem_alloc_size = GLOB_MEM_ALLOC_SIZE;

    static struct option long_options[] = {
        { "matrix", required_argument, NULL, 'm' },
        { "laplacian-3d", required_argument, NULL, 'l' },
        { "laplacian-3d-generator", required_argument, NULL, 'g' },
        { "laplacian", required_argument, NULL, 'a' },
        { "processes-per-node", required_argument, NULL, 'n' },
        { "time", no_argument, NULL, 't' },
        { "energy", no_argument, NULL, 'e' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    while ((ch = getopt_long(argc, argv, "n:m:l:g:a:h:e:t", long_options, NULL)) != -1) {
        switch (ch) {
        case 'n':
            procs_per_node = atoi(optarg);
            break;
        case 't':
            time = 1;
            break;
        case 'e':
            energy = 1;
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

    handles* h = Handles::init();

    // *******************
    // ENERGY - gpoweru start
    // *******************
    if (energy == 1 && myid % procs_per_node == 0) {
        printf("\n####### Process %d, running GPowerU #######\n", myid);
        if (GPowerU_init() != 0) {
            fprintf(stderr, "%s: error: initializing...\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    // *********************
    // ENERGY - gen matrix
    // *********************
    if (energy == 1) {
        gettimeofday(&matrix_t, NULL);
    }

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
    // ENERGY - setup
    // *********************
    if (energy == 1) {
        gettimeofday(&setup_t, NULL);
    }

    // init memory pool
    MemoryPool::initContext(Alocal->full_n, Alocal->n);
    iPtemp1 = NULL;
    vPtemp1 = NULL;
    MY_CUDA_CHECK(cudaMallocHost(&iAtemp1, sizeof(itype) * mem_alloc_size));
    MY_CUDA_CHECK(cudaMallocHost(&vAtemp1, sizeof(vtype) * mem_alloc_size));
    MY_CUDA_CHECK(cudaMalloc(&idevtemp1, sizeof(itype) * mem_alloc_size));
    MY_CUDA_CHECK(cudaMalloc(&vdevtemp1, sizeof(vtype) * mem_alloc_size));
    MY_CUDA_CHECK(cudaMalloc(&idevtemp2, sizeof(itype) * mem_alloc_size));

    CustomCudaMalloc::init((Alocal->nnz) * 8, (Alocal->nnz) * 4);
    CustomCudaMalloc::init((Alocal->nnz) * 2, (Alocal->nnz) * 2, 1);
    CustomCudaMalloc::init((Alocal->nnz) * 4, (Alocal->nnz) * 4, 2);

    vector<int>* _bitcol = NULL;
    _bitcol = get_missing_col(Alocal, NULL);
    compute_rows_to_rcv_CPU(Alocal, NULL, _bitcol);
    Vector::free(_bitcol);

    // *********************
    // ENERGY - spmm start
    // *********************
    if (energy == 1) {
        gettimeofday(&start_kernel_t, NULL);
    }
    // *********************
    // TIME
    // *********************
    double TOT_TIMEM;
    if (time == 1 && ISMASTER) {
        TOT_TIMEM = MPI_Wtime();
    }

    CSR* APlocal = nsparseMGPU_commu_new(h, Alocal, Plocal, false);
    cudaDeviceSynchronize();

    // *********************
    // TIME
    // *********************
    if (time == 1 && ISMASTER) {
        printf("TOTAL_TIME: %f\n", (MPI_Wtime() - TOT_TIMEM));
    }
    // *********************
    // ENERGY - spmm stop
    // *********************
    if (energy == 1) {
        gettimeofday(&stop_kernel_t, NULL);
    }

    CSRm::free(Alocal);
    CSRm::free(Plocal);
    CSRm::free(APlocal);

    if (xsize > 0) {
        cudaFree(xvalstat);
    }
    CustomCudaMalloc::free(1);

    if (d_temp_storage_max_min) {
        MY_CUDA_CHECK(cudaFree(d_temp_storage_max_min));
    }
    if (min_max) {
        MY_CUDA_CHECK(cudaFree(min_max));
    }

    // *********************
    // ENERGY - teardown
    // *********************
    if (energy == 1) {
        gettimeofday(&teardown_t, NULL);
    }
    // *********************
    // ENERGY - write files
    // *********************
    if (energy == 1) {
        char filename[256];
        snprintf(filename, sizeof(filename), "data/spmm_%d_of_%d.time", myid, nprocs);
        mkdir("data", 0777);
        fp_time = fopen(filename, "w");

        fprintf(fp_time, "gen matrix;%ld;%ld\n", matrix_t.tv_sec, matrix_t.tv_usec);
        fprintf(fp_time, "setup;%ld;%ld\n", setup_t.tv_sec, setup_t.tv_usec);
        fprintf(fp_time, "start kernel;%ld;%ld\n", start_kernel_t.tv_sec, start_kernel_t.tv_usec);
        fprintf(fp_time, "stop kernel;%ld;%ld\n", stop_kernel_t.tv_sec, stop_kernel_t.tv_usec);
        fprintf(fp_time, "teardown;%ld;%ld\n", teardown_t.tv_sec, teardown_t.tv_usec);

        fclose(fp_time);
    }
    // *******************
    // ENERGY - gpoweru shutdown
    // *******************
    if (energy == 1) {
        if (myid % procs_per_node == 0) {
            if (GPowerU_end(5) != 0) {
                fprintf(stderr, " error: terminating...\n");
                _exit(1);
            }
        } else {
            sleep(5);
        }
    }

    MPI_Finalize();
    return 0;
}
