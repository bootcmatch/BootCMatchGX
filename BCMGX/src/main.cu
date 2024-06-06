#include "basic_kernel/halo_communication/extern2.h" //PICO

#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "generator/laplacian.h"
#include "solver/solve.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/distribuite.h"
#include "utility/function_cnt.h" //PICO
#include "utility/globals.h"
#include "utility/mpi.h"
#include "utility/utils.h"

#include <assert.h>
#include <getopt.h>
#include <mpi.h>
#include <nsparse.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

using namespace std;

#define USAGE                                                                                                     \
    "Usage: %s [--matrix <FILE_NAME> | --laplacian <SIZE>] --settings <FILE_NAME> --info <FILE_NAME>\n\n"         \
    "\tYou can specify only one out of the three available options: --matrix, --laplacian-3d and --laplacian\n\n" \
    "\t-m, --matrix <FILE_NAME>                    Read the matrix from file <FILE_NAME>.\n"                      \
    "\t-l, --laplacian-3d <FILE_NAME>              Read generation parameters from file <FILE_NAME>.\n"           \
    "\t-g, --laplacian-3d-generator [ 7p | 27p ]   Choose laplacian 3d generator (7 points or 27 points).\n"      \
    "\t-a, --laplacian <SIZE>                      Generate a matrix whose size is <SIZE>^3.\n"                   \
    "\t-s, --settings <FILE_NAME>                  Read settings from file <FILE_NAME>.\n"                        \
    "\t-e, --errlog <FILE_NAME>                    Write process-specific log to <FILE_NAME><PROC_ID>.\n"         \
    "\t-o, --out <FILE_NAME>                       Write solution to <FILE_NAME>.\n"                              \
    "\t-i, --info <FILE_NAME>                      Write info to <FILE_NAME>.\n"

extern vtype* d_temp_storage_max_min;
extern vtype* min_max;
void release_bin(sfBIN bin);
extern sfBIN global_bin;

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
    PUSH_RANGE(__func__, 1)

    enum opts { MTX,
        LAP_3D,
        LAP,
        NONE } opt
        = NONE;
    char* log_file_name = NULL;
    char* mtx_file = NULL;
    char* lap_3d_file = NULL;
    char* settings_file = NULL;
    char* output_file_name = NULL;
    char* info_file_name = NULL;
    signed char ch = 0;
    itype n = 0;
    generator_t generator = LAP_27P;

    static struct option long_options[] = {
        { "errlog", required_argument, NULL, 'e' },
        { "matrix", required_argument, NULL, 'm' },
        { "laplacian-3d", required_argument, NULL, 'l' },
        { "laplacian-3d-generator", required_argument, NULL, 'g' },
        { "laplacian", required_argument, NULL, 'a' },
        { "settings", required_argument, NULL, 's' },
        { "out", required_argument, NULL, 'o' },
        { "info", required_argument, NULL, 'i' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    while ((ch = getopt_long(argc, argv, "e:m:l:g:a:s:p:o:i:h", long_options, NULL)) != -1) {
        switch (ch) {
        case 'e':
            log_file_name = strdup(optarg);
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
        case 's':
            settings_file = strdup(optarg);
            break;
        case 'o':
            output_file_name = strdup(optarg);
            break;
        case 'i':
            info_file_name = strdup(optarg);
            break;
        case 'h':
        default:
            printf(USAGE, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (opt == NONE || settings_file == NULL || generator == INVALIG_GEN || info_file_name == NULL) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    // setup AMG:
    int myid, nprocs, device_id;
    StartMpi(&myid, &nprocs, &argc, &argv);

    if (log_file_name) {
        open_log_file(myid, log_file_name);
    }

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

    params p = Params::initFromFile(settings_file);
    if (p.error != 0) {
        return -1;
    }

    CSR* Alocal = NULL;
    if (opt == MTX) { // The master reads the matrix and distributes it.
        Alocal = read_local_matrix_from_mtx(mtx_file);
    } else if (opt == LAP_3D) {
        Alocal = generate_lap3d_local_matrix(generator, lap_3d_file);
    } else if (opt == LAP) {
        Alocal = generate_lap_local_matrix(n);
    }

    Vectorinit_CNT;

    // BUG? Alocal->n  ===>  Alocal->m/nprocs
    vector<vtype>* rhs = Vector::init<vtype>(Alocal->n, true, true);
    Vector::fillWithValue(rhs, 1.);

    vector<vtype>* x0 = Vector::init<vtype>(Alocal->n, true, true);
    Vector::fillWithValue(x0, 0.);

    itype full_n = Alocal->full_n;

    vector<vtype>* sol;
    xsize = 0;
    xvalstat = NULL;

    SolverOut solverOut;
    sol = solve(Alocal, rhs, x0, p, &solverOut);

    if (ISMASTER) {
        dump(info_file_name, p, &solverOut);
    }

    if (xsize > 0 && xvalstat) {
        cudaFree(xvalstat);
    }

    // controlla per prec!= bcmg
    if (global_bin.stream) {
        release_bin(global_bin);
    }

    if (d_temp_storage_max_min) {
        MY_CUDA_CHECK(cudaFree(d_temp_storage_max_min));
    }
    if (min_max) {
        MY_CUDA_CHECK(cudaFree(min_max));
    }

#if 1
    vector<vtype>* collectedSol = aggregate_vector(sol, full_n);
    if (ISMASTER) {
        if (output_file_name) {
            FILE* output_file = fopen(output_file_name, "w");
            if (output_file == NULL) {
                printf("Error opening %s for writing\n", output_file_name);
            }
            Vector::print(collectedSol, -1, output_file);
            fclose(output_file);
        }
        printf("...done.\n");
    }
    Vector::free(collectedSol);
#endif

    Vector::free(sol);
    MPI_Finalize();
    POP_RANGE
    return 0;
}
