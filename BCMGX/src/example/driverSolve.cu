#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "halo_communication/halo_communication.h"
#include "preconditioner/prec_setup.h"
#include "preconditioner/prec_finalize.h"
#include "solver/solve.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/distribuite.h"
#include "utility/globals.h"
#include "utility/input.h"
#include "utility/memory.h"
#include "utility/mpi.h"
#include "utility/utils.h"
#include "utility/string.h"
#include "utility/profiling.h"

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

#define USAGE                                                                                                               \
    "Usage: %s [--matrix <FILE_NAME> | --laplacian <SIZE> | --laplacian-3d <FILE_NAME>] --settings <FILE_NAME>\n\n"         \
    "\tYou can specify only one out of the three available options: --matrix, --laplacian-3d and --laplacian\n\n"           \
    "\t-a, --laplacian <SIZE>                      Generate a matrix whose size is <SIZE>^3.\n"                             \
    "\t-B, --out-prefix <STRING>                   Use <PREFIX> when writing additional files to output dir.\n"             \
    "\t-d, --dump-matrix <FILE_NAME>               Write process-specific local input matrix to <FILE_NAME><PROC_ID>.\n"    \
    "\t-e, --errlog <FILE_NAME>                    Write process-specific log to <FILE_NAME><PROC_ID>.\n"                   \
    "\t-g, --laplacian-3d-generator [ 7p | 27p ]   Choose laplacian 3d generator (7 points or 27 points).\n"                \
    "\t-h, --help                                  Print this message.\n"                                                   \
    "\t-i, --info <FILE_NAME>                      Write info to <FILE_NAME>.\n"                                            \
    "\t-l, --laplacian-3d <FILE_NAME>              Read generation parameters from file <FILE_NAME>.\n"                     \
    "\t-m, --matrix <FILE_NAME>                    Read the matrix from file <FILE_NAME>.\n"                                \
    "\t-M, --detailed-metrics <FILE_NAME>          Write process-specific detailed profile log to <FILE_NAME><PROC_ID>.\n"  \
    "\t-o, --out <FILE_NAME>                       Write solution to <FILE_NAME>.\n"                                        \
    "\t-O, --out-dir <DIR>                         Write additional files to <DIR>.\n"                                      \
    "\t-p, --summary-prof <FILE_NAME>              Write process-specific summary profile log to <FILE_NAME><PROC_ID>.\n"   \
    "\t-P, --detailed-prof <FILE_NAME>             Write process-specific detailed profile log to <FILE_NAME><PROC_ID>.\n"  \
    "\t-s, --settings <FILE_NAME>                  Read settings from file <FILE_NAME>.\n"                                  \
    "\t-S, --out-suffix <STRING>                   Use <SUFFIX> when writing additional files to output dir.\n"             \
    "\t-x, --extended-prof                         Write extended profile info inside the info-file.\n\n"                     

#define MAX_AGGREGATED_SOLUTION_SIZE ((2L << 31) - 1L)

extern vtype* d_temp_storage_max_min;
extern vtype* min_max;
void release_bin(sfBIN bin);
extern sfBIN global_bin;

int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Command line options
    // -------------------------------------------------------------------------

    enum opts { MTX,
        LAP_3D,
        LAP,
        NONE } opt
        = NONE;

    char* mtx_file = NULL;
    char* lap_3d_file = NULL;
    generator_t generator = LAP_27P;
    itype n = 0;
    std::string settings_file;
    char* output_file_name = NULL;
    char* info_file_name = NULL;
    char* log_file_name = NULL;

    std::string summary_profile_prefix  = "";
    std::string detailed_profile_prefix = "";
    std::string detailed_metrics_prefix = "";
    std::string dump_matrix_prefix = "";

    signed char ch = 0;
    
    static struct option long_options[] = {
        { "laplacian", required_argument, NULL, 'a' },
        { "out-prefix", required_argument, NULL, 'B' },
        { "dump-matrix", required_argument, NULL, 'd' },
        { "errlog", required_argument, NULL, 'e' },
        { "laplacian-3d-generator", required_argument, NULL, 'g' },
        { "help", no_argument, NULL, 'h' },
        { "info", required_argument, NULL, 'i' },
        { "laplacian-3d", required_argument, NULL, 'l' },
        { "matrix", required_argument, NULL, 'm' },
        { "detailed-metrics", required_argument, NULL, 'M' },
        { "out", required_argument, NULL, 'o' },
        { "out-dir", required_argument, NULL, 'O' },
        { "summary-prof", required_argument, NULL, 'p' },
        { "detailed-prof", required_argument, NULL, 'P' },
        { "settings", required_argument, NULL, 's' },
        { "out-suffix", required_argument, NULL, 'S' },
        { "extended-prof", no_argument, NULL, 'x' },
    };

    while ((ch = getopt_long(argc, argv, "a:B:d:e:g:hi:l:m:M:o:O:p:P:s:S:x", long_options, NULL)) != -1) {
        switch (ch) {
        case 'a':
            n = atoi(optarg);
            opt = LAP;
            break;
        case 'B':
            output_prefix = optarg;
            break;
        case 'd':
            dump_matrix_prefix = optarg;
            break;
        case 'e':
            log_file_name = strdup(optarg);
            break;
        case 'g':
            generator = get_generator(optarg);
            break;
        case 'i':
            info_file_name = strdup(optarg);
            break;
        case 'l':
            lap_3d_file = strdup(optarg);
            opt = LAP_3D;
            break;
        case 'm':
            mtx_file = strdup(optarg);
            opt = MTX;
            break;
        case 'M':
            detailed_metrics_prefix = optarg;
            break;
        case 'o':
            output_file_name = strdup(optarg);
            break;
        case 'O':
            output_dir = optarg;
            break;
        case 'p':
            summary_profile_prefix = optarg;
            detailed_prof = true;
            break;
        case 'P':
            detailed_profile_prefix = optarg;
            detailed_prof = true;
            break;
        case 's':
            settings_file = optarg;
            break;
        case 'S':
            output_suffix = optarg;
            break;
        case 'x':
            detailed_prof = true;
            break;
        case 'h':
        default:
            printf(USAGE, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (opt == NONE || settings_file.empty() || generator == INVALIG_GEN) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    if (getenv("SCALENNZMISSING")) {
        scalennzmiss = atoi(getenv("SCALENNZMISSING"));
    }

    // -------------------------------------------------------------------------
    // Initialize MPI
    // -------------------------------------------------------------------------

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
    if (device_id != assigned_device_id) {
        fprintf(stderr, "Trying to set device %d. Total devices: %d. Assigned device: %d\n", device_id, deviceCount, assigned_device_id);
    }
    CHECK_DEVICE(cudaSetDevice(assigned_device_id));

    // -------------------------------------------------------------------------
    // Load configuration
    // -------------------------------------------------------------------------

    params p = ends_with(settings_file, ".properties")
        ? Params::initFromPropertiesFile(settings_file.c_str())
        : Params::initFromFile(settings_file.c_str());
    if (p.error != 0) {
        return -1;
    }

    // -------------------------------------------------------------------------
    // Read/Generate input matrix
    // -------------------------------------------------------------------------

    CSR* Alocal = NULL;
    if (opt == MTX) { // The master reads the matrix and distributes it.
        Alocal = read_local_matrix_from_mtx(mtx_file);
    } else if (opt == LAP_3D) {
        Alocal = generate_lap3d_local_matrix(generator, lap_3d_file);
    } else if (opt == LAP) {
        Alocal = generate_lap_local_matrix(n);
    }

    if (!dump_matrix_prefix.empty()) {
        char fname[256] = {0};
        snprintf(fname, 256, "%s%d.mtx", dump_matrix_prefix.c_str(), myid);
        CSRMatrixPrintMM(Alocal, fname);
    }

    itype full_n = Alocal->full_n;

    // -------------------------------------------------------------------------
    // Initialize solution
    // -------------------------------------------------------------------------

    vector<vtype>* rhs = Vector::init<vtype>(Alocal->n, true, true);
    Vector::fillWithValue(rhs, 1.);

    vector<vtype>* x0 = Vector::init<vtype>(Alocal->n, true, true);
    Vector::fillWithValue(x0, 0.);

    vector<vtype>* sol;
    xsize = 0;
    xvalstat = NULL;

    SolverOut solverOut;

    // -------------------------------------------------------------------------
    // Setup preconditioner
    // -------------------------------------------------------------------------

    handles* h = Handles::init();

    cgsprec pr;
    if (p.sprec != PreconditionerType::BCMG) {
        halo_info hi = haloSetup(Alocal, NULL);
        Alocal->halo = hi;
        shrink_col(Alocal, NULL);
    }

    pr.ptype = p.sprec;

    if (ISMASTER) {
        printf("Setting preconditioner\n");
    }
    prec_setup(h, Alocal, &pr, p, &(solverOut.precOut));

    CUDA_FREE(buffer_4_getmct);
    CUDA_FREE(glob_d_BlocksCount);
    CUDA_FREE(glob_d_BlocksOffset);

    // -------------------------------------------------------------------------
    // Solve
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Solving\n");
    }
    sol = solve(h, Alocal, rhs, x0, p, pr, &solverOut);
    Vector::free(rhs);
    Vector::free(x0);
    CSRm::free(Alocal);

    // -------------------------------------------------------------------------
    // Profiling
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Profiling\n");
    }
    size_t summaryLen = 0;
    ProfInfoSummary *summary = computeLocalProfilingInfoSummary(&summaryLen);

    if (!detailed_profile_prefix.empty()) {
        std::string filename = detailed_profile_prefix + to_string(myid);
        dumpLocalProfilingInfo(filename.c_str(), summary, summaryLen);
    }

    if (!summary_profile_prefix.empty()) {
        std::string filename = summary_profile_prefix + to_string(myid);
        FILE* file = fopen(filename.c_str(), "w");
        if (file == NULL) {
            printf("Error opening %s for writing\n", filename.c_str());
            dumpLocalProfilingInfoSummary(stderr, summary, summaryLen);
        } else {
            dumpLocalProfilingInfoSummary(file, summary, summaryLen);
            fclose(file);
        }
    }
    
    // -------------------------------------------------------------------------
    // Output
    // -------------------------------------------------------------------------

    if (info_file_name) {
        if (ISMASTER) {
            printf("Dumping output\n");
        }
        if (ISMASTER) {
            dump(info_file_name, p, pr, &solverOut);
        }
        dumpProfilingInfo(info_file_name, summary, summaryLen);
    }

    delete(summary);

    // -------------------------------------------------------------------------
    // Finalize preconditioner
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Finalizing preconditioner\n");
    }
    prec_finalize(Alocal, &pr, p, &(solverOut.precOut));

    // -------------------------------------------------------------------------
    // Release memory (1)
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Releasing memory (1)\n");
    }
    Handles::free(h);

    CUDA_FREE(xvalstat);
    if (global_bin.stream) {
        release_bin(global_bin);
    }
    CUDA_FREE(d_temp_storage_max_min);
    CUDA_FREE(min_max);

    // -------------------------------------------------------------------------
    // Aggregate and dump solution
    // -------------------------------------------------------------------------

    if (output_file_name) {
        if (full_n <= MAX_AGGREGATED_SOLUTION_SIZE) {
            if (ISMASTER) {
                printf("Dumping solution\n");
            }
            vector<vtype>* collectedSol = aggregate_vector(sol, full_n);
            if (ISMASTER) {
                FILE* output_file = fopen(output_file_name, "w");
                if (output_file == NULL) {
                    printf("Error opening %s for writing\n", output_file_name);
                }
                Vector::print(collectedSol, -1, output_file);
                fclose(output_file);
                printf("...done.\n");
            }
            Vector::free(collectedSol);
        } else {
            if (ISMASTER) {
                printf("WARNING: solution cannot be aggregated when size > %ld\n", MAX_AGGREGATED_SOLUTION_SIZE);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Release memory (2)
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Releasing memory (2)\n");
    }
    Vector::free(sol);

    // -------------------------------------------------------------------------
    // Shutdown
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Shutdown\n");
    }
    MPI_Finalize();
    FREE(taskmap);
    FREE(itaskmap);
    return 0;
}
