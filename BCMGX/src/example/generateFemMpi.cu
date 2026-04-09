#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "generator/fem.h"
#include "utility/distribute.h"
#include "utility/globals.h"
#include "utility/mpi.h"
#include "utility/utils.h"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define USAGE                                                                                                   \
    "Usage: %s <options>\n\n"                                                                                   \
    "\t-E, --E <double>                            E (Pa), default: " STR(DEFAULT_FEM_E) ".\n"                  \
    "\t-n, --nu <double>                           nu, default: " STR(DEFAULT_FEM_NU) ".\n"                     \
    "\t-x, --nx <int>                              nx, default: " STR(DEFAULT_FEM_NX) ".\n"                     \
    "\t-y, --ny <int>                              ny, default: " STR(DEFAULT_FEM_NY) ".\n"                     \
    "\t-P, --P <int>                               P, default: " STR(DEFAULT_FEM_P) ".\n"                       \
    "\t-Q, --Q <int>                               Q, default: " STR(DEFAULT_FEM_Q) ".\n"                       \
    "\t-e, --errlog <FILE_NAME>                    Write process-specific log to <FILE_NAME><PROC_ID>.\n"       \
    "\t-h, --help                                  Print this message.\n"                                       \
    "\t-O, --out-dir <DIR>                         Write additional files to <DIR>.\n"                          \
    "\t-S, --out-suffix <STRING>                   Use <SUFFIX> when writing additional files to output dir.\n" \
    "\t-t, --trace                                 Enable trace.\n"                                             \
    "\n"

int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Command line options
    // -------------------------------------------------------------------------

    // Parametri materiale (acciaio tipico)
    double E = DEFAULT_FEM_E; // Modulo di Young (Pa)
    double nu = DEFAULT_DEM_NU; // Coefficiente di Poisson

    int nx = DEFAULT_FEM_NX;
    int ny = DEFAULT_FEM_NY;
    int P = DEFAULT_FEM_P;
    int Q = DEFAULT_FEM_Q;

    char* log_file_name = NULL;

    static struct option long_options[] = {
        { "E", required_argument, NULL, 'E' },
        { "nu", required_argument, NULL, 'n' },
        { "nx", required_argument, NULL, 'x' },
        { "ny", required_argument, NULL, 'y' },
        { "P", required_argument, NULL, 'P' },
        { "Q", required_argument, NULL, 'Q' },
        { "errlog", required_argument, NULL, 'e' },
        { "help", no_argument, NULL, 'h' },
        { "out-dir", required_argument, NULL, 'O' },
        { "out-suffix", required_argument, NULL, 'S' },
        { "trace", no_argument, NULL, 't' },
    };

    signed char ch = 0;
    while ((ch = getopt_long(argc, argv, "E:n:x:y:P:Q:e:hO:S:t", long_options, NULL)) != -1) {
        switch (ch) {
        case 'E':
            E = atof(optarg);
            break;
        case 'n':
            nu = atof(optarg);
            break;
        case 'x':
            nx = atoi(optarg);
            break;
        case 'y':
            ny = atoi(optarg);
            break;
        case 'P':
            P = atoi(optarg);
            break;
        case 'Q':
            Q = atoi(optarg);
            break;
        case 'e':
            log_file_name = strdup(optarg);
            break;
        case 'O':
            output_dir = optarg;
            break;
        case 'S':
            output_suffix = optarg;
            break;
        case 't':
            trace_enabled = true;
            break;
        case 'h':
        default:
            DIE(USAGE, argv[0]);
        }
    }

    // -------------------------------------------------------------------------
    // Initialize MPI
    // -------------------------------------------------------------------------

    BCM::init(&argc, &argv, log_file_name);
    _MPI_ENV;

    // -------------------------------------------------------------------------
    // Generate matrix
    // -------------------------------------------------------------------------

    vector<double>* rhs_local = NULL;
    CSR *A_local = generateLocalFem(nx, ny, P, Q, E, nu, &rhs_local);

    // -------------------------------------------------------------------------
    // Join results
    // -------------------------------------------------------------------------

    
    CSR* A_global = nprocs > 1 ? join_matrix_mpi(A_local) : A_local;
    if (nprocs > 1) {
        CSRm::free(A_local);
    }
    vector<double>* rhs_global = nprocs > 1 ? aggregate_vector_all(rhs_local, A_local->full_n) : rhs_local;
    if (nprocs > 1) {
        Vector::free(rhs_local);
    }

    // -------------------------------------------------------------------------
    // Dump output
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        char filename[1024];

        snprintf(filename, 1024, "%s/A_%dx%d_%dx%d_mpi.mtx", output_dir.c_str(), nx, ny, P, Q);
        export_matrix_market(filename, A_global);

        snprintf(filename, 1024, "%s/rhs_%dx%d_%dx%d_mpi.mtx", output_dir.c_str(), nx, ny, P, Q);
        export_matrix_market(filename, rhs_global);

        CSRm::free(A_global);
    }
    Vector::free(rhs_global);

    printf("Fine\n");

    // -------------------------------------------------------------------------
    // Shutdown
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Shutdown\n");
    }
    BCM::shutdown();

    return 0;
}
