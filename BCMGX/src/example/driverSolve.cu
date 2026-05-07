#include "config/Params.h"
#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "halo_communication/halo_communication.h"

#include "preconditioner/prec_setup.h"
#include "preconditioner/prec_finalize.h"
#include "solver/solve.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/distribute.h"
#include "utility/globals.h"
#include "utility/input.h"
#include "utility/memory.h"
#include "utility/mpi.h"
#include "utility/utils.h"
#include "utility/string.h"
#include "utility/profiling.h"
#include "solver/krylov_base/powerMethod.h"

#include <getopt.h>
#include <mpi.h>
#include <nsparse.h>
#include <regex>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#define USAGE                                                                                                                           \
    "Usage: %s [--matrix <FILE_NAME> | --laplacian <SIZE> | --laplacian-3d <FILE_NAME> | --fem <FILE_NAME>] --settings <FILE_NAME>\n\n" \
    "\tYou can specify only one out of the four available options: --matrix, --laplacian, --laplacian-3d, --fem.\n\n"                   \
    "\t-a, --laplacian <SIZE>                      Generate a matrix whose size is <SIZE>^3.\n"                                         \
    "\t-B, --out-prefix <STRING>                   Use <PREFIX> when writing additional files to output dir.\n"                         \
    "\t-d, --dump-matrix <FILE_NAME>               Write process-specific local input matrix to <FILE_NAME><PROC_ID>.\n"                \
    "\t-e, --errlog <FILE_NAME>                    Write process-specific log to <FILE_NAME><PROC_ID>.\n"                               \
    "\t-f, --fem <FILE_NAME>                       Read generation parameters from file <FILE_NAME> (.properties).\n"                   \
    "\t-g, --laplacian-3d-generator [ 7p | 27p ]   Choose laplacian 3d generator (7 points or 27 points).\n"                            \
    "\t-h, --help                                  Print this message.\n"                                                               \
    "\t-i, --info <FILE_NAME>                      Write info to <FILE_NAME>.\n"                                                        \
    "\t-l, --laplacian-3d <FILE_NAME>              Read generation parameters from file <FILE_NAME>.\n"                                 \
    "\t-m, --matrix <FILE_NAME>                    Read the matrix from file <FILE_NAME>.\n"                                            \
    "\t-r, --rhs <FILE_NAME>                       Read the rhs vector from file <FILE_NAME>.\n"                                        \
    "\t-M, --detailed-metrics <FILE_NAME>          Write process-specific detailed profile log to <FILE_NAME><PROC_ID>.\n"              \
    "\t-o, --out <FILE_NAME>                       Write solution to <FILE_NAME>.\n"                                                    \
    "\t-R, --out-rhs <FILE_NAME>                   Write rhs vector to <FILE_NAME>.\n"                                                  \
    "\t-O, --out-dir <DIR>                         Write additional files to <DIR>.\n"                                                  \
    "\t-p, --summary-prof <FILE_NAME>              Write process-specific summary profile log to <FILE_NAME><PROC_ID>.\n"               \
    "\t-P, --detailed-prof <FILE_NAME>             Write process-specific detailed profile log to <FILE_NAME><PROC_ID>.\n"              \
    "\t-s, --settings <FILE_NAME>                  Read settings from file <FILE_NAME>.\n"                                              \
    "\t-S, --out-suffix <STRING>                   Use <SUFFIX> when writing additional files to output dir.\n"                         \
    "\t-t, --trace                                 Enable trace.\n"                                                                     \
    "\t-w, --wait                                  Wait 1 minute before starting.\n"                                                    \
    "\t-x, --extended-prof                         Write extended profile info inside the info-file.\n"                                 \
    "\n"

#define MAX_AGGREGATED_SOLUTION_SIZE ((2L << 31) - 1L)

bool is_cgs(SolverType st) {
    return st == SolverType::CGS || st == SolverType::LSGS || st == SolverType::CGSN || st == SolverType::PIPELINED_CGS || st == SolverType::CGS_CUBLAS;
}

std::string filename_for(const std::string &output_file_name, const std::string &solverStr) {
    return std::regex_replace(output_file_name, std::regex("%s"), "_" + solverStr);
}

int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Command line options
    // -------------------------------------------------------------------------

    enum opts { MTX,
        LAP_3D,
        LAP,
        FEM,
        NONE } opt
        = NONE;

    char* input_file = NULL;
    char* rhs_file = NULL;

    generator_t generator = LAP_27P;
    itype n = 0;
    std::string settings_file;
    char* output_file_name = NULL;
    char* rhs_output_file_name = NULL;
    char* info_file_name = NULL;
    char* log_file_name = NULL;

    std::string summary_profile_prefix  = "";
    std::string detailed_profile_prefix = "";
    std::string detailed_metrics_prefix = "";
    std::string dump_matrix_prefix = "";

    bool wait = false;

    signed char ch = 0;
    
    static struct option long_options[] = {
        { "laplacian", required_argument, NULL, 'a' },
        { "out-prefix", required_argument, NULL, 'B' },
        { "dump-matrix", required_argument, NULL, 'd' },
        { "errlog", required_argument, NULL, 'e' },
        { "fem", required_argument, NULL, 'f' },
        { "laplacian-3d-generator", required_argument, NULL, 'g' },
        { "help", no_argument, NULL, 'h' },
        { "info", required_argument, NULL, 'i' },
        { "laplacian-3d", required_argument, NULL, 'l' },
        { "matrix", required_argument, NULL, 'm' },
        { "rhs", required_argument, NULL, 'r' },
        { "detailed-metrics", required_argument, NULL, 'M' },
        { "out", required_argument, NULL, 'o' },
        { "out-rhs", required_argument, NULL, 'R' },
        { "out-dir", required_argument, NULL, 'O' },
        { "summary-prof", required_argument, NULL, 'p' },
        { "detailed-prof", required_argument, NULL, 'P' },
        { "settings", required_argument, NULL, 's' },
        { "out-suffix", required_argument, NULL, 'S' },
        { "trace", no_argument, NULL, 't' },
        { "wait", no_argument, NULL, 'w' },
        { "extended-prof", no_argument, NULL, 'x' },
    };

    while ((ch = getopt_long(argc, argv, "a:B:d:e:f:g:hi:l:m:r:M:o:O:p:P:R:s:S:twx", long_options, NULL)) != -1) {
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
        case 'f':
            input_file = strdup(optarg);
            opt = FEM;
            break;
        case 'g':
            generator = get_generator(optarg);
            break;
        case 'i':
            info_file_name = strdup(optarg);
            break;
        case 'l':
            input_file = strdup(optarg);
            opt = LAP_3D;
            break;
        case 'm':
            input_file = strdup(optarg);
            opt = MTX;
            break;
        case 'r':
            rhs_file = strdup(optarg);
            break;
        case 'M':
            detailed_metrics_prefix = optarg;
            break;
        case 'o':
            output_file_name = strdup(optarg);
            break;
        case 'R':
            rhs_output_file_name = strdup(optarg);
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
        case 't':
            trace_enabled = true;
            break;
        case 'w':
            wait = true;
            break;
        case 'x':
            detailed_prof = true;
            break;
        case 'h':
        default:
            DIE(USAGE, argv[0]);
        }
    }

    if (opt == NONE) {
        printf("No input specified\n");
        DIE(USAGE, argv[0]);
    }

    if (settings_file.empty()) {
        printf("No settings file specified\n");
        DIE(USAGE, argv[0]);
    }

    if (opt == LAP_3D && generator == INVALIG_GEN) {
        printf("Invalid lap 3d generator\n");
        DIE(USAGE, argv[0]);
    }

    // -------------------------------------------------------------------------
    // Initialize MPI
    // -------------------------------------------------------------------------

    BCM::init(&argc, &argv, log_file_name);
    _MPI_ENV;

    // -------------------------------------------------------------------------
    // Initialize MPI
    // -------------------------------------------------------------------------

    if (wait) {
        create_pid_file(myid);
        sleep(60); // Pausa 1 minuto
    }

    // -------------------------------------------------------------------------
    // Load configuration
    // -------------------------------------------------------------------------

    // InputParameters ip = ends_with(settings_file, ".properties")
    //     ? Params::initFromPropertiesFile(settings_file.c_str())
    //     : Params::initFromFile(settings_file.c_str());

    InputParameters ip = Params::initFromPropertiesFile(settings_file.c_str());
    if (ip.error != 0) {
        return -1;
    }

    // -------------------------------------------------------------------------
    // Rhs. Fem returns that too.
    // -------------------------------------------------------------------------
    vector<vtype>* rhs = NULL;

    // -------------------------------------------------------------------------
    // Read/Generate input matrix
    // -------------------------------------------------------------------------

    CSR* Alocal = NULL;
    if (opt == MTX) { // The master reads the matrix and distributes it.
        Alocal = read_local_matrix_from_mtx(input_file);
    } else if (opt == LAP_3D) {
        Alocal = generate_lap3d_local_matrix(generator, input_file);
    } else if (opt == LAP) {
        Alocal = generate_lap_local_matrix(n);
    } else if (opt == FEM) {
        Alocal = generate_fem_local_matrix(input_file, &rhs);
    }

    if (!dump_matrix_prefix.empty()) {
        char fname[256] = {0};
        snprintf(fname, 256, "%s%d.mtx", dump_matrix_prefix.c_str(), myid);
        // CSRMatrixPrintMM(Alocal, fname);
        CSRm::printMM(Alocal, fname, false);
    }

    itype full_n = Alocal->full_n;

    // -------------------------------------------------------------------------
    // Setup preconditioner
    // -------------------------------------------------------------------------

    handles* h = Handles::init();

    Preconditioner pr;
    if (ip.sprec != PreconditionerType::BCMG) {
        halo_info hi = haloSetup(Alocal, NULL);
        Alocal->halo = hi;
        shrink_col(Alocal, NULL);
    }

    pr.type = ip.sprec;

    if (ISMASTER) {
        printf("Setting preconditioner\n");
    }
    prec_setup(h, Alocal, &pr, ip);

    CUDA_FREE(buffer_4_getmct);
    CUDA_FREE(glob_d_BlocksCount);  
    CUDA_FREE(glob_d_BlocksOffset);

    // -------------------------------------------------------------------------
    // Initialize solution
    // -------------------------------------------------------------------------

    if (!rhs) {
        if (rhs_file) {
            vector<vtype>* rhs_full = Vector::load<vtype>(rhs_file, false);
            rhs = Vector::init<vtype>(Alocal->n, true, false);
            memcpy(rhs->val, rhs_full->val + Alocal->row_shift, Alocal->n * sizeof(vtype));
            Vector::free(rhs_full);
            vector<vtype>* rhs_d = Vector::copyToDevice(rhs);
            Vector::free(rhs);
            rhs = rhs_d;
        } else {
            rhs = Vector::init<vtype>(Alocal->n, true, true);
            #if 0
            unsigned long long seed = 1234ULL;
            Vector::fillWithRandomValues(rhs, 0., 1., seed);
            #else
            Vector::fillWithValue(rhs, 1.);
            #endif
        }
    }
    vector<vtype>* x0 = Vector::init<vtype>(Alocal->n, true, true);

    // -------------------------------------------------------------------------
    // Dump rhs
    // -------------------------------------------------------------------------

    if (rhs_output_file_name) {
        if (full_n <= MAX_AGGREGATED_SOLUTION_SIZE) {
            if (ISMASTER) {
                printf("Dumping rhs\n");
            }
            vector<vtype>* collectedRhs = aggregate_vector_all(rhs, full_n);
            if (ISMASTER) {
                FILE* rhs_output_file = fopen(rhs_output_file_name, "w");
                if (rhs_output_file == NULL) {
                    printf("Error opening %s for writing\n", rhs_output_file_name);
                } else {
                    Vector::print(collectedRhs, -1, rhs_output_file);
                    fclose(rhs_output_file);
                }
            }
            Vector::free(collectedRhs);
        } else {
            if (ISMASTER) {
                printf("WARNING: rhs cannot be aggregated when size > %ld\n", MAX_AGGREGATED_SOLUTION_SIZE);
            }
        }
    }

    // -------------------------------------------------------------------------
    // If LSGS with Chebyshev compute A max eigval
    // -------------------------------------------------------------------------

    CurrentParameters cp;

    for (SolverType st : ip.solver_types) {
    	if ((st == SolverType::LSGS || st == SolverType::CGSN ) && ip.use_chebyshev == 1) {
    		if (ISMASTER) {
    			printf("Compute A max eigval, tol: %lf\n",ip.power_method_tol);
    		}
    		ip.A_max_eigval = power_method(h,Alocal,ip,&pr,cp);
    		if (ISMASTER) printf("max_eigval %.10lf\n",ip.A_max_eigval);
    		break;
    	}
    }

    // -------------------------------------------------------------------------

    bool append = false;
    for (SolverType st : ip.solver_types) {
        cp.solver_type = st;
        for (int sstep : ip.ssteps) {
            cp.sstep = sstep;

            {
                // -------------------------------------------------------------------------
                // Solve
                // -------------------------------------------------------------------------

                std::string solverStr = is_cgs(st)
                    ? solver_type_to_string(st) + std::to_string(sstep)
                    : solver_type_to_string(st);

                if (!rhs_file) {
                    #if 0
                    unsigned long long seed = 1234ULL;
                    Vector::fillWithRandomValues(rhs, 0., 1., seed);
                    #else
                    Vector::fillWithValue(rhs, 1.);
                    #endif
                }
                Vector::fillWithValue(x0, 0.);

                if (ISMASTER) {
                    printf("\nSolving (%s)\n", solverStr.c_str());
                }
                SolverOut solverOut;
                beginProfiling(__FILE__, __FUNCTION__, solverStr.c_str());
                vector<vtype>* sol = solve(h, Alocal, rhs, x0, ip, cp, pr, &solverOut);
                endProfiling(__FILE__, __FUNCTION__, solverStr.c_str());
                
                // -------------------------------------------------------------------------
                // Output
                // -------------------------------------------------------------------------

                if (info_file_name) {
                    if (ISMASTER) {
                        printf("Dumping output (%s)\n", solverStr.c_str());
                    }
                    if (ISMASTER) {
                        dump(info_file_name, ip, cp, pr, &solverOut, append);
                        append = true;
                    }
                }

                // -------------------------------------------------------------------------
                // Aggregate and dump solution
                // -------------------------------------------------------------------------

                if (output_file_name) {
                    if (full_n <= MAX_AGGREGATED_SOLUTION_SIZE) {
                        if (ISMASTER) {
                            printf("Dumping solution (%s)\n", solverStr.c_str());
                        }
                        vector<vtype>* collectedSol = aggregate_vector_all(sol, full_n);
                        if (ISMASTER) {
                            std::string output_filename_aux = filename_for(output_file_name, solverStr);
                            FILE* output_file = fopen(output_filename_aux.c_str(), "w");
                            if (output_file == NULL) {
                                printf("Error opening %s for writing\n", output_filename_aux.c_str());
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
                // Release memory (local)
                // -------------------------------------------------------------------------

                if (ISMASTER) {
                    printf("Releasing memory (local, %s)\n", solverStr.c_str());
                }
                Vector::free(sol);
            }

            if (!is_cgs(st)) {
                break;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Profiling
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Profiling\n");
    }
    size_t summaryLen = 0;
    ProfInfoSummary *summary = computeLocalProfilingInfoSummary(&summaryLen);

    if (!detailed_profile_prefix.empty()) {
        std::string filename = detailed_profile_prefix + std::to_string(myid);
        dumpLocalProfilingInfo(filename.c_str(), summary, summaryLen);
    }

    if (!summary_profile_prefix.empty()) {
        std::string filename = summary_profile_prefix + std::to_string(myid);
        FILE* file = fopen(filename.c_str(), "w");
        if (file == NULL) {
            printf("Error opening %s for writing\n", filename.c_str());
            dumpLocalProfilingInfoSummary(stderr, summary, summaryLen);
        } else {
            dumpLocalProfilingInfoSummary(file, summary, summaryLen);
            fclose(file);
        }
    }

    if (info_file_name) {
        if (ISMASTER) {
            printf("Dumping profile info\n");
        }
        dumpProfilingInfo(info_file_name, summary, summaryLen);
    }

    delete(summary);

    // -------------------------------------------------------------------------
    // Finalize preconditioner
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("\nFinalizing preconditioner\n");
    }
    prec_finalize(Alocal, &pr, ip);

    // -------------------------------------------------------------------------
    // Release memory (global)
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Releasing memory (global)\n");
    }
    Vector::free(rhs);
    Vector::free(x0);
    CSRm::free(Alocal);
    Handles::free(h);

    // -------------------------------------------------------------------------
    // Shutdown
    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Shutdown\n");
    }
    BCM::shutdown();
    return 0;
}
