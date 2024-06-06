#include "datastruct/CSR.h"
#include "generator/laplacian.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/globals.h"
#include <getopt.h>

#define USAGE                                         \
    "Usage: %s -i <FILE_NAME> -o <FILE_NAME>\n\n"     \
    "\t-i, --in  <FILE_NAME>      Input mtx file.\n"  \
    "\t-o, --out <FILE_NAME>      Output mtx file.\n" \
    "\t-l, --log <FILE_NAME>      Log file.\n"        \
    "\n"

int main(int argc, char** argv)
{
    char* log_file_name = NULL;
    char* in_file_name = NULL;
    char* out_file_name = NULL;

    static struct option long_options[] = {
        { "log", required_argument, NULL, 'l' },
        { "in", required_argument, NULL, 'i' },
        { "out", required_argument, NULL, 'o' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    signed char ch;
    while ((ch = getopt_long(argc, argv, "l:i:o:h", long_options, NULL)) != -1) {
        switch (ch) {
        case 'l':
            log_file_name = strdup(optarg);
            break;
        case 'i':
            in_file_name = strdup(optarg);
            break;
        case 'o':
            out_file_name = strdup(optarg);
            break;
        case 'h':
        default:
            printf(USAGE, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (in_file_name == NULL || out_file_name == NULL) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    if (log_file_name) {
        log_file = fopen(log_file_name, "w");
        if (!log_file) {
            printf("Error opening log file");
            exit(1);
        }
        if (atexit(close_log_file)) {
            fprintf(stderr, "Error registering atexit\n");
            exit(EXIT_FAILURE);
        }
    }

    CSR* A = read_matrix_from_file(in_file_name, 0, false);
    check_and_fix_order(A);
    if (log_file) {
        CSRm::print(A, 3, 0, log_file);
    }
    CSRMatrixPrintMM(A, out_file_name);

    CSRm::free(A);

    Free(log_file_name);
    Free(in_file_name);
    Free(out_file_name);

    return 0;
}
