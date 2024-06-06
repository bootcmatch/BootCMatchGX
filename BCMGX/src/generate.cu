#include <stdio.h>

#include "datastruct/CSR.h"
#include "utility/globals.h"
#include <getopt.h>

#define USAGE                                 \
    "Usage: %s [options] --out <FILE_NAME>\n" \
    "Options:\n"                              \
    "\t--sparse     Generate sparse matrix\n" \
    "\t--size <N>   Matrix size\n"            \
    "\t--verbose    Verbose mode\n"           \
    "\n"

typedef enum {
    BCM_SUCCESS,
    BCM_MEMORY_ALLOCATION_ERROR,
    BCM_IO_ERROR,
    BCM_SINGULAR_MATRIX_ERROR
} bcmError_t;

/**
 * Fill matrix with numbers from 0 to N^2.
 */
bcmError_t fillDenseMatrix(CSR** ret, int N)
{
    stype rows = N;
    gstype cols = N;

    CSR* A = CSRm::init(rows,
        cols,
        rows * cols, /* nnz */
        true, /* allocate_mem */
        false, /* on_the_device */
        false, /* symmetric */
        0, /* full_n */
        0); /* row_shift */
    if (A == NULL) {
        return BCM_MEMORY_ALLOCATION_ERROR;
    }

    long counter = 1;
    A->row[0] = 0;
    for (stype r = 0; r < rows; r++) {
        for (gstype c = 0; c < cols; c++) {
            A->val[counter - 1] = (vtype)counter;
            A->col[counter - 1] = c;
            counter++;
        }
        A->row[r + 1] = A->row[r] + cols;
    }

    *ret = A;
    return BCM_SUCCESS;
}

bcmError_t fillSparseMatrix(CSR** ret, int N)
{
    stype rows = N;
    gstype cols = N;

    CSR* A = CSRm::init(rows,
        cols,
        N + 2 * (N - 1), /* nnz */
        true, /* allocate_mem */
        false, /* on_the_device */
        false, /* symmetric */
        0, /* full_n */
        0); /* row_shift */
    if (A == NULL) {
        return BCM_MEMORY_ALLOCATION_ERROR;
    }

    // int coef = cols;
    int coef = 10;

    long counter = 0;
    A->row[0] = 0;
    for (stype r = 0; r < rows; r++) {
        if ((int)r - 1 >= 0) {
            stype c = r - 1;
            A->val[counter] = (vtype)(r * coef + c + 1);
            A->col[counter] = c;
            counter++;
        }
        {
            stype c = r;
            A->val[counter] = (vtype)(r * coef + c + 1);
            A->col[counter] = c;
            counter++;
        }
        if (r + 1 <= cols) {
            stype c = r + 1;
            A->val[counter] = (vtype)(r * coef + c + 1);
            A->col[counter] = c;
            counter++;
        }
        A->row[r + 1] = counter;
    }

    *ret = A;
    return BCM_SUCCESS;
}

int main(int argc, char** argv)
{
    char* out_file = NULL;
    int opt;
    int sparse = 0;
    int verbose = 0;
    int N = 10;

    static struct option long_options[] = {
        { "out", required_argument, NULL, 'o' },
        { "size", required_argument, NULL, 'n' },
        { "sparse", no_argument, NULL, 's' },
        { "verbose", no_argument, NULL, 'v' },
        { "help", no_argument, NULL, 'h' }
    };

    while ((opt = getopt_long(argc, argv, "o:n:s:v:h", long_options, NULL)) != -1) {
        switch (opt) {
        case 'o':
            out_file = strdup(optarg);
            break;
        case 'n':
            N = atoi(optarg);
            break;
        case 's':
            sparse = 1;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'h':
        default:
            printf(USAGE, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (out_file == NULL) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    CSR* A = NULL;
    printf("Generating matrix\n");
    // bcmError_t err = testMatrix(&A, 10);
    bcmError_t err = sparse
        ? fillSparseMatrix(&A, N)
        : fillDenseMatrix(&A, N);
    if (err != BCM_SUCCESS) {
        printf("Error allocating test matrix\n");
        exit(EXIT_FAILURE);
    }
    printf("Done\n");

    if (verbose) {
        CSRm::print(A, 3, 0, stderr);
    }

    printf("Saving matrix to <%s>.\n", out_file);
    CSRMatrixPrintMM(A, out_file);
    printf("Done\n");

    return 0;
}
