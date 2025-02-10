#include "datastruct/CSR.h"
#include "generator/laplacian.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/globals.h"
#include <algorithm>

#define USAGE "Usage:\n\t%s <OUT_MTX> <IN_MTX_1> ... <IN_MTX_N>\n"

bool matrix_sorter(CSR* lhs, CSR* rhs)
{
    return lhs->row_shift < rhs->row_shift;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    const char* out_file_name = argv[1];

    itype full_n = 0;
    itype tot_nnz = 0;
    CSR* matrices[argc - 2] = { 0 };
    for (int i = 2; i < argc; i++) {
        CSR* A = read_matrix_from_file(argv[i], 0, false);
        matrices[i - 2] = A;
        full_n += A->n;
        tot_nnz += A->nnz;
    }

    std::sort(matrices, matrices + (argc - 2), &matrix_sorter);

    CSR* res = CSRm::init(
        full_n,
        matrices[0]->m,
        tot_nnz,
        true, // allocate_mem
        false, // on_the_device
        false, // is_symmetric
        full_n,
        0 // row_shift
    );

    int nnz_shift = 0;
    int row_shift = 0;
    for (int i = 0; i < argc - 2; i++) {
        CSR* A = matrices[i];
        if (row_shift > 0) {
            for (int j = 0; j < A->n + 1; j++) {
                A->row[j] += nnz_shift;
            }
        }
        memcpy(res->row + row_shift, A->row, (A->n + 1) * sizeof(itype));
        memcpy(res->col + nnz_shift, A->col, A->nnz * sizeof(itype));
        memcpy(res->val + nnz_shift, A->val, A->nnz * sizeof(vtype));
        nnz_shift += A->nnz;
        row_shift += A->n;
        CSRm::free(A);
        matrices[i] = NULL;
    }

    check_and_fix_order(res);
    CSRMatrixPrintMM(res, out_file_name);
    CSRm::free(res);

    return 0;
}
