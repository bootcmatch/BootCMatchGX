#include "datastruct/CSR.h"
#include "datastruct/csrlocinfo.h"
#include "op/spspmpi.h"

CSR* nsparseMGPU(CSR*, CSR*, csrlocinfo*);

int main(int argc, char **argv) {
    if (getenv("SPSP_LIB")) {
        SPSP_LIB = string_to_spsplib(getenv("SPSP_LIB"));
    }

    char* log_file_name = NULL;
    BCM::init(&argc, &argv, log_file_name);

    ASSERT(argc >= 3);

    printf("Reading A from %s\n", argv[1]);
    CSR* A = read_matrix_from_file(argv[1], 0, true);

    printf("Reading P from %s\n", argv[2]);
    CSR* P = read_matrix_from_file(argv[2], 0, true);

    csrlocinfo info;
    info.fr = 0;
    info.lr = P->n;
    info.row = P->row;
    info.col = NULL;
    info.val = P->val;

    printf("Computing product\n");
    CSR* AP = nsparseMGPU(A, P, &info);
    ASSERT(AP->nnz);

    CSRm::free(A);
    CSRm::free(P);

    if (argc > 3) {
        CSRm::printMMsimple(AP, argv[3], false);
    }

    printf("[DONE]\n");
    BCM::shutdown();

    return 0;
}