#include "datastruct/CSR.h"
#include "halo_communication/local_permutation.h"
#include "op/spspmpi.h"
#include "utility/col8.h"
#include "utility/handles.h"
#include "utility/globals.h"
#include "utility/mpi.h"
#include "utility/utils.h"

#define GLOB_MEM_ALLOC_SIZE 2000000

extern itype *iAtemp1;
extern vtype *vAtemp1;
extern itype *idevtemp1;
extern vtype *vdevtemp1;
extern itype *idevtemp2;

int main(int argc, char **argv) {
    BCM::init(&argc, &argv, "out/testShrinkLog");
    output_dir = "./out";
    _MPI_ENV;
    
    char matrix_path[1024] = {0};
    snprintf(matrix_path, 1023, "src/test/data/mtx/Alocal_myid%d_nprocs%d.mtx", myid, nprocs);
    fprintf(log_file, "Alocal %d\n", myid);
    CSR *dA = read_matrix_from_file(matrix_path, 0, true);
    if (myid) {
        int shift = 4; // *** <== Setting this to 4 instead of 2 will pass the test
        CSRm::shift_cols(dA, -shift);
        dA->row_shift = shift;
        dA->col_shifted = -shift;
    }
    fprintf(log_file, "n=%d\n", dA->n);
    fprintf(log_file, "m=%d\n", dA->m);
    fprintf(log_file, "full_n=%d\n", dA->full_n);
    fprintf(log_file, "row_shift=%d\n", dA->row_shift);
    
    if (ISMASTER) {
        fprintf(stderr, "\nA:\n");
    }
    CSRm::debug(dA, stderr);

    // CSRm::printMM(dA, log_file);
    CSRm::print(dA, 3, -1, log_file);

    snprintf(matrix_path, 1023, "src/test/data/mtx/Plocal_myid%d_nprocs%d.mtx", myid, nprocs);
    fprintf(log_file, "Plocal %d\n", myid);
    CSR *dP = read_matrix_from_file(matrix_path, 0, true);
    if (myid) {
        int shift = 4;
        CSRm::shift_cols(dP, -shift);
        dP->row_shift = shift;
        dP->col_shifted = -shift;
    }
    fprintf(log_file, "n=%d\n", dP->n);
    fprintf(log_file, "m=%d\n", dP->m);
    fprintf(log_file, "full_n=%d\n", dP->full_n);
    fprintf(log_file, "row_shift=%d\n", dP->row_shift);
    
    if (ISMASTER) {
        fprintf(stderr, "\nP:\n");
    }
    CSRm::debug(dP, stderr);

    // CSRm::printMM(dP, log_file);
    CSRm::print(dP, 3, -1, log_file);

    handles* h = Handles::init();

    iAtemp1 = CUDA_MALLOC_HOST(itype, GLOB_MEM_ALLOC_SIZE, true);
    vAtemp1 = CUDA_MALLOC_HOST(vtype, GLOB_MEM_ALLOC_SIZE, true);
    idevtemp1 = CUDA_MALLOC(itype, GLOB_MEM_ALLOC_SIZE, true);
    vdevtemp1 = CUDA_MALLOC(vtype, GLOB_MEM_ALLOC_SIZE, true);
    idevtemp2 = CUDA_MALLOC(itype, GLOB_MEM_ALLOC_SIZE, true);

    CSR *dAP = CSRm::product(h, dA, dP);

    ASSERT(CSRm::checkProduct(h, dA, dP, dAP));

    if (SPSP_LIB == SpSpLib::CUSPARSE) {
        void spgemmcusparseFree();
        spgemmcusparseFree();
    }

    CUDA_FREE_HOST(iAtemp1);
    CUDA_FREE_HOST(vAtemp1);
    CUDA_FREE(idevtemp1);
    CUDA_FREE(vdevtemp1);
    CUDA_FREE(idevtemp2);

    CSRm::free(dAP);
    CSRm::free(dP);
    CSRm::free(dA);
    
    BCM::shutdown();
    return 0;
}