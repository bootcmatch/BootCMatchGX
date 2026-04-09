#include <assert.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#include "datastruct/CSR.h"
#include "halo_communication/halo_communication.h"
#include "halo_communication/local_permutation.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/distribute.h"
#include "utility/globals.h"
#include "utility/handles.h"
#include "utility/input.h"
#include "utility/mpi.h"

// Given A:
//     4 1 0 2
//     1 3 1 0
//     0 1 5 1
//     2 0 1 4
// and x = [1 -1 2 0]
// Ax should be [3 0 9 4]
// Anorm(A, x) = x^T(Ax) = sqrt(21) ~= 4.58
int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Initialize MPI
    // -------------------------------------------------------------------------

    // Start MPI
    int myid, nprocs, device_id;
    StartMpi(&myid, &nprocs, &argc, &argv);

    // -------------------------------------------------------------------------
    // Assign GPU
    // -------------------------------------------------------------------------

    int deviceCount = 0;
    CHECK_DEVICE(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount);
    device_id = assignDeviceToProcess();
    int assigned_device_id = device_id % deviceCount;
    fprintf(stderr, "Trying to set device %d. Total devices: %d. Assigned device: %d\n", device_id, deviceCount, assigned_device_id);
    CHECK_DEVICE(cudaSetDevice(assigned_device_id));

    // -------------------------------------------------------------------------
    // Init data
    // -------------------------------------------------------------------------

    std::initializer_list<vtype> Aglobal = {
        4, 1, 0, 2,
        1, 3, 1, 0,
        0, 1, 5, 1,
        2, 0, 1, 4
    };
    int full_n = 4;
    int col = 4;

    // -------------------------------------------------------------------------

    handles* h = Handles::init();

    int n = full_n / nprocs;
    int row_shift = n * myid;
    if (myid == nprocs - 1) {
        n += full_n % nprocs;
    }
    int max_nnz = n * col;

    CSR* h_Alocal = CSRm::init(
        n, // Rows
        col, // Cols
        max_nnz, /* nnz */
        true, /* allocate_mem */
        false, /* on_the_device */
        false, /* symmetric */
        full_n, /* full_n */
        row_shift); /* row_shift */
    
    h_Alocal->nnz = 0; // This will force fillWithValues() to set the correct number

    CSRm::fillWithValues(h_Alocal, Aglobal);
    CSRm::debug(h_Alocal, stderr);
    
    CSR *d_Alocal = CSRm::copyToDevice(h_Alocal);
    CSRm::free(h_Alocal);
    
    double coef = 0.5;
    CSRm::forEach(d_Alocal, [coef]__device__(CSR *A, itype irow, size_t innz) {
        gsstype col8 = A->col8[innz];
        A->val[innz] *= coef;
        vtype val = A->val[innz];
        printf("CSR[%ld, %ld] = %lf\n", (gsstype)(irow + A->row_shift), col8 - A->col_shifted, val);
    });

    CSRm::debug(d_Alocal, stderr);

    // -------------------------------------------------------------------------
    // Finalize
    // -------------------------------------------------------------------------

    CSRm::free(d_Alocal);
    Handles::free(h);
    MPI_Finalize();
    return 0;
}
