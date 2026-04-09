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

// return S = I - omega invD A
CSR *getSmoother(CSR *A, vector<vtype> *diag) {
    CSR *invDA = CSRm::diagonalScaling(A, diag);
    CSRm::debug(invDA, stderr);
    
    double omega = 4.0 / (3.0 * CSRm::globalInfinityNorm(invDA));
    printf("omega = %lf\n", omega);

    CSRm::forEach(invDA, [omega]__device__(CSR *invDA, itype irow, size_t innz) {
        gsstype col8 = invDA->col8[innz];
        invDA->val[innz] *= -omega;
        if (irow == col8) {
            invDA->val[innz] += 1;
        }
        vtype val = invDA->val[innz];
        // printf("CSR[%ld, %ld] = %lf\n", (gsstype)(irow + invDA->row_shift), col8 - invDA->col_shifted, val);
    });

    return invDA;
}

// A (new) = I - omega invD A (old)
void getSmootherInPlace(CSR *A, vector<vtype> *diag) {
    CSRm::diagonalScalingInPlace(A, diag);
    CSRm::debug(A, stderr);
    
    double omega = 4.0 / (3.0 * CSRm::globalInfinityNorm(A));
    printf("omega = %lf\n", omega);

    CSRm::forEach(A, [omega]__device__(CSR *A, itype irow, size_t innz) {
        gsstype col8 = A->col8[innz];
        A->val[innz] *= -omega;
        if (irow == col8) {
            A->val[innz] += 1;
        }
        vtype val = A->val[innz];
        // printf("CSR[%ld, %ld] = %lf\n", (gsstype)(irow + A->row_shift), col8 - A->col_shifted, val);
    });
}

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

    CSR* hA = CSRm::init(
        n, // Rows
        col, // Cols
        max_nnz, /* nnz */
        true, /* allocate_mem */
        false, /* on_the_device */
        false, /* symmetric */
        full_n, /* full_n */
        row_shift); /* row_shift */
    
    hA->nnz = 0; // This will force fillWithValues() to set the correct number

    CSRm::fillWithValues(hA, Aglobal);
    CSRm::debug(hA, stderr);
    
    CSR *A = CSRm::copyToDevice(hA);
    CSRm::free(hA);

    vector<vtype> *diag = CSRm::diag(A);
    Vector::debug(diag, stderr);

    CSR *S = getSmoother(A, diag);
    CSRm::debug(S, stderr);

    getSmootherInPlace(A, diag);
    CSRm::debug(A, stderr);

    // -------------------------------------------------------------------------
    // Finalize
    // -------------------------------------------------------------------------

    CSRm::free(A);
    CSRm::free(S);
    Vector::free(diag);
    Handles::free(h);
    MPI_Finalize();
    return 0;
}
