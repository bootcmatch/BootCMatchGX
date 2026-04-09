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
#include "op/Anorm.h"
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

    std::initializer_list<vtype> xglobal = {
        1, -1, 2, 0
    };

    // -------------------------------------------------------------------------

    handles* h = Handles::init();

    CSR* h_Alocal = CSRm::init(
        full_n / nprocs, // Rows
        col, // Cols
        (full_n / nprocs) * col, /* nnz */
        true, /* allocate_mem */
        false, /* on_the_device */
        false, /* symmetric */
        full_n, /* full_n */
        (full_n / nprocs) * myid); /* row_shift */
    
    h_Alocal->nnz = 0; // This will force fillWithValues() to set the correct number

    CSRm::fillWithValues(h_Alocal, Aglobal);

    for (int i = 0; i < nprocs; i++) {
        if (myid == i) {
            CSRm::print(h_Alocal, 3, -1, stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    vector<vtype> *h_xlocal = Vector::init<vtype>(
        h_Alocal->n, // n
        true, // allocate_mem
        false // on_the_device
    );

    Vector::fillWithValues(h_xlocal, h_Alocal->full_n, xglobal);

    if (myid == 0) {
        fprintf(stderr, "\nx\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < nprocs; i++) {
        if (myid == i) {
            Vector::print(h_xlocal, -1, stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }


    CSR *d_Alocal = CSRm::copyToDevice(h_Alocal);
    vector<vtype> *d_xlocal = Vector::copyToDevice(h_xlocal);
    cudaDeviceSynchronize();

    CSRm::free(h_Alocal);
    Vector::free(h_xlocal);

    // -------------------------------------------------------------------------

    taskmap = MALLOC(int, nprocs);
    itaskmap = MALLOC(int, nprocs);
    for (int i = 0; i < nprocs; i++) {
        taskmap[i] = i;
        itaskmap[i] = i;
    }

    if (getenv("SCALENNZMISSING")) {
        scalennzmiss = atoi(getenv("SCALENNZMISSING"));
    }

    halo_info halo = haloSetup(d_Alocal, NULL);
    d_Alocal->halo = halo;
    shrink_col(d_Alocal, NULL);

    if (myid == 0) {
        fprintf(stderr, "\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < nprocs; i++) {
        if (myid == i) {
            CSRm::print(d_Alocal, 5, 0, stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // -------------------------------------------------------------------------

    vtype norm = Anorm(h->cublas_h, d_Alocal, d_xlocal);

    if (myid == 0) {
        fprintf(stderr, "\n\nNorm: %lf\n", norm);
    }

    // -------------------------------------------------------------------------
    // Finalize
    // -------------------------------------------------------------------------

    CSRm::free(d_Alocal);
    Vector::free(d_xlocal);
    Handles::free(h);
    MPI_Finalize();
    return 0;
}
