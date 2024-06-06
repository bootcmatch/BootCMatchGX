#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "utility/distribuite.h"
#include "utility/utils.h"
#include <assert.h>
#include <getopt.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    int myid, nprocs, device_id;
    StartMpi(&myid, &nprocs, &argc, &argv);

    size_t n = 1024;
    size_t full_n = nprocs * n;

    vector<vtype>* sol = Vector::init<vtype>(n, true, true);
    Vector::fillWithValue(sol, 1.);

    taskmap = (int*)Malloc(nprocs * sizeof(*taskmap));
    itaskmap = (int*)Malloc(nprocs * sizeof(*itaskmap));
    if (taskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for taskmap\n", nprocs * sizeof(*taskmap));
        exit(1);
    }
    if (itaskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for itaskmap\n", nprocs * sizeof(*itaskmap));
        exit(1);
    }
    for (int i = 0; i < nprocs; i++) {
        taskmap[i] = i;
        itaskmap[i] = i;
    }

    std::cerr << "Before aggregate vector\n";
    vector<vtype>* collectedSol = aggregate_vector(sol, full_n);
    std::cerr << "After aggregate vector\n";

    Vector::free(sol);

    if (ISMASTER) {
        Vector::print(collectedSol, -1, stdout);
    }

    Vector::free(collectedSol);
    MPI_Finalize();
    return 0;
}
