#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "utility/distribuite.h"
#include "utility/memory.h"
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

    taskmap = MALLOC(int, nprocs);
    itaskmap = MALLOC(int, nprocs);

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
