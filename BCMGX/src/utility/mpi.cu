#include "utility/mpi.h"
#include <stdio.h>

void StartMpi(int* id, int* np, int* argc, char*** argv)
{
    int myid, nprocs;
    CHECK_MPI(MPI_Init(argc, argv)); /* initialize MPI environment */
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &myid)); /* get task number */
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nprocs)); /* get number of tasks */
    *id = myid;
    *np = nprocs;
}
