#include "utility/mpi.h"

// #define ISMASTER (myid == 0)

// #define _MPI_ENV                          \
//     int myid, nprocs;                     \
//     MPI_Comm_rank(MPI_COMM_WORLD, &myid); \
//     MPI_Comm_size(MPI_COMM_WORLD, &nprocs)
// #define CHECK_MPI(X)                                                                       \
//     {                                                                                      \
//         if (X != MPI_SUCCESS) {                                                            \
//             fprintf(stderr, "[ERROR MPI] :\n\t LINE: %d; FILE: %s\n", __LINE__, __FILE__); \
//             exit(1);                                                                       \
//         }                                                                                  \
//     }

void StartMpi(int* id, int* np, int* argc, char*** argv)
{
    int myid, nprocs;
    MPI_Init(argc, argv); /* initialize MPI environment */
    MPI_Comm_rank(MPI_COMM_WORLD, &myid); /* get task number */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get number of tasks */
    *id = myid;
    *np = nprocs;
}
