#pragma once
#include <mpi.h>

#define ISMASTER (myid == 0)

#define _MPI_ENV                          \
    int myid, nprocs;                     \
    MPI_Comm_rank(MPI_COMM_WORLD, &myid); \
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs)

#define CHECK_MPI(X)                                                                       \
    {                                                                                      \
        if (X != MPI_SUCCESS) {                                                            \
            fprintf(stderr, "[ERROR MPI] :\n\t LINE: %d; FILE: %s\n", __LINE__, __FILE__); \
            exit(1);                                                                       \
        }                                                                                  \
    }

// --------------- PICO ------------------

#define MPI_CHOICE(T, P)                                              \
    {                                                                 \
        if (ISMASTER) {                                               \
            printf(T);                                                \
            fflush(stdout);                                           \
            scanf("%d", &P);                                          \
        }                                                             \
        MPI_Barrier(MPI_COMM_WORLD);                                  \
        CHECK_MPI(                                                    \
            MPI_Bcast(&P, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD)); \
        MPI_Barrier(MPI_COMM_WORLD);                                  \
    }

// ---------------------------------------

void StartMpi(int* id, int* np, int* argc, char*** argv);
