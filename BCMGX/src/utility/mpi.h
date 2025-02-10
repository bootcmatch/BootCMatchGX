/**
 * @file
 */
#pragma once
#include <mpi.h>

/**
 * @brief Macro to check if the current process is the master process.
 * 
 * This macro compares the rank of the current process (`myid`) to 0, which
 * is typically the master process in an MPI program. It is used to execute
 * code conditionally only on the master process.
 */
#define ISMASTER (myid == 0)

/**
 * @brief Macro to check the success of MPI function calls.
 * 
 * This macro checks if an MPI function call has succeeded by comparing its return
 * value to `MPI_SUCCESS`. If the function call failed, an error message is printed
 * containing the line and file number, and the program exits with a non-zero status.
 * 
 * @param X The MPI function call to check for success.
 */
#define CHECK_MPI(X)                                                                       \
    {                                                                                      \
        if (X != MPI_SUCCESS) {                                                            \
            fprintf(stderr, "[ERROR MPI] :\n\t LINE: %d; FILE: %s\n", __LINE__, __FILE__); \
            exit(1);                                                                       \
        }                                                                                  \
    }

/**
 * @brief Macro to initialize MPI environment and retrieve process information.
 * 
 * This macro initializes the MPI environment, retrieves the rank of the current
 * process (`myid`), and the total number of processes (`nprocs`) in the MPI communicator.
 * It is commonly used at the start of MPI programs to set up the environment.
 */
#define _MPI_ENV                                     \
    int myid, nprocs;                                \
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &myid)); \
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nprocs))

// --------------- PICO ------------------

/**
 * @brief Macro to perform an MPI-based interactive choice for the user.
 * 
 * This macro allows the master process to interactively print a message and read an integer
 * input from the user. The input is then broadcast to all processes in the MPI communicator
 * using `MPI_Bcast`. All processes synchronize using `MPI_Barrier` before and after the broadcast.
 * 
 * @param T A format string to be printed to the master process's console.
 * @param P The integer variable where the input value will be stored.
 */
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

/**
 * @brief Function to initialize MPI and get process information.
 * 
 * This function initializes the MPI environment and retrieves the rank of the current process
 * (`myid`) and the total number of processes (`nprocs`). It then stores these values in the 
 * provided pointers `id` and `np`, respectively.
 * 
 * @param id Pointer to an integer where the rank of the current process will be stored.
 * @param np Pointer to an integer where the total number of processes will be stored.
 * @param argc Pointer to the argument count for the MPI program (typically from `main`).
 * @param argv Pointer to the argument vector for the MPI program (typically from `main`).
 */
void StartMpi(int* id, int* np, int* argc, char*** argv);
