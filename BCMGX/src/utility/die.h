#pragma once
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>

void close_log_file();
void print_stacktrace(int nsymbols, FILE* out);

#define STACKTRACE_COUNT 10

#if defined(__GNUC__) || defined(__clang__)
#define UNREACHABLE() __builtin_unreachable()
#else
#define UNREACHABLE() exit(1)
#endif

#define DIE(...)                            \
    {                                       \
        cudaDeviceSynchronize();            \
        fprintf(stderr, __VA_ARGS__);       \
        print_stacktrace(                   \
            STACKTRACE_COUNT, stderr);      \
        close_log_file();                   \
        int _mpi_initialized = 0;           \
        MPI_Initialized(&_mpi_initialized); \
        if (_mpi_initialized) {             \
            MPI_Abort(MPI_COMM_WORLD, 1);   \
        } else {                            \
            exit(1);                        \
        }                                   \
        UNREACHABLE();                      \
    }

#define DIE_MASTER(...)                     \
    {                                       \
        cudaDeviceSynchronize();            \
        if (ISMASTER) {                     \
            fprintf(stderr, __VA_ARGS__);   \
        }                                   \
        print_stacktrace(                   \
            STACKTRACE_COUNT, stderr);      \
        close_log_file();                   \
        int _mpi_initialized = 0;           \
        MPI_Initialized(&_mpi_initialized); \
        if (_mpi_initialized) {             \
            MPI_Abort(MPI_COMM_WORLD, 1);   \
        } else {                            \
            exit(1);                        \
        }                                   \
        UNREACHABLE();                      \
    }
