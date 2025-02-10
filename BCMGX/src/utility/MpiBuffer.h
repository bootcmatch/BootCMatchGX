/**
 * @file
 */
#pragma once

#include "utility/memory.h"
#include "utility/mpi.h"
#include "utility/utils.h"

/**
 * @brief A structure for managing a buffer for MPI communication.
 * 
 * The `MpiBuffer` template structure is used for managing an MPI buffer that stores data
 * to be communicated between processes. It holds information about the size of the buffer,
 * the number of processes, and the necessary counters and offsets for each process.
 * This structure simplifies the allocation and initialization of buffers for communication
 * in parallel applications using MPI.
 * 
 * @tparam T The type of data stored in the buffer (e.g., `int`, `double`).
 */
template <typename T>
struct MpiBuffer {
    itype size; /**< The size of the buffer */
    itype nprocs; /**< The number of processes in the MPI communicator */
    T* buffer; /**< Pointer to the buffer of type `T` (length: `size`) */
    itype* counter; /**< Pointer to an array that stores the count of elements for each process (length: `nprocs`) */
    itype* offset; /**< Pointer to an array that stores the offset for each process (length: `nprocs`) */

    /**
     * @brief Default constructor for the `MpiBuffer` struct.
     * 
     * The constructor initializes the MPI environment, sets the number of processes (`nprocs`),
     * and allocates memory for the `counter` and `offset` arrays based on the number of processes.
     * The buffer itself is initialized to `NULL` and its size is set to `0`.
     */
    MpiBuffer()
        : size(0)
        , buffer(NULL)
    {
        _MPI_ENV;

        this->nprocs = nprocs;

        counter = MALLOC(itype, nprocs, true);
        offset = MALLOC(itype, nprocs, true);
    }

    /**
     * @brief Destructor for the `MpiBuffer` struct.
     * 
     * The destructor frees the allocated memory for the buffer, `counter`, and `offset` arrays.
     */
    ~MpiBuffer()
    {
        FREE(buffer);
        FREE(counter);
        FREE(offset);
    }

    /**
     * @brief Initializes the buffer by allocating memory.
     * 
     * This function frees any previously allocated memory for the buffer and reallocates memory
     * for it based on the size of the buffer. If `size` is non-zero, the buffer is allocated
     * with the type `T` (the type of data stored in the buffer).
     */
    void init()
    {
        FREE(buffer);
        if (size) {
            buffer = MALLOC(T, size, true);
        }
    }
};
