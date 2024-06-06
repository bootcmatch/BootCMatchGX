#pragma once

#include "utility/mpi.h"
#include "utility/utils.h"

template <typename T>
struct MpiBuffer {
    itype size;
    itype nprocs;
    T* buffer; // len: size
    itype* counter; // len: nproc
    itype* offset; // len: nproc

    MpiBuffer()
        : size(0)
        , buffer(NULL)
    {
        _MPI_ENV;

        this->nprocs = nprocs;

        counter = (itype*)Malloc(nprocs * sizeof(itype));
        offset = (itype*)Malloc(nprocs * sizeof(itype));

        memset(counter, 0, nprocs * sizeof(itype));
        memset(offset, 0, nprocs * sizeof(itype));
    }

    ~MpiBuffer()
    {
        if (buffer) {
            free(buffer);
        }

        if (counter) {
            free(counter);
        }

        if (offset) {
            free(offset);
        }
    }

    void init()
    {
        if (buffer) {
            free(buffer);
            buffer = NULL;
        }

        if (size) {
            buffer = (T*)Malloc(size * sizeof(T));
        }
    }
};
