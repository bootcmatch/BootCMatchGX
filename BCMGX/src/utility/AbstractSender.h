#pragma once

#include "utility/MpiBuffer.h"
#include "utility/ProcessSelector.h"
#include "utility/arrays.h"
#include <string>

template <typename T>
class AbstractSender {
protected:
    bool use_row_shift;
    FILE* debug;
    MPI_Datatype mpi_data_type;

public:
    AbstractSender(FILE* debug, MPI_Datatype mpi_data_type)
        : use_row_shift(false)
        , debug(debug)
        , mpi_data_type(mpi_data_type)
    {
    }

    virtual int getProcessForItem(int index, T item) = 0;

    virtual T mapItemForProcess(T item, int proc)
    {
        return item;
    }

    virtual std::string toString(T item) = 0;
    virtual void debugItems(const char* title, T* arr, size_t len, bool isOnDevice) = 0;

    void setUseRowShift(bool use_row_shift)
    {
        this->use_row_shift = use_row_shift;
    }

    void send(T* items, size_t size,
        MpiBuffer<T>* sendBuffer, MpiBuffer<T>* rcvBuffer)
    {
        _MPI_ENV;

        // Iterate over items in order to find how many items
        // should be sent to each process.
        // ---------------------------------------------------------------------------
        itype item2proc[size] = { 0 };

        int whichproc = 0;
        for (int i = 0; i < size; i++) {
            // Find the process item i must be sent to
            whichproc = getProcessForItem(i, items[i]);

            if (debug) {
                fprintf(debug, "Element %s should be sent to process %d\n",
                    toString(items[i]).c_str(),
                    whichproc);
            }

            // whichproc=taskmap[whichproc];
            if (whichproc == myid) {
                fprintf(stderr,
                    "Task %d, unexpected whichproc for item %s\n",
                    myid, toString(items[i]).c_str());
                exit(EXIT_FAILURE);
            }

            sendBuffer->counter[whichproc]++;
            item2proc[i] = whichproc;
            sendBuffer->size++;
        }

        if (debug) {
            debugArray("items to be sent to process %d = %d\n", sendBuffer->counter, nprocs, false, debug);
        }

        // Compute offsets to be used in order to fill sendBuffer->buffer,
        // where items to be sent are grouped by receiver (process id).
        //
        // Examples with 3 processes:
        //                     +------------------+------------------+------------------+
        // sendBuffer->buffer: | items for proc 0 | items for proc 1 | items for proc 2 |
        //                     +------------------+------------------+------------------+
        // sendBuffer->offset  ^                  ^                  ^
        //
        // sendBuffer->counter [ # items for p 0  , # items for p 1  , # items for p 2  ]
        //
        // ---------------------------------------------------------------------------
        for (int i = 1; i < nprocs; i++) {
            sendBuffer->offset[i] = sendBuffer->offset[i - 1] + sendBuffer->counter[i - 1];
            sendBuffer->counter[i - 1] = 0;
        }
        sendBuffer->counter[nprocs - 1] = 0;

        if (debug) {
            debugArray("send offset for process %d = %d\n", sendBuffer->offset, nprocs, false, debug);
        }

        sendBuffer->init();
        for (int i = 0; i < size; i++) {
            whichproc = item2proc[i];
            sendBuffer->buffer[sendBuffer->offset[whichproc] + sendBuffer->counter[whichproc]] = mapItemForProcess(items[i], whichproc);
            sendBuffer->counter[whichproc]++;
        }
        if (sendBuffer->counter[myid] != 0) {
            fprintf(stderr, "self sendBuffer->counter should be zero! myid = %d\n", myid);
            exit(EXIT_FAILURE);
        }

        if (debug) {
            debugItems("sendBuffer->buffer", sendBuffer->buffer, sendBuffer->size, false);
            debugArray("sendBuffer->counter[%d] = %d\n", sendBuffer->counter, nprocs, false, debug);
        }

        // Processes must exchange the number of items to be transferred.
        // ---------------------------------------------------------------------------
        if (MPI_Alltoall(sendBuffer->counter, sizeof(itype), MPI_BYTE,
                rcvBuffer->counter, sizeof(itype), MPI_BYTE, MPI_COMM_WORLD)
            != MPI_SUCCESS) {
            fprintf(stderr, "Error in MPI_Alltoall, exchanging counters\n");
            exit(EXIT_FAILURE);
        }
        if (rcvBuffer->counter[myid] != 0) {
            fprintf(stderr, "self rcvBuffer->counter should be zero! %d\n", myid);
            exit(EXIT_FAILURE);
        }

        if (debug) {
            debugArray("rcvBuffer->counter[%d] = %d\n", rcvBuffer->counter, nprocs, false, debug);
        }

        // Compute offsets to be used in order to fill rcvBuffer->buffer,
        // where items to be sent are grouped by sender (process id).
        // ---------------------------------------------------------------------------
        rcvBuffer->size = rcvBuffer->counter[0];
        for (int i = 1; i < nprocs; i++) {
            rcvBuffer->offset[i] = rcvBuffer->offset[i - 1] + rcvBuffer->counter[i - 1];
            rcvBuffer->size += rcvBuffer->counter[i];
        }

        if (debug) {
            debugArray("rcvBuffer->offset[%d] = %d\n", rcvBuffer->offset, nprocs, false, debug);
            // fprintf(stderr, "pid: %d, sendBuffer->size: %d, rcvBuffer->size: %d\n", myid, sendBuffer->size, rcvBuffer->size);
        }

        // Exchange data
        // ---------------------------------------------------------------------------
        rcvBuffer->init();
        if (MPI_Alltoallv(
                sendBuffer->buffer,
                sendBuffer->counter,
                sendBuffer->offset,
                mpi_data_type,
                rcvBuffer->buffer,
                rcvBuffer->counter,
                rcvBuffer->offset,
                mpi_data_type,
                MPI_COMM_WORLD)
            != MPI_SUCCESS) {
            fprintf(stderr, "Error in MPI_Alltoallv of whichprow rows\n");
            exit(1);
        }

        if (debug) {
            debugItems("rcvBuffer->buffer", rcvBuffer->buffer, rcvBuffer->size, false);
        }
    }
};