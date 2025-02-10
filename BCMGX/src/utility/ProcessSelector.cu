#include "datastruct/CSR.h"
#include "utility/ProcessSelector.h"
#include "utility/arrays.h"
#include "utility/bswhichprocess.h"
#include "utility/memory.h"

ProcessSelector::ProcessSelector(CSR* dlA, FILE* debug)
    : debug(debug)
    , use_row_shift(false)
{
    _MPI_ENV;

    this->nprocs = nprocs;
    this->row_shift = dlA->row_shift;

    // Each process communicates the number of rows assigned to it
    // (dlA->n) and receives the nr of rows assigned to other processes
    // (rows_per_process).
    // This piece of information will be used to determine which process
    // should receive previously collected data.
    // ---------------------------------------------------------------------------
    rows_per_process = MALLOC(stype, nprocs);
    CHECK_MPI(MPI_Allgather(
        &dlA->n, // Sent buffer
        sizeof(stype), // Sent count
        MPI_BYTE, // Sent count datatype
        rows_per_process, // Receive buffer
        sizeof(stype), // Receive count
        MPI_BYTE, // Receive count datatype
        MPI_COMM_WORLD // Communicator
        ));

    if (debug) {
        debugArray("rows assigned to process %d: %d\n", rows_per_process, nprocs, false, debug);
    }

    // ---------------------------------------------------------------------------
    row_shift_per_process = MALLOC(gstype, nprocs);
    CHECK_MPI(MPI_Allgather(
        &dlA->row_shift, // Sent buffer
        sizeof(gstype), // Sent count
        MPI_BYTE, // Sent count datatype
        row_shift_per_process, // Receive buffer
        sizeof(gstype), // Receive count
        MPI_BYTE, // Receive count datatype
        MPI_COMM_WORLD // Communicator
        ));

    // Compute the last row index assigned to each process.
    // This piece of information will be used to determine which process
    // should receive previously collected data.
    // ---------------------------------------------------------------------------
    last_row_index_per_process = MALLOC(gstype, nprocs);
    last_row_index_per_process[0] = rows_per_process[0] - 1;
    for (int i = 1; i < nprocs; i++) {
        last_row_index_per_process[i] = last_row_index_per_process[i - 1]
            + (gstype)rows_per_process[taskmap ? taskmap[i] : i];
    }

    if (debug) {
        debugArray("last row index assigned to process %d: %d\n", last_row_index_per_process, nprocs, false, debug);
    }
}

ProcessSelector::~ProcessSelector()
{
    FREE(last_row_index_per_process);
    FREE(rows_per_process);
    FREE(row_shift_per_process);
}

void ProcessSelector::setUseRowShift(bool use_row_shift)
{
    this->use_row_shift = use_row_shift;
}

int ProcessSelector::getProcessByRow(gsstype /*itype*/ row)
{
    if (use_row_shift) {
        row += row_shift;
    }

    int whichproc = bswhichprocess(
        last_row_index_per_process,
        nprocs,
        row);
    if (whichproc > nprocs - 1) {
        whichproc = nprocs - 1;
    }

    if (taskmap) {
        whichproc = taskmap[whichproc];
    }

    if (!(row_shift_per_process[whichproc] <= row && row < row_shift_per_process[whichproc] + rows_per_process[whichproc])) {
        _MPI_ENV;
        fprintf(stderr,
            "Process %d: Asking row %ld to process %d, but process %d owns rows [%lu, %lu]\n",
            myid,
            row,
            whichproc,
            whichproc,
            row_shift_per_process[whichproc],
            row_shift_per_process[whichproc] + rows_per_process[whichproc] - 1);
        if (ISMASTER) {
            debugArray("rows_per_process[%s] = %u", rows_per_process, nprocs, false, stderr);
            debugArray("row_shift_per_process[%s] = %lu", row_shift_per_process, nprocs, false, stderr);
            debugArray("last_row_index_per_process[%s] = %lu", last_row_index_per_process, nprocs, false, stderr);
        }
        exit(1);
    }

    return whichproc;
}
