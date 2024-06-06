#include "datastruct/CSR.h"
#include "utility/ProcessSelector.h"
#include "utility/arrays.h"

/**
 * Function prototype.
 * Given an ordered array arr of length len, returns the index
 * of the highest element less than val.
 * Implementation in spspmpi.cu.
 */
int bswhichprocess(gsstype* arr, int len, gsstype val);

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
    rows_per_process = (stype*)Malloc(nprocs * sizeof(stype));
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
    row_shift_per_process = (gstype*)Malloc(nprocs * sizeof(gstype));
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
    last_row_index_per_process = (gsstype*)Malloc(nprocs * sizeof(gsstype));
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
    ::Free(last_row_index_per_process);
    ::Free(rows_per_process);
    ::Free(row_shift_per_process);
}

void ProcessSelector::setUseRowShift(bool use_row_shift)
{
    this->use_row_shift = use_row_shift;
}

int ProcessSelector::getProcessByRow(itype row)
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
        fprintf(stderr,
            "Asking row %d to process %d, but process %d owns rows [%d, %d]\n",
            row,
            whichproc,
            whichproc,
            row_shift_per_process[whichproc],
            row_shift_per_process[whichproc] + rows_per_process[whichproc] - 1);
        exit(1);
    }

    return whichproc;
}
