/**
 * @file
 */
#pragma once

#include "utility/setting.h"
#include <stdio.h>

struct CSR;

/**
 * @brief Class responsible for selecting the correct process based on row assignments.
 * 
 * The `ProcessSelector` class handles the selection of processes based on row assignments.
 * It gathers information on how many rows each process is assigned and computes the last row
 * index assigned to each process. This information is used to determine which process should
 * handle a specific row, optionally using a row shift to adjust the selection.
 */
class ProcessSelector {
private:
    bool use_row_shift; /**< Flag to determine if row shift should be used */
    stype* rows_per_process; /**< Array storing the number of rows assigned to each process */
    gstype* row_shift_per_process; /**< Array storing the row shift assigned to each process */
    gstype* last_row_index_per_process; /**< Array storing the last row index assigned to each process */
    FILE* debug; /**< File pointer for debugging output */
    int nprocs; /**< Number of processes in the MPI communicator */

public:
    gstype row_shift; /**< The row shift used for adjusting row assignments */

    /**
     * @brief Constructor for ProcessSelector.
     * 
     * This constructor initializes the `ProcessSelector` class by gathering necessary 
     * information from all processes in the MPI communicator, such as the number of rows 
     * assigned to each process, the row shifts, and the last row index assigned to each process.
     * 
     * @param dlA Pointer to the CSR matrix data structure containing the row assignments.
     * @param debug A file pointer for debugging output. If NULL, no debugging output is produced.
     */
    ProcessSelector(CSR* dlA, FILE* debug);

    /**
     * @brief Destructor for ProcessSelector.
     * 
     * Frees any dynamically allocated memory used by the `ProcessSelector` class.
     */
    ~ProcessSelector();

    /**
     * @brief Returns the process responsible for a specific row.
     * 
     * Given a row index, this function determines which process should handle the row.
     * If the row shift is enabled, the row index is adjusted by the row shift value before
     * determining the responsible process.
     * 
     * @param row The row index whose corresponding process is being requested.
     * 
     * @return The process ID (rank) responsible for handling the given row.
     */
    int getProcessByRow(gsstype row);

    /**
     * @brief Sets whether to use the row shift when selecting processes.
     * 
     * This function sets the `use_row_shift` flag, which determines if the row selection should
     * be adjusted by the row shift value.
     * 
     * @param use_row_shift Boolean flag to enable or disable row shift.
     */
    void setUseRowShift(bool use_row_shift);
};
