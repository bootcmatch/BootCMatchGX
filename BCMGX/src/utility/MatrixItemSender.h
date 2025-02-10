/**
 * @file
 */
#pragma once

#include "datastruct/matrixItem.h"
#include "utility/AbstractSender.h"
#include "utility/ProcessSelector.h"
#include <string>

/**
 * @class MatrixItemSender
 * @brief A class responsible for sending matrix items to a process.
 * 
 * This class extends `AbstractSender` with a template specialization for `matrixItem_t`. It is designed to handle the
 * mapping, selection, and debugging of matrix items that need to be sent to processes. The class utilizes a `ProcessSelector`
 * to determine which process will handle each matrix item.
 */
class MatrixItemSender : public AbstractSender<matrixItem_t> {
private:
    ProcessSelector* processSelector; /**< Pointer to the ProcessSelector used for process selection */

public:
    /**
     * @brief Constructs a `MatrixItemSender` object.
     * 
     * This constructor initializes the `MatrixItemSender` with a `ProcessSelector` and a debug output file.
     * 
     * @param processSelector A pointer to the `ProcessSelector` used for selecting the process.
     * @param debug A pointer to a `FILE` object for debugging output.
     */
    MatrixItemSender(ProcessSelector* processSelector, FILE* debug);

    /**
     * @brief Gets the process index for a specific matrix item.
     * 
     * This method determines which process should handle the specified matrix item based on the given index and the
     * `matrixItem_t` object.
     * 
     * @param index The index of the matrix item.
     * @param item The `matrixItem_t` object representing the matrix item.
     * 
     * @return The index of the process that will handle the matrix item.
     */
    int getProcessForItem(int index, matrixItem_t item);

    /**
     * @brief Maps a matrix item to a specific process.
     * 
     * This method maps the given matrix item to the specified process index. It may modify the item in the process
     * of mapping it, based on whether row shifting is used.
     * 
     * @param item The `matrixItem_t` object to be mapped.
     * @param proc The index of the process to which the item is mapped.
     * 
     * @return The modified `matrixItem_t` object that is mapped to the specified process.
     */
    matrixItem_t mapItemForProcess(matrixItem_t item, int proc);

    /**
     * @brief Converts a matrix item to a string representation.
     * 
     * This method generates a string representation of a `matrixItem_t` object, useful for debugging and logging.
     * The representation includes the row, column, and value of the matrix item.
     * 
     * @param item The `matrixItem_t` object to be converted.
     * 
     * @return A string representation of the matrix item in the format: "(row, col)=value".
     */
    std::string toString(matrixItem_t item);

    /**
     * @brief Debugs matrix items.
     * 
     * This method outputs the contents of an array of `matrixItem_t` items to the debug output file. The array can
     * be on the device or the host. It utilizes the helper function `debugMatrixItems` for detailed logging.
     * 
     * @param title A title to print before the item details.
     * @param arr The array of `matrixItem_t` items to be debugged.
     * @param len The length of the array.
     * @param isOnDevice A flag indicating whether the items are on the device (GPU) or host (CPU).
     */
    void debugItems(const char* title, matrixItem_t* arr, size_t len, bool isOnDevice);
};
