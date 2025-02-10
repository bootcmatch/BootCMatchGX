/**
 * @file
 */
#pragma once

#include "utility/AbstractSender.h"
#include "utility/ProcessSelector.h"
#include "utility/setting.h"
#include <string>

/**
 * @class ColumnIndexSender
 * @brief Handles sending column indices to the appropriate processes.
 *
 * This class is responsible for determining the process responsible for a given column index
 * and mapping the column index appropriately when sending.
 */
class ColumnIndexSender : public AbstractSender<itype> {
private:
    ProcessSelector* processSelector; ///< Pointer to the process selector.

public:
    /**
     * @brief Constructs a ColumnIndexSender.
     * @param processSelector Pointer to the process selector.
     * @param debug File pointer for debugging output.
     */
    ColumnIndexSender(ProcessSelector* processSelector, FILE* debug);

    /**
     * @brief Determines the process responsible for a given item.
     * @param index The index of the item.
     * @param item The column index item.
     * @return The process ID responsible for the item.
     */
    int getProcessForItem(int index, itype item);

    /**
     * @brief Maps an item to be sent to a specific process.
     * @param item The column index item.
     * @param proc The target process.
     * @return The mapped column index for the given process.
     */
    itype mapItemForProcess(itype item, int proc);

    /**
     * @brief Converts a column index item to a string.
     * @param item The column index item.
     * @return A string representation of the item.
     */
    std::string toString(itype item);

    /**
     * @brief Debugs an array of items by printing them to the debug file.
     * @param title The title for the debug output.
     * @param arr The array of items.
     * @param len The length of the array.
     * @param isOnDevice Whether the array is on a device (GPU).
     */
    void debugItems(const char* title, itype* arr, size_t len, bool isOnDevice);
};

/**
 * @struct ColumnIndexComparator
 * @brief Comparator for sorting column indices.
 */
struct ColumnIndexComparator {
    /**
     * @brief Compares two column indices.
     * @param a First column index.
     * @param b Second column index.
     * @return True if `a` is less than `b`, false otherwise.
     */
    __device__ bool operator()(const itype& a, const itype& b) const
    {
        return a < b;
    }

    /**
     * @brief Identity function for a column index.
     * @param a The column index.
     * @return The same column index.
     */
    __device__ itype operator()(const itype& a) const
    {
        return a;
    }
};
