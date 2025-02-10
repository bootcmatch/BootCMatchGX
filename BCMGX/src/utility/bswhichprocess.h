/**
 * @file
 */
#pragma once

#include "utility/setting.h"

/**
 * @brief Determines which process owns a given element using binary search.
 *
 * This function searches for the process that should handle a given element `val` 
 * by performing a binary search on the partition boundaries stored in `arr`.
 *
 * @param arr Array containing the partition boundaries for each process.
 * @param len The total number of processes.
 * @param val The element whose owning process is to be determined.
 * @return The index of the process that owns element `val`.
 */
int bswhichprocess(gsstype* arr, int len, gsstype val);

/**
 * @brief Determines which process owns a given element using binary search.
 *
 * This function searches for the process that should handle a given element `val` 
 * by performing a binary search on the partition boundaries stored in `arr`.
 *
 * @param arr Array containing the partition boundaries for each process.
 * @param len The total number of processes.
 * @param val The element whose owning process is to be determined.
 * @return The index of the process that owns element `val`.
 */
int bswhichprocess(gstype* arr, int len, gsstype val);
