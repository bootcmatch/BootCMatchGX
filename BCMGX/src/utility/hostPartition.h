/**
 * @file
 */
#pragma once

#include "utility/utils.h"

/**
 * @brief Partitions an array based on a selection operator.
 * 
 * This function partitions the input array into two parts based on the result of the selection operator (`select_op`).
 * It places elements that satisfy the condition (i.e., where `select_op(d_in[i])` returns true) in the beginning of the 
 * output array and elements that do not satisfy the condition in the latter part. It returns the newly allocated array 
 * containing the partitioned elements. The length of the selected part (where the condition is true) is also computed.
 * 
 * @tparam T The type of elements in the array (e.g., `int`, `float`, etc.).
 * @tparam Operator The type of the operator used to partition the array.
 * 
 * @param[in] d_in The input array to partition.
 * @param[in] len The number of elements in the input array.
 * @param[in] select_op A functor or lambda expression used to select elements from the input array.
 *                     Elements for which `select_op(d_in[i])` returns true are placed at the beginning of the output array.
 * @param[out] selectedLen The number of elements that satisfy the selection condition (i.e., `select_op(d_in[i]) == true`).
 * 
 * @return A pointer to the output array containing the partitioned elements.
 *         The elements satisfying the condition are placed first, followed by the elements that do not.
 *         The length of the selected portion is stored in `selectedLen`.
 * 
 * @note The function allocates memory for the output array. The caller is responsible for freeing this memory after use.
 */
template <typename T, typename Operator>
T* hostPartition(T* d_in, size_t len, Operator select_op, size_t* selectedLen)
{
    if (!len) {
        *selectedLen = 0;
        return NULL;
    }

    T* d_out = MALLOC(T, len);

    size_t trueIndex = 0;
    size_t falseIndex = len - 1;

    for (size_t i = 0; i < len; i++) {
        if (select_op(d_in[i])) {
            d_out[trueIndex] = d_in[i];
            trueIndex++;
        } else {
            d_out[falseIndex] = d_in[i];
            falseIndex--;
        }
    }

    *selectedLen = trueIndex;

    return d_out;
}
