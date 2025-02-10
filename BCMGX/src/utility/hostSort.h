/**
 * @file
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

/**
 * @brief Sorts an array on the host using the specified comparator.
 * 
 * This function sorts a given array of elements on the host using the Thrust library's
 * sorting algorithm. The sorting is done in place using the provided comparator.
 * If the size of the array is zero, the function does nothing.
 *
 * @tparam T The type of elements in the array (e.g., `int`, `float`, etc.).
 * @tparam K A type that is used by the comparator for the comparison (typically the same type as `T`).
 * @tparam Comparator A functor or lambda that defines the comparison logic between two elements.
 * 
 * @param[in,out] d_array The array to be sorted. This is a pointer to the array in memory.
 * @param[in] size The number of elements in the array.
 * @param[in] comparator A functor or lambda expression that compares two elements of type `T`.
 * 
 * @note The array is sorted in place, meaning that it will be reordered according to the comparator.
 * 
 * @see thrust::sort for more information on sorting with Thrust.
 */
template <typename T, typename K, typename Comparator>
void hostSort(T* d_array, size_t size, Comparator comparator)
{
    if (!size) {
        return;
    }

    thrust::sort(
        thrust::host,
        d_array,
        d_array + size,
        comparator);
}
