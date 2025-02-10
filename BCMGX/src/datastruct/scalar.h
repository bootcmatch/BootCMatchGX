/**
 * @file
 * @brief This file defines a templated structure `scalar` and utility functions to manipulate scalar values
 *        that reside either on the host or device memory. It is designed for use in GPU programming with device
 *        memory operations.
 * 
 * The `scalar` structure and the functions within the `Scalar` namespace handle operations such as initialization,
 * memory transfer between device and host, and memory cleanup for scalar types.
 */
#pragma once

#include "utility/setting.h"
#include "utility/utils.h"
#include <iostream>

/**
 * @struct scalar
 * @brief A templated structure to represent a scalar value that can reside either on the host or the device.
 * 
 * This structure stores a boolean flag indicating if the scalar resides on the device or not, as well as a pointer
 * to the scalar value.
 * 
 * @tparam T The data type of the scalar (e.g., `float`, `int`, `double`).
 */
template <typename T>
struct scalar {
    bool on_the_device; /**< Flag indicating whether the scalar is on the device memory */
    T* val; /**< Pointer to the scalar value */
};

/**
 * @namespace Scalar
 * @brief A namespace that contains utility functions for handling scalar values on host and device memory.
 * 
 * Functions in this namespace provide operations for initializing, copying, freeing, and retrieving values of scalars
 * that reside on either the host or device.
 */
namespace Scalar {

/**
 * @brief Initializes a scalar and places it either on the host or device.
 * 
 * This function creates a scalar value and initializes it on either the device or host, based on the `on_the_device`
 * flag. The scalar's value is also set during initialization.
 * 
 * @tparam T The data type of the scalar (e.g., `float`, `int`, `double`).
 * @param val The value to initialize the scalar with.
 * @param on_the_device Flag indicating if the scalar should be allocated on the device (`true`) or host (`false`).
 * @return A pointer to the initialized scalar.
 */
template <typename T>
scalar<T>* init(T val, bool on_the_device);

/**
 * @brief Copies a scalar value from host to device memory.
 * 
 * This function copies a scalar value from host memory to device memory.
 * 
 * @tparam T The data type of the scalar.
 * @param v A pointer to the scalar to copy to the device.
 * @return A pointer to the copied scalar on the device.
 */
template <typename T>
scalar<T>* copyToDevice(scalar<T>* v);

/**
 * @brief Copies a scalar value from device to host memory.
 * 
 * This function copies a scalar value from device memory to host memory.
 * 
 * @tparam T The data type of the scalar.
 * @param v_d A pointer to the scalar stored on the device.
 * @return A pointer to the scalar copied to the host.
 */
template <typename T>
scalar<T>* copyToHost(scalar<T>* v_d);

/**
 * @brief Frees the memory associated with a scalar value.
 * 
 * This function frees the memory allocated for a scalar value, either on the host or the device, depending on
 * where it resides.
 * 
 * @tparam T The data type of the scalar.
 * @param v A pointer to the scalar whose memory is to be freed.
 */
template <typename T>
void free(scalar<T>* v);

/**
 * @brief Prints the value of a scalar.
 * 
 * This function prints the value of a scalar, either from the host or device memory, depending on where the scalar
 * resides.
 * 
 * @tparam T The data type of the scalar.
 * @param v A pointer to the scalar whose value is to be printed.
 */
template <typename T>
void print(scalar<T>* v);

/**
 * @brief Retrieves the value of a scalar from device memory with minimal overhead.
 * 
 * This function retrieves the scalar value from device memory without the overhead of copying the entire object.
 * 
 * @tparam T The data type of the scalar.
 * @param v_d A pointer to the scalar stored on the device.
 * @return A pointer to the scalar value directly from device memory.
 */
template <typename T>
T* getvalueFromDevice(scalar<T>* v_d);

}
