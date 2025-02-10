/**
 * @file
 */
#pragma once

#include "datastruct/CSR.h"

/**
 * @enum generator_t
 * @brief Enum for the Laplacian matrix generators.
 * 
 * This enum specifies the types of Laplacian generators that can be used.
 */
enum generator_t {
    LAP_7P, /**< 7-point Laplacian generator */
    LAP_27P, /**< 27-point Laplacian generator */
    INVALIG_GEN /**< Invalid generator */
};

/**
 * @brief Returns the generator type based on the provided string.
 * 
 * This function maps the input string to the corresponding generator type defined
 * in the `generator_t` enum. If the string does not match any of the predefined
 * types, the function returns `INVALIG_GEN`.
 * 
 * @param[in] str The string representing the Laplacian generator type.
 * 
 * @return A generator_t value corresponding to the input string.
 * @retval LAP_7P if the string is "7p".
 * @retval LAP_27P if the string is "27p".
 * @retval INVALIG_GEN if the string is neither "7p" nor "27p".
 */
generator_t get_generator(const char* str);

/**
 * @brief Reads a local matrix from a file and distributes it across MPI processes.
 * 
 * This function reads a matrix from the specified `.mtx` file, splits it across the available
 * MPI processes, and returns a local submatrix for the current process. The matrix is adjusted
 * by shifting its columns, and the result is returned.
 * 
 * @param[in] mtx_file Path to the input matrix file in `.mtx` format.
 * 
 * @return A pointer to the local CSR matrix for the current process.
 * 
 * @note This function handles MPI initialization and distribution of the matrix across
 *       all processes. Only the master process reads the matrix from the file.
 */
CSR* read_local_matrix_from_mtx_host(const char* mtx_file);

/**
 * @brief Reads a local matrix from a file and distributes it across MPI processes.
 * 
 * This function reads a matrix from the specified `.mtx` file, splits it across the available
 * MPI processes, and returns a local submatrix for the current process. The matrix is adjusted
 * by shifting its columns, and the result is returned.
 * 
 * @param[in] mtx_file Path to the input matrix file in `.mtx` format.
 * 
 * @return A pointer to the local CSR matrix for the current process.
 * 
 * @note This function handles MPI initialization and distribution of the matrix across
 *       all processes. Only the master process reads the matrix from the file.
 */
CSR* read_local_matrix_from_mtx(const char* mtx_file);

/**
 * @brief Generates a local Laplacian matrix for 3D grids on the host.
 * 
 * This function generates a local 3D Laplacian matrix on the host using a predefined
 * 7-point stencil generator and returns it.
 * 
 * @param[in] n The size of the Laplacian matrix.
 * 
 * @return A pointer to the generated local Laplacian matrix.
 */
CSR* generate_lap_local_matrix_host(itype n);

/**
 * @brief Generates a local Laplacian matrix for 3D grids on the device.
 * 
 * This function generates a local 3D Laplacian matrix on the host and copies it to the device
 * before returning the resulting matrix on the device.
 * 
 * @param[in] n The size of the Laplacian matrix.
 * 
 * @return A pointer to the generated local Laplacian matrix on the device.
 */
CSR* generate_lap_local_matrix(itype n);

/**
 * @brief Generates a local Laplacian matrix for a 3D grid using a specified generator on the host.
 * 
 * This function reads parameters from a specified file, generates a Laplacian matrix for a 3D
 * grid using either the 7-point or 27-point stencil, and returns the resulting matrix. 
 * It verifies that the number of MPI processes matches the expected dimensions based on the
 * parameters in the file.
 * 
 * @param[in] generator The type of Laplacian generator (7-point or 27-point).
 * @param[in] lap_3d_file Path to the file containing the parameters for generating the Laplacian matrix.
 * 
 * @return A pointer to the generated Laplacian matrix on the host.
 * 
 * @throws std::runtime_error If the number of MPI processes does not match the expected dimensions
 *         specified in the file.
 */
CSR* generate_lap3d_local_matrix_host(generator_t generator, const char* lap_3d_file);

/**
 * @brief Generates a local Laplacian matrix for a 3D grid using a specified generator on the device.
 * 
 * This function generates a local Laplacian matrix for a 3D grid using a specified generator
 * on the host, copies the resulting matrix to the device, and returns the device-side matrix.
 * 
 * @param[in] generator The type of Laplacian generator (7-point or 27-point).
 * @param[in] lap_3d_file Path to the file containing the parameters for generating the Laplacian matrix.
 * 
 * @return A pointer to the generated Laplacian matrix on the device.
 */
CSR* generate_lap3d_local_matrix(generator_t generator, const char* lap_3d_file);
