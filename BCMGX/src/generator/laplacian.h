/**
 * @file
 */
#pragma once
#include "utility/setting.h"

struct CSR;

/**
 * @brief Generates a local Laplacian matrix for a 3D grid with a given size.
 * 
 * This function generates a sparse matrix representing a 3D Laplacian operator.
 * It is typically used in finite difference methods to approximate the second 
 * derivative in 3D.
 * 
 * @param n The number of grid points in each dimension (assumes a cubic grid).
 * 
 * @return A pointer to the generated CSR (Compressed Sparse Row) matrix representing the 3D Laplacian.
 * 
 * @note The matrix generated here typically represents a discretized 3D Laplacian operator 
 *       using a 7-point stencil.
 */
CSR* generateLocalLaplacian3D(itype n);

/**
 * @brief Generates a local 7-point Laplacian matrix for a 3D grid with given dimensions.
 * 
 * This function generates a sparse matrix representing a 3D Laplacian operator using 
 * a 7-point stencil. It discretizes the Laplacian operator on a grid of size `nx` by `ny` by `nz`.
 * The grid is subdivided into `P`, `Q`, and `R` subdomains.
 * 
 * @param nx The number of grid points along the x-axis.
 * @param ny The number of grid points along the y-axis.
 * @param nz The number of grid points along the z-axis.
 * @param P The number of subdomains along the x-axis.
 * @param Q The number of subdomains along the y-axis.
 * @param R The number of subdomains along the z-axis.
 * 
 * @return A pointer to the generated CSR matrix representing the 7-point Laplacian.
 * 
 * @note This function is typically used for 3D simulations where the 7-point stencil 
 *       (considering 6 neighbors and the center point) is used for finite difference approximations.
 */
CSR* generateLocalLaplacian3D_7p(itype nx, itype ny, itype nz, itype P, itype Q, itype R);

/**
 * @brief Generates a local 27-point Laplacian matrix for a 3D grid with given dimensions.
 * 
 * This function generates a sparse matrix representing a 3D Laplacian operator using 
 * a 27-point stencil. It discretizes the Laplacian operator on a grid of size `nx` by `ny` by `nz`.
 * The grid is subdivided into `P`, `Q`, and `R` subdomains.
 * 
 * @param nx The number of grid points along the x-axis.
 * @param ny The number of grid points along the y-axis.
 * @param nz The number of grid points along the z-axis.
 * @param P The number of subdomains along the x-axis.
 * @param Q The number of subdomains along the y-axis.
 * @param R The number of subdomains along the z-axis.
 * 
 * @return A pointer to the generated CSR matrix representing the 27-point Laplacian.
 * 
 * @note This function is used for more accurate 3D simulations where the 27-point stencil 
 *       (considering 26 neighbors plus the center point) is used for finite difference approximations.
 */
CSR* generateLocalLaplacian3D_27p(itype nx, itype ny, itype nz, itype P, itype Q, itype R);

/**
 * @brief Reads `nx` by `ny`, `nz`, `P`, `Q`, and `R` parameters to be used in order to generate
 * a Laplacian matrix and return them as an integer array.
 * 
 * @param file_name The path to the file.
 * 
 * @return A pointer to the integer array containing the parameters.
 * 
 * @note The format of the Laplacian file should be consistent with the expected format to be correctly interpreted.
 */
int* read_laplacian_file(const char* file_name);
