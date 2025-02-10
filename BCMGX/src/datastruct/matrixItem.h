/**
 * @file
 * 
 * This file contains definitions for structs and functions related to matrix item manipulation,
 * particularly for operations in sparse matrix formats (like CSR).
 */
#pragma once

#include "utility/arrays.h"
#include "utility/setting.h"
#include <mpi.h>

/**
 * @brief Custom type to represent an item of a matrix.
 */
struct matrixItem_t {
    long row;   ///< Row index of the matrix item.
    long col;   ///< Column index of the matrix item.
    double val; ///< Value of the matrix item.
};

/**
 * @brief Selector that filters matrix items with a column index less than a specified value.
 */
struct MatrixItemColumnIndexLessThanSelector {
    const long lower; ///< The threshold value for the column index.

    /**
     * @brief Constructor to initialize the lower threshold.
     * 
     * @param lower The column index threshold.
     */
    __host__ __device__ __forceinline__ explicit MatrixItemColumnIndexLessThanSelector(const long lower)
        : lower(lower)
    {
    }

    /**
     * @brief Comparison operator to check if the matrix item's column is less than the lower threshold.
     * 
     * @param x The matrix item to compare.
     * @return True if the column index of `x` is less than the lower threshold.
     */
    __device__ __forceinline__ bool operator()(const matrixItem_t& x) const
    {
        return (x.col < lower);
    }
};

/**
 * @brief Selector that filters matrix items with a column index greater than a specified value.
 */
struct MatrixItemColumnIndexGreaterThanSelector {
    const long upper; ///< The threshold value for the column index.

    /**
     * @brief Constructor to initialize the upper threshold.
     * 
     * @param upper The column index threshold.
     */
    __host__ __device__ __forceinline__ explicit MatrixItemColumnIndexGreaterThanSelector(const long upper)
        : upper(upper)
    {
    }

    /**
     * @brief Comparison operator to check if the matrix item's column is greater than the upper threshold.
     * 
     * @param x The matrix item to compare.
     * @return True if the column index of `x` is greater than the upper threshold.
     */
    __device__ __forceinline__ bool operator()(const matrixItem_t& x) const
    {
        return (x.col > upper);
    }
};

/**
 * @brief Selector that filters matrix items with a column index outside a specified range.
 */
struct MatrixItemColumnIndexOutOfBoundsSelector {
    const long lower; ///< The lower bound for the column index.
    const long upper; ///< The upper bound for the column index.

    /**
     * @brief Constructor to initialize the lower and upper bounds.
     * 
     * @param lower The lower bound for the column index.
     * @param upper The upper bound for the column index.
     */
    __host__ __device__ __forceinline__ explicit MatrixItemColumnIndexOutOfBoundsSelector(const long lower, const long upper)
        : lower(lower)
        , upper(upper)
    {
    }

    /**
     * @brief Comparison operator to check if the matrix item's column is out of bounds.
     * 
     * @param x The matrix item to compare.
     * @return True if the column index of `x` is outside the bounds (lower, upper).
     */
    __host__ __device__ __forceinline__ bool operator()(const matrixItem_t& x) const
    {
        return (x.col < lower || x.col > upper);
    }
};

/**
 * @brief Selector that filters matrix items with a column index within a specified range.
 */
struct MatrixItemColumnIndexInBoundsSelector {
    const long lower; ///< The lower bound for the column index.
    const long upper; ///< The upper bound for the column index.

    /**
     * @brief Constructor to initialize the lower and upper bounds.
     * 
     * @param lower The lower bound for the column index.
     * @param upper The upper bound for the column index.
     */
    __host__ __device__ __forceinline__ explicit MatrixItemColumnIndexInBoundsSelector(const long lower, const long upper)
        : lower(lower)
        , upper(upper)
    {
    }

    /**
     * @brief Comparison operator to check if the matrix item's column is within the bounds.
     * 
     * @param x The matrix item to compare.
     * @return True if the column index of `x` is within the bounds (lower, upper).
     */
    __device__ __forceinline__ bool operator()(const matrixItem_t& x) const
    {
        return (lower <= x.col && x.col <= upper);
    }
};

/**
 * @brief Comparator for matrix items based on their row and column indices.
 */
struct MatrixItemComparator {
    gstype _ncols; ///< The number of columns in the matrix.

    /**
     * @brief Constructor to initialize the number of columns.
     * 
     * @param ncols The number of columns in the matrix.
     */
    __host__ __device__ __forceinline__
    MatrixItemComparator(gstype ncols)
        : _ncols(ncols)
    {
    }

    /**
     * @brief Comparison operator for matrix items based on their row and column indices.
     * 
     * @param a The first matrix item.
     * @param b The second matrix item.
     * @return True if `a` is less than `b` based on row and column indices.
     */
    __host__ __device__ __forceinline__ bool operator()(const matrixItem_t& a, const matrixItem_t& b) const
    {
        return a.row < b.row || a.row == b.row && a.col < b.col;
    }

    /**
     * @brief Function to compute a unique linear index for a matrix item based on row and column indices.
     * 
     * @param a The matrix item.
     * @return The linear index corresponding to the matrix item's row and column.
     */
    __host__ __device__ __forceinline__
        gstype
        operator()(const matrixItem_t& a) const
    {
        return a.row * _ncols + a.col;
    }
};

/**
 * @brief Comparator for matrix items based on their transposed row and column indices.
 */
struct MatrixItemTransposedComparator {
    gstype _nrows; ///< The number of rows in the matrix.

    /**
     * @brief Constructor to initialize the number of rows.
     * 
     * @param nrows The number of rows in the matrix.
     */
    __host__ __device__ __forceinline__
    MatrixItemTransposedComparator(gstype nrows)
        : _nrows(nrows)
    {
    }

    /**
     * @brief Comparison operator for matrix items based on their transposed row and column indices.
     * 
     * @param a The first matrix item.
     * @param b The second matrix item.
     * @return True if `a` is less than `b` based on transposed row and column indices.
     */
    __host__ __device__ __forceinline__ bool operator()(const matrixItem_t& a, const matrixItem_t& b) const
    {
        return a.col < b.col || a.col == b.col && a.row < b.row;
    }

    /**
     * @brief Function to compute a unique linear index for a transposed matrix item.
     * 
     * @param a The matrix item.
     * @return The linear index corresponding to the transposed matrix item's row and column.
     */
    __host__ __device__ __forceinline__
        gstype
        operator()(const matrixItem_t& a) const
    {
        return a.col * _nrows + a.row;
    }
};

/**
 * @brief Mapper to extract the column index from a matrix item.
 */
struct MatrixItemColumnMapper {
    /**
     * @brief Extracts the column index from a matrix item.
     * 
     * @param a The matrix item.
     * @return The column index of the matrix item.
     */
    __host__ __device__ __forceinline__
        itype
        operator()(const matrixItem_t& a) const
    {
        return a.col;
    }
};

/**
 * @brief MPI datatype for `matrixItem_t`.
 */
extern MPI_Datatype MPI_MATRIX_ITEM_T;

/**
 * @brief Registers the MPI datatype for `matrixItem_t`.
 * 
 * This function registers the custom MPI datatype for `matrixItem_t` to be used in MPI communications.
 */
void registerMatrixItemMpiDatatypes();

/**
 * @brief Debug function to print matrix items.
 * 
 * This function prints the contents of a matrix item array to a file for debugging purposes.
 * 
 * @param title The title to display.
 * @param arr The array of matrix items.
 * @param len The length of the array.
 * @param isOnDevice Indicates if the data is on the device (GPU).
 * @param f The file to write the output to.
 */
void debugMatrixItems(const char* title, matrixItem_t* arr, size_t len, bool isOnDevice, FILE* f);

/**
 * @brief Fills the CSR representation from a vector of matrix items.
 * 
 * This function converts a list of matrix items to a CSR (Compressed Sparse Row) format.
 * 
 * @param items The array of matrix items.
 * @param nnz The number of non-zero elements.
 * @param n The number of rows in the matrix.
 * @param rowShift The shift for the row indices.
 * @param rowRet The CSR row pointers (output).
 * @param colRet The CSR column indices (output).
 * @param valRet The CSR values (output).
 * @param transposed Whether the matrix is transposed.
 * @param allocateMemory Whether to allocate memory for the CSR arrays.
 */
void fillCsrFromMatrixItems(
    matrixItem_t* items,
    size_t nnz,
    size_t n,
    int rowShift,
    itype** rowRet,
    itype** colRet,
    vtype** valRet,
    bool transposed,
    bool allocateMemory);

/**
 * @brief Fills the CSR representation without GPU support.
 * 
 * This function converts a list of matrix items to a CSR format without GPU acceleration.
 * 
 * @param items The array of matrix items.
 * @param nnz The number of non-zero elements.
 * @param n The number of rows in the matrix.
 * @param rowShift The shift for the row indices.
 * @param rowRet The CSR row pointers (output).
 * @param colRet The CSR column indices (output).
 * @param valRet The CSR values (output).
 * @param transposed Whether the matrix is transposed.
 * @param allocateMemory Whether to allocate memory for the CSR arrays.
 */
void fillCsrFromMatrixItems_nogpu(
    matrixItem_t* items,
    size_t nnz,
    size_t n,
    int rowShift,
    itype** rowRet,
    itype** colRet,
    vtype** valRet,
    bool transposed,
    bool allocateMemory);
