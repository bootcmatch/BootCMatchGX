#pragma once

#include <mpi.h>
#include "utility/arrays.h"
#include "utility/setting.h"

/**
 * Custom type to represent an item of a matrix.
 */
struct matrixItem_t {
    int row;
    int col;
    double val;
};

struct MatrixItemColumnIndexLessThanSelector {
    const int lower;

    __host__ __device__ __forceinline__ explicit MatrixItemColumnIndexLessThanSelector(const int lower)
        : lower(lower)
    {
    }

    __device__ __forceinline__ bool operator()(const matrixItem_t& x) const
    {
        return (x.col < lower);
    }
};

struct MatrixItemColumnIndexGreaterThanSelector {
    const int upper;

    __host__ __device__ __forceinline__ explicit MatrixItemColumnIndexGreaterThanSelector(const int upper)
        : upper(upper)
    {
    }

    __device__ __forceinline__ bool operator()(const matrixItem_t& x) const
    {
        return (x.col > upper);
    }
};

struct MatrixItemColumnIndexOutOfBoundsSelector {
    const int lower;
    const int upper;

    __host__ __device__ __forceinline__ explicit MatrixItemColumnIndexOutOfBoundsSelector(const int lower, const int upper)
        : lower(lower)
        , upper(upper)
    {
    }

    __device__ __forceinline__ bool operator()(const matrixItem_t& x) const
    {
        return (x.col < lower || x.col > upper);
    }
};

struct MatrixItemColumnIndexInBoundsSelector {
    const int lower;
    const int upper;

    __host__ __device__ __forceinline__ explicit MatrixItemColumnIndexInBoundsSelector(const int lower, const int upper)
        : lower(lower)
        , upper(upper)
    {
    }

    __device__ __forceinline__ bool operator()(const matrixItem_t& x) const
    {
        return (lower <= x.col && x.col <= upper);
    }
};

struct MatrixItemComparator {
    gstype _ncols;

    __host__ __device__ __forceinline__
    MatrixItemComparator(gstype ncols)
        : _ncols(ncols)
    {
    }

    __host__ __device__ __forceinline__ bool operator()(const matrixItem_t& a, const matrixItem_t& b) const
    {
        return a.row < b.row || a.row == b.row && a.col < b.col;
    }

    __host__ __device__ __forceinline__
        gstype
        operator()(const matrixItem_t& a) const
    {
        return a.row * _ncols + a.col;
    }
};

struct MatrixItemTransposedComparator {
    gstype _nrows;

    __host__ __device__ __forceinline__
    MatrixItemTransposedComparator(gstype nrows)
        : _nrows(nrows)
    {
    }

    __host__ __device__ __forceinline__ bool operator()(const matrixItem_t& a, const matrixItem_t& b) const
    {
        return a.col < b.col || a.col == b.col && a.row < b.row;
    }

    __host__ __device__ __forceinline__
        gstype
        operator()(const matrixItem_t& a) const
    {
        return a.col * _nrows + a.row;
    }
};

struct MatrixItemColumnMapper {
    __host__ __device__ __forceinline__
        itype
        operator()(const matrixItem_t& a) const
    {
        return a.col;
    }
};

extern MPI_Datatype MPI_MATRIX_ITEM_T;

void registerMatrixItemMpiDatatypes();

void debugMatrixItems(const char* title, matrixItem_t* arr, size_t len, bool isOnDevice, FILE* f);

/**
 * Scans a vector of matrix items and fills the CSR.
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
