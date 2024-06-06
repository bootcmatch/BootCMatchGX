#pragma once

#include "utility/mpi.h"
#include "utility/setting.h"
#include "utility/utils.h"
#include <curand.h>
#include <iostream>
#include <stdarg.h>
#include <stdio.h>

template <typename T>
struct vector {
    int n = 0; // number of non-zero
    bool on_the_device = false;
    T* val = NULL; // array of nnz values
};

template <typename T>
struct vectordh {
    int n = 0;
    T* val = NULL;
    T* val_ = NULL;
};

template <typename T>
struct vectorCollection {
    unsigned int n = 0; // number of non-zero
    vector<T>** val = NULL;
};

namespace Vector {

template <typename T>
vector<T>* init(unsigned int n, bool allocate_mem, bool on_the_device);

template <typename T>
vectordh<T>* initdh(int n);
template <typename T>
void copydhToD(vectordh<T>* v);
template <typename T>
void copydhToH(vectordh<T>* v);
template <typename T>
void freedh(vectordh<T>* v);

template <typename T>
void fillWithValue(vector<T>* v, T value);
template <typename T>
void fillWithValueWithOff(vector<T>* v, T value, itype, itype);

template <typename T>
vector<T>* clone(vector<T>* a);

template <typename T>
vector<T>* localize_global_vector(vector<T>* global_v, int local_len, int shift);

template <typename T>
__global__ void _copy_kernel(itype n, T* dest, T* source);

template <typename T>
vector<T>* copyToDevice(vector<T>* v);

template <typename T>
void copyTo(vector<T>* dest, vector<T>* source);

template <typename T>
void copyToWithOff(vector<T>* dest, vector<T>* source, itype, itype);

template <typename T>
vector<T>* copyToHost(vector<T>* v_d);

template <typename T>
void free(vector<T>* v);

template <typename T>
void freeAsync(vector<T>* v, cudaStream_t stream);

template <typename T>
bool equals(vector<T>* a, vector<T>* b);

template <typename T>
bool is_zero(vector<T>* a);

template <typename T>
vector<T>* load(const char* file_name, bool on_the_device);

template <typename T>
void print(vector<T>* v, int n_ = -1, FILE* fp = stdout);

template <typename T>
T dot(cublasHandle_t handle, vector<T>* a, vector<T>* b, int stride_a = 1, int stride_b = 1);

template <typename T>
T norm(cublasHandle_t handle, vector<T>* a, int stride_a = 1);

template <typename T>
T norm_MPI(cublasHandle_t handle, vector<T>* a);

template <typename T>
void axpy(cublasHandle_t handle, vector<T>* x, vector<T>* y, T alpha, int inc = 1);
template <typename T>
void axpyWithOff(cublasHandle_t handle, vector<T>* x, vector<T>* y, T alpha, itype, itype);

template <typename T>
void scale(cublasHandle_t handle, vector<T>* x, T alpha, int inc = 1);

// vectorCollection of vector
namespace Collection {

    template <typename T>
    vectorCollection<T>* init(unsigned int n);

    template <typename T>
    void free(vectorCollection<T>* c);
}
}

template <typename T>
void dump(vector<T>* vec, const char* filename_fmt, ...)
{
    char filename[1024] = { 0 };
    va_list args;
    va_start(args, filename_fmt);
    vsnprintf(filename, sizeof(filename), filename_fmt, args);
    va_end(args);
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Could not open %s\n", filename);
        exit(1);
    }
    Vector::print(vec, -1, fp);
    fclose(fp);
}
