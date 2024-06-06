#pragma once

#include <iostream>
#include <curand.h>
#include "utility/setting.h"
#include "utility/utils.h"
#include "utility/myMPI.h"

template <typename T>
struct vector{
  int n; // number of non-zero
  bool on_the_device;
  T *val; // array of nnz values
};

template <typename T>
struct vectorCollection{
  unsigned int n; // number of non-zero
  vector<T> **val;
};

namespace Vector{

  template <typename T>
  vector<T>* init(unsigned int n, bool allocate_mem, bool on_the_device);

  template <typename T>
  void fillWithValue(vector<T> *v, T value);
  template <typename T>
  void fillWithValueWithOff(vector<T> *v, T value, itype, itype);

  template <typename T>
  vector<T>* clone(vector<T> *a);
  
  template <typename T>
  vector<T>* localize_global_vector(vector<T> *global_v, int local_len, int shift);

  template <typename T>
  __global__
  void _copy_kernel(itype n, T* dest, T* source);
  
  template <typename T>
  vector<T>* copyToDevice(vector<T> *v);

  template <typename T>
  void copyTo(vector<T> *dest, vector<T> *source);

  template <typename T>
  void copyToWithOff(vector<T> *dest, vector<T> *source, itype, itype);

  template <typename T>
  vector<T>* copyToHost(vector<T> *v_d);

  template <typename T>
  void free(vector<T> *v);
  
  template <typename T>
  void freeAsync(vector<T> *v, cudaStream_t stream);
  
  template <typename T>
  bool equals(vector<T> *a, vector<T> *b);
  
  template <typename T>
  bool is_zero(vector<T> *a);

  template <typename T>
  void print(vector<T> *v, int n_=-1, FILE* fp = stdout);

  template <typename T>
  T dot(cublasHandle_t handle, vector<T> *a, vector<T> *b, int stride_a=1, int stride_b=1);

  template <typename T>
  T norm(cublasHandle_t handle, vector<T> *a, int stride_a=1);

  template <typename T>
  T norm_MPI(cublasHandle_t handle, vector<T> *a);

  template <typename T>
  void axpy(cublasHandle_t handle, vector<T> *x, vector<T> *y, T alpha, int inc=1);
  template <typename T>
  void axpyWithOff(cublasHandle_t handle, vector<T> *x, vector<T> *y, T alpha, itype, itype);

  template <typename T>
  void scale(cublasHandle_t handle, vector<T> *x, T alpha, int inc=1);

  // vectorCollection of vector
  namespace Collection{

    template<typename T>
    vectorCollection<T>* init(unsigned int n);

    template<typename T>
    void free(vectorCollection<T> *c);
  }
}
