#include "vector.h"
#include "utility/cudamacro.h"
namespace Vector{

  template <typename T>
  vector<T>* init(int n, bool allocate_mem, bool on_the_device){
    assert(n>0);
    vector<T> *v = NULL;
    // on the host
    v = (vector<T>*) malloc( sizeof(vector<T>) );
    CHECK_HOST(v);

    v->n = n;
    v->on_the_device = on_the_device;

    if(allocate_mem){
      if(on_the_device){
        // on the device
        cudaError_t err;
        err = cudaMalloc( (void**) &v->val, n * sizeof(T) );
        CHECK_DEVICE(err);
      }else{
        // on the host
        v->val = (T*) malloc( n * sizeof(T) );
        CHECK_HOST(v->val);
      }
    }
    return v;
  }

  template<typename T>
  __global__ void _fillKernel(int n, T *v, const T val){
      int tidx = threadIdx.x + blockDim.x * blockIdx.x;
      int stride = blockDim.x * gridDim.x;

      for(; tidx < n; tidx += stride)
          v[tidx] = val;
  }

  template <typename T>
  void fillWithValue(vector<T> *v, T value){
    if(v->on_the_device){
      if(value == 0){
        cudaError_t err = cudaMemset(v->val, value, v->n * sizeof(T));
        CHECK_DEVICE(err);
      }else{
        dim3 block (BLOCKSIZE);
        dim3 grid ( ceil( (double) v->n / (double) block.x));
        _fillKernel<<<grid, block>>>(v->n, v->val, value);
      }
    }else{
      std::fill_n(v->val, v->n, value);
    }
  }

  template <typename T>
  void fillWithValueWithOff(vector<T> *v, T value, itype n, itype off){
    if(v->on_the_device){
      if(value == 0){
        cudaError_t err = cudaMemset(v->val+off, value, n * sizeof(T));
        CHECK_DEVICE(err);
      }else{
        dim3 block (BLOCKSIZE);
        dim3 grid ( ceil( (double) n / (double) block.x));
        _fillKernel<<<grid, block>>>(n, v->val+off, value);
      }
    }else{
      std::fill_n(v->val+off, n, value);
    }
  }

  template <typename T>
  vector<T>* clone(vector<T> *a){

    assert( a->on_the_device );

    vector<T> *b = Vector::init<T>(a->n, true, true);

    cudaError_t err;
    err = cudaMemcpy(b->val, a->val, b->n * sizeof(T), cudaMemcpyDeviceToDevice);
    CHECK_DEVICE(err);

    return b;
  }

  template <typename T>
  vector<T>* copyToDevice(vector<T> *v){

    assert( !v->on_the_device );

    int n = v->n;

    // alocate vector on the device memory
    vector<T> *v_d = init<T>(n, true, true);

    cudaError_t err = cudaMemcpy(v_d->val, v->val, n * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);

    return v_d;
  }

  template <typename T>
  __global__
  void _copy_kernel(itype n, T* dest, T* source){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= n)
      return;

      dest[i] = source[i];
  }



  template <typename T>
  void copyTo(vector<T> *dest, vector<T> *source){

    int n = dest->n;

    gridblock gb = gb1d(n, BLOCKSIZE);
     _copy_kernel<<<gb.g, gb.b>>>(n, dest->val, source->val);

  }

  template <typename T>
  void copyToWithOff(vector<T> *dest, vector<T> *source, itype n, itype off){

    gridblock gb = gb1d(n, BLOCKSIZE);
     _copy_kernel<<<gb.g, gb.b>>>(n, dest->val+off, source->val+off);

  }

  template <typename T>
  vector<T>* copyToHost(vector<T> *v_d){

    assert( v_d->on_the_device );

    int n = v_d->n;

    // alocate vector on the host memory
    vector<T> *v = init<T>(n, true, false);

    cudaError_t err;

    err = cudaMemcpy(v->val, v_d->val, n * sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    return v;
  }

  template <typename T>
  void free(vector<T> *v){
    if(v->on_the_device){
      cudaError_t err;
      err = cudaFree(v->val);
      CHECK_DEVICE(err);
    }else{
      assert(v->val>0);
      std::free(v->val);
    }
    std::free(v);
  }

  template <typename T>
  void print(vector<T> *v, int n_){
    vector<T> *v_;

    int n;

    if(n_ == -1)
      n = v->n;
    else
      n = n_;

    if(v->on_the_device){
      v_ = Vector::copyToHost<T>(v);
    }else{
      v_ = v;
    }

    int i;
  	for(i=0; i<n; i++){
  		std::cout << v_->val[i];
      std::cout << " ";
  	}
  	std::cout << "\n\n";

    if(v->on_the_device){
      Vector::free<T>(v_);
    }

  }

   template <typename T>
  __global__
  void _elementwise_div(itype n, T *a, T *b, T *c){
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= n)
      return;

    c[i] = a[i] / b[i];
  }

  template <typename T>
  vector<T>* elementwise_div(vector<T> *a, vector<T> *b, vector<T> *c){
    assert(a->n == b->n);

    if(c == NULL)
      c = Vector::init<T>(a->n, true, true);

    gridblock gb = gb1d(a->n, BLOCKSIZE);
    _elementwise_div<<<gb.g, gb.b>>>(a->n, a->val, b->val, c->val);

    return c;
  }


template <typename T>
  T dot(cublasHandle_t handle, vector<T> *a, vector<T> *b, int stride_a, int stride_b){

    assert(a->on_the_device == b->on_the_device);

    T result;
    cublasStatus_t cublas_state;
    cublas_state = cublasDdot(handle, a->n, a->val, stride_a, b->val, stride_b, &result);
    CHECK_CUBLAS(cublas_state);
    return result;
}

  template <typename T>
  void axpy(cublasHandle_t handle, vector<T> *x, vector<T> *y, T alpha, int inc){

    assert(x->on_the_device == y->on_the_device);
    assert(x->n == y->n);

    cublasStatus_t cublas_state;
    cublas_state = cublasDaxpy(handle, x->n, &alpha, x->val, inc, y->val, inc);
    CHECK_CUBLAS(cublas_state);
  }

  template <typename T>
  void axpyWithOff(cublasHandle_t handle, vector<T> *x, vector<T> *y, T alpha, itype n, itype off ){

    assert(x->on_the_device == y->on_the_device);
    assert(x->n == y->n);
    int inc=1;

    cublasStatus_t cublas_state;
    cublas_state = cublasDaxpy(handle, n, &alpha, x->val+off, inc, y->val+off, inc);
    CHECK_CUBLAS(cublas_state);
  }

  template <typename T>
  T norm_MPI(cublasHandle_t handle, vector<T> *a){
    _MPI_ENV;
    assert(a->on_the_device);

    double result_local = dot(handle, a, a);
    double result = 0.;
    CHECK_MPI(
      MPI_Allreduce(
        &result_local,
        &result,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
      )
    );

    result = sqrt( result );

    return result;
  }

  template <typename T>
  T norm(cublasHandle_t handle, vector<T> *a, int stride_a){
    assert(a->on_the_device);

    T result;
    cublasStatus_t cublas_state;
    cublas_state = cublasDnrm2(handle, a->n, a->val, stride_a, &result);
    CHECK_CUBLAS(cublas_state);
    return result;
  }

  template <typename T>
  void scale(cublasHandle_t handle, vector<T> *x, T alpha, int inc){

    assert(x->on_the_device);

    cublasStatus_t cublas_state;
    cublas_state = cublasDscal(handle, x->n, &alpha, x->val, inc);
    CHECK_CUBLAS(cublas_state);

  }

  // only for very hard DEBUG
  template<typename T>
  int checkEPSILON(vector<T> *A_){
    vector<T> *A = NULL;

    if(A_->on_the_device)
      A = Vector::copyToHost(A_);
    else
      A = A_;

    int count = 0;
    for(int i=0; i<A->n; i++){

      if(A->val[i] == DBL_EPSILON)
        count++;
    }

    return count;
  }

  // vectorCollection of vector
  namespace Collection{

    template<typename T>
    vectorCollection<T>* init(int n){
      vectorCollection<T> *c = NULL;
      c = (vectorCollection<T>*) malloc(sizeof(vectorCollection<T>));
      CHECK_HOST(c);

      c->n = n;
      c->val = (vector<T>**) malloc( n * sizeof(vector<T>*));
      CHECK_HOST(c->val);
      for(int i=0; i<c->n; i++)
        c->val[i] = NULL;

      return c;
    }

    template<typename T>
    void free(vectorCollection<T> *c){

      for(int i=0; i<c->n; i++){
        if(c->val[i] != NULL)
          Vector::free(c->val[i]);
      }

      std::free(c->val);
      std::free(c);
    }
  }
}

namespace Vector{
  template vector<itype>* init<itype>(int, bool, bool);
  template vector<vtype>* init<vtype>(int, bool, bool);

  template void fillWithValue<itype>(vector<itype>*, itype);
  template void fillWithValue<vtype>(vector<vtype>*, vtype);
  template void fillWithValueWithOff<vtype>(vector<vtype>*, vtype, itype, itype);

  template vector<itype>* clone<itype>(vector<itype>*);
  template vector<vtype>* clone<vtype>(vector<vtype>*);

  template vector<itype>* copyToDevice<itype>(vector<itype>*);
  template vector<vtype>* copyToDevice<vtype>(vector<vtype>*);

  template void copyTo<itype>(vector<itype>*, vector<itype>*);
  template void copyTo<vtype>(vector<vtype>*, vector<vtype>*);
  template void copyToWithOff<vtype>(vector<vtype>*, vector<vtype>*, itype, itype);

  template vector<itype>* copyToHost<itype>(vector<itype>*);
  template vector<vtype>* copyToHost<vtype>(vector<vtype>*);

  template void free<itype>(vector<itype>*);
  template void free<vtype>(vector<vtype>*);

  template void print<itype>(vector<itype>*, int);
  template void print<vtype>(vector<vtype>*, int);

  template vtype dot<vtype>(cublasHandle_t, vector<vtype>*, vector<vtype>*, int, int);

  template vtype norm<vtype>(cublasHandle_t, vector<vtype>*, int);

  template vtype norm_MPI<vtype>(cublasHandle_t, vector<vtype>*);

  template void axpy<vtype>(cublasHandle_t, vector<vtype>*, vector<vtype>*, vtype, int);
  template void axpyWithOff<vtype>(cublasHandle_t, vector<vtype>*, vector<vtype>*, vtype, itype, itype );

  template void scale<vtype>(cublasHandle_t, vector<vtype>*, vtype, int);

  namespace Collection{

    template vectorCollection<itype>* init<itype>(int);
    template vectorCollection<vtype>* init<vtype>(int);

    template void free<itype>(vectorCollection<itype>*);
    template void free<vtype>(vectorCollection<vtype>*);
  }
}
