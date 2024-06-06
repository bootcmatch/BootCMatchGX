#include "utility/cudamacro.h"
#include "utility/function_cnt.h"
#include "utility/utils.h"
#include "vector.h"

#include <string.h>
#include <type_traits>
#include <unistd.h>

#define BUFSIZE 1024

namespace Vector {

template <typename T>
vector<T>* init(unsigned int n, bool allocate_mem, bool on_the_device)
{

    if (n == 0) {
        fprintf(stderr, "error in Vector::init: n as int = %d, n as unsigned int = %u\n", n, n);
    }
    assert(n > 0);
    vector<T>* v = NULL;
    // on the host
    v = (vector<T>*)Malloc(sizeof(vector<T>));
    CHECK_HOST(v);

    v->n = n;
    v->on_the_device = on_the_device;

    if (allocate_mem) {
        if (on_the_device) {
            // on the device
            cudaError_t err;
            err = cudaMalloc((void**)&v->val, n * sizeof(T));
            CHECK_DEVICE(err);
        } else {
            // on the host
            v->val = (T*)Malloc(n * sizeof(T));
            CHECK_HOST(v->val);
        }
    }
    return v;
}

template <typename T>
vectordh<T>* initdh(int n)
{

    assert(n > 0);
    vectordh<T>* v = NULL;

    v = (vectordh<T>*)Malloc(sizeof(vectordh<T>));
    CHECK_HOST(v);

    v->n = n;

    cudaError_t err;
    err = cudaMalloc((void**)&v->val, n * sizeof(T));
    CHECK_DEVICE(err);

    v->val_ = (T*)Malloc(n * sizeof(T));
    CHECK_HOST(v->val_);

    return v;
}

template <typename T>
void copydhToD(vectordh<T>* v)
{
    cudaError_t err;
    err = cudaMemcpy(v->val, v->val_, v->n * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);
}

template <typename T>
void copydhToH(vectordh<T>* v)
{
    cudaError_t err;
    err = cudaMemcpy(v->val_, v->val, v->n * sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);
}

template <typename T>
void freedh(vectordh<T>* v)
{

    cudaError_t err;
    err = cudaFree(v->val);
    CHECK_DEVICE(err);

    assert(v->val_);
    std::free(v->val_);

    std::free(v);
}

template <typename T>
__global__ void _fillKernel(int n, T* v, const T val)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; tidx < n; tidx += stride) {
        v[tidx] = val;
    }
}

template <typename T>
void fillWithValue(vector<T>* v, T value)
{
    if (v->on_the_device) {
        if (value == 0) {
            cudaError_t err = cudaMemset(v->val, value, v->n * sizeof(T));
            CHECK_DEVICE(err);
        } else {
            dim3 block(BLOCKSIZE);
            dim3 grid(ceil((double)v->n / (double)block.x));
            _fillKernel<<<grid, block>>>(v->n, v->val, value);
        }
    } else {
        std::fill_n(v->val, v->n, value);
    }
}

template <typename T>
void fillWithValueWithOff(vector<T>* v, T value, itype n, itype off)
{
    if (v->on_the_device) {
        if (value == 0) {
            cudaError_t err = cudaMemset(v->val + off, value, n * sizeof(T));
            CHECK_DEVICE(err);
        } else {
            dim3 block(BLOCKSIZE);
            dim3 grid(ceil((double)n / (double)block.x));
            _fillKernel<<<grid, block>>>(n, v->val + off, value);
        }
    } else {
        std::fill_n(v->val + off, n, value);
    }
}

template <typename T>
vector<T>* clone(vector<T>* a)
{

    vector<T>* b = NULL;
    if (a->on_the_device) {
        Vectorinit_CNT
            b
            = Vector::init<T>(a->n, true, true);

        cudaError_t err;
        err = cudaMemcpy(b->val, a->val, b->n * sizeof(T), cudaMemcpyDeviceToDevice);
        CHECK_DEVICE(err);
    } else {
        b = Vector::init<T>(a->n, true, false);
        memcpy(b->val, a->val, sizeof(T) * a->n);
    }

    return b;
}

template <typename T>
vector<T>* localize_global_vector(vector<T>* global_v, int local_len, int shift)
{

    if (global_v->n < local_len + shift) {
        printf("(global_v->n = %d) < (local_len = %d) + (shift = %d)\n", global_v->n, local_len, shift);
    }
    assert(global_v->n >= local_len + shift);
    vector<T>* local_v = NULL;
    if (global_v->on_the_device) {
        Vectorinit_CNT
            local_v
            = Vector::init<T>(local_len, true, true);

        cudaError_t err;
        err = cudaMemcpy(local_v->val, &(global_v->val[shift]), local_len * sizeof(T), cudaMemcpyDeviceToDevice);
        CHECK_DEVICE(err);
    } else {
        local_v = Vector::init<T>(local_len, true, false);
        memcpy(local_v->val, &(global_v->val[shift]), sizeof(T) * local_len);
    }

    return local_v;
}

template <typename T>
vector<T>* copyToDevice(vector<T>* v)
{

    assert(!v->on_the_device);

    int n = v->n;

    // alocate vector on the device memory
    vector<T>* v_d = init<T>(n, true, true);

    cudaError_t err = cudaMemcpy(v_d->val, v->val, n * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);

    return v_d;
}

template <typename T>
__global__ void _copy_kernel(itype n, T* dest, T* source)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    dest[i] = source[i];
}

template <typename T>
void copyTo(vector<T>* dest, vector<T>* source)
{

    int n = dest->n < source->n ? dest->n : source->n; /* Massimo Mach 16 2024 for debugging */

    GridBlock gb = gb1d(n, BLOCKSIZE);
    _copy_kernel<<<gb.g, gb.b>>>(n, dest->val, source->val);
}

template <typename T>
void copyToWithOff(vector<T>* dest, vector<T>* source, itype n, itype off)
{

    GridBlock gb = gb1d(n, BLOCKSIZE);
    _copy_kernel<<<gb.g, gb.b>>>(n, dest->val + off, source->val + off);
}

template <typename T>
vector<T>* copyToHost(vector<T>* v_d)
{

    assert(v_d->on_the_device);

    int n = v_d->n;

    // allocate vector on the host memory
    vector<T>* v = init<T>(n, true, false);

    cudaError_t err;

    err = cudaMemcpy(v->val, v_d->val, n * sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    return v;
}

template <typename T>
void free(vector<T>* v)
{
    if (v->on_the_device) {
        cudaError_t err;
        err = cudaFree(v->val);
        CHECK_DEVICE(err);
    } else {
        assert(v->val);
        std::free(v->val);
    }
    std::free(v);
}

template <typename T>
void freeAsync(vector<T>* v, cudaStream_t stream)
{
    if (v->on_the_device) {
        cudaError_t err;
        // err = cudaFree(v->val);
        err = cudaFreeAsync(v->val, stream);
        CHECK_DEVICE(err);
    } else {
        assert(v->val > 0);
        std::free(v->val);
    }
    std::free(v);
}

template <typename T>
bool equals(vector<T>* a, vector<T>* b)
{
    vector<T>*a_ = NULL, *b_ = NULL;
    bool a_dev_flag, b_dev_flag, r = true;

    if (a->on_the_device) {
        a_ = Vector::copyToHost(a);
        a_dev_flag = true;
    } else {
        a_ = a;
        a_dev_flag = false;
    }

    if (b->on_the_device) {
        b_ = Vector::copyToHost(b);
        b_dev_flag = true;
    } else {
        b_ = b;
        b_dev_flag = false;
    }

    cudaDeviceSynchronize();
    if ((a_->n != b_->n)) {
        printf("ERROR: a_->n = %d != %d = b_->n\n", a_->n, b_->n);
        r = false;
    } else {
        for (int i = 0; i < a_->n && r == true; i++) {
            if (a_->val[i] != b_->val[i]) {
                r = false;
            }
        }
    }

    if (a_dev_flag) {
        Vector::free(a_);
    }
    if (b_dev_flag) {
        Vector::free(b_);
    }
    return (r);
}

template <typename T>
bool is_zero(vector<T>* a)
{
    assert(a != NULL);
    vector<T>* a_ = NULL;
    bool a_dev_flag, r = true;

    if (a->on_the_device) {
        a_ = Vector::copyToHost(a);
        a_dev_flag = true;
    } else {
        a_ = a;
        a_dev_flag = false;
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < a_->n && r == true; i++) {
        if (a_->val[i] != 0) {
            r = false;
        }
    }

    if (a_dev_flag) {
        Vector::free(a_);
    }
    return (r);
}

template <typename T>
vector<T>* load(const char* file_name, bool on_the_device)
{
    FILE* fp = fopen(file_name, "r");
    if (fp == NULL) {
        fprintf(stdout, "Error opening file %s, errno = %d: %s\n", file_name, errno, strerror(errno));
        exit(1);
    }

    int n = 0;
    char buffer[BUFSIZE + 1] = { 0 };
    while (fgets(buffer, BUFSIZE, fp) != NULL) {
        n++;
    }

    vector<T>* v = init<T>(n, true, false);
    rewind(fp);
    n = 0;
    while (fgets(buffer, BUFSIZE, fp) != NULL) {
        T val;
        int read = 0;
        if (std::is_same<T, int>::value) {
            read = sscanf(buffer, "%d", &val);
        } else if (std::is_same<T, double>::value) {
            read = sscanf(buffer, "%lf", &val);
        } else {
            printf("Error loading vector from %s: unknown type at line %d\n", file_name, n + 1);
            exit(1);
        }
        if (read != 1) {
            printf("Error loading vector from %s: Missing value at line %d\n", file_name, n + 1);
            exit(1);
        }
        v->val[n] = val;

        n++;
    }

    fclose(fp);

    if (on_the_device) {
        vector<T>* v_ = copyToDevice(v);
        Vector::free(v);
        v = v_;
    }

    return v;
}

template <typename T>
void print(vector<T>* v, int n_, FILE* fp)
{
    vector<T>* v_;

    int n;

    if (n_ == -1) {
        n = v->n;
    } else {
        n = n_;
    }

    if (v->on_the_device) {
        v_ = Vector::copyToHost<T>(v);
    } else {
        v_ = v;
    }

    int i;
    for (i = 0; i < n; i++) {
        if (std::is_same<T, int>::value) {
            fprintf(fp, "%d\n", v_->val[i]);
        } else if (std::is_same<T, double>::value) {
            fprintf(fp, "%g\n", v_->val[i]);
        } else {
            fprintf(fp, "unknown type");
        }
    }

    if (v->on_the_device) {
        Vector::free<T>(v_);
    }
}

template <typename T>
__global__ void _elementwise_div(itype n, T* a, T* b, T* c)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    c[i] = a[i] / b[i];
}

template <typename T>
vector<T>* elementwise_div(vector<T>* a, vector<T>* b, vector<T>* c)
{
    assert(a->n == b->n);

    if (c == NULL) {
        Vectorinit_CNT
            c
            = Vector::init<T>(a->n, true, true);
    }

    GridBlock gb = gb1d(a->n, BLOCKSIZE);
    _elementwise_div<<<gb.g, gb.b>>>(a->n, a->val, b->val, c->val);

    return c;
}

template <typename T>
T dot(cublasHandle_t handle, vector<T>* a, vector<T>* b, int stride_a, int stride_b)
{
    PUSH_RANGE(__func__, 4)

    assert(a->on_the_device == b->on_the_device);

    T result;
    cublasStatus_t cublas_state;
    cublas_state = cublasDdot(handle, a->n, a->val, stride_a, b->val, stride_b, &result);
    CHECK_CUBLAS(cublas_state);

    POP_RANGE
    return result;
}

template <typename T>
void axpy(cublasHandle_t handle, vector<T>* x, vector<T>* y, T alpha, int inc)
{
    PUSH_RANGE(__func__, 4)

    assert(x->on_the_device == y->on_the_device);
    assert(x->n == y->n);

    cublasStatus_t cublas_state;
    cublas_state = cublasDaxpy(handle, x->n, &alpha, x->val, inc, y->val, inc);
    CHECK_CUBLAS(cublas_state);

    POP_RANGE
}

template <typename T>
void axpyWithOff(cublasHandle_t handle, vector<T>* x, vector<T>* y, T alpha, itype n, itype off)
{

    assert(x->on_the_device == y->on_the_device);
    assert(x->n == y->n);
    int inc = 1;

    cublasStatus_t cublas_state;
    cublas_state = cublasDaxpy(handle, n, &alpha, x->val + off, inc, y->val + off, inc);
    CHECK_CUBLAS(cublas_state);
}

template <typename T>
T norm_MPI(cublasHandle_t handle, vector<T>* a)
{
    PUSH_RANGE(__func__, 4)

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
            MPI_COMM_WORLD));

    result = sqrt(result);

    POP_RANGE
    return result;
}

template <typename T>
T norm(cublasHandle_t handle, vector<T>* a, int stride_a)
{
    assert(a->on_the_device);

    T result;
    cublasStatus_t cublas_state;
    cublas_state = cublasDnrm2(handle, a->n, a->val, stride_a, &result);
    CHECK_CUBLAS(cublas_state);
    return result;
}

template <typename T>
void scale(cublasHandle_t handle, vector<T>* x, T alpha, int inc)
{

    assert(x->on_the_device);

    cublasStatus_t cublas_state;
    cublas_state = cublasDscal(handle, x->n, &alpha, x->val, inc);
    CHECK_CUBLAS(cublas_state);
}

// only for very hard DEBUG
template <typename T>
int checkEPSILON(vector<T>* A_)
{
    vector<T>* A = NULL;

    if (A_->on_the_device) {
        A = Vector::copyToHost(A_);
    } else {
        A = A_;
    }

    int count = 0;
    for (int i = 0; i < A->n; i++) {

        if (A->val[i] == DBL_EPSILON) {
            count++;
        }
    }

    return count;
}

// vectorCollection of vector
namespace Collection {

    template <typename T>
    vectorCollection<T>* init(unsigned int n)
    {
        vectorCollection<T>* c = NULL;
        c = (vectorCollection<T>*)Malloc(sizeof(vectorCollection<T>));
        CHECK_HOST(c);

        c->n = n;
        c->val = (vector<T>**)Malloc(n * sizeof(vector<T>*));
        CHECK_HOST(c->val);
        for (int i = 0; i < c->n; i++) {
            c->val[i] = NULL;
        }

        return c;
    }

    template <typename T>
    void free(vectorCollection<T>* c)
    {

        for (int i = 0; i < c->n; i++) {
            if (c->val[i] != NULL) {
                Vector::free(c->val[i]);
            }
        }

        std::free(c->val);
        std::free(c);
    }
}
}

namespace Vector {
template vector<itype>* init<itype>(unsigned int, bool, bool);
template vector<gstype>* init<gstype>(unsigned int, bool, bool);
template vector<vtype>* init<vtype>(unsigned int, bool, bool);

template vectordh<vtype>* initdh<vtype>(int);
template void copydhToD<vtype>(vectordh<vtype>*);
template void copydhToH<vtype>(vectordh<vtype>*);
template void freedh<vtype>(vectordh<vtype>*);

template void fillWithValue<itype>(vector<itype>*, itype);
template void fillWithValue<vtype>(vector<vtype>*, vtype);
template void fillWithValueWithOff<vtype>(vector<vtype>*, vtype, itype, itype);

template vector<itype>* clone<itype>(vector<itype>*);
template vector<gstype>* clone<gstype>(vector<gstype>*);
template vector<vtype>* clone<vtype>(vector<vtype>*);

template vector<itype>* copyToDevice<itype>(vector<itype>*);
template vector<gstype>* copyToDevice<gstype>(vector<gstype>*);
template vector<vtype>* copyToDevice<vtype>(vector<vtype>*);

template void copyTo<itype>(vector<itype>*, vector<itype>*);
template void copyTo<vtype>(vector<vtype>*, vector<vtype>*);
template void copyToWithOff<vtype>(vector<vtype>*, vector<vtype>*, itype, itype);

template vector<itype>* copyToHost<itype>(vector<itype>*);
template vector<vtype>* copyToHost<vtype>(vector<vtype>*);

template void free<itype>(vector<itype>*);
template void free<vtype>(vector<vtype>*);

template vector<itype>* load<itype>(const char* file_name, bool on_the_device);
template vector<vtype>* load<vtype>(const char* file_name, bool on_the_device);

template void print<itype>(vector<itype>*, int, FILE*);
template void print<vtype>(vector<vtype>*, int, FILE*);

template vtype dot<vtype>(cublasHandle_t, vector<vtype>*, vector<vtype>*, int, int);

template vtype norm<vtype>(cublasHandle_t, vector<vtype>*, int);

template vtype norm_MPI<vtype>(cublasHandle_t, vector<vtype>*);

template void axpy<vtype>(cublasHandle_t, vector<vtype>*, vector<vtype>*, vtype, int);
template void axpyWithOff<vtype>(cublasHandle_t, vector<vtype>*, vector<vtype>*, vtype, itype, itype);

template void scale<vtype>(cublasHandle_t, vector<vtype>*, vtype, int);

namespace Collection {

    template vectorCollection<itype>* init<itype>(unsigned int);
    template vectorCollection<vtype>* init<vtype>(unsigned int);

    template void free<itype>(vectorCollection<itype>*);
    template void free<vtype>(vectorCollection<vtype>*);
}
}
