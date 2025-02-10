#include "scalar.h"

#include "utility/memory.h"
#include "utility/utils.h"

namespace Scalar {

template <typename T>
scalar<T>* init(T val, bool on_the_device)
{
    // on the host
    scalar<T>* v = MALLOC(scalar<T>, 1, true);
    v->on_the_device = on_the_device;

    if (on_the_device) {
        // on the device
        v->val = CUDA_MALLOC(T, 1, false);
        cudaError_t err = cudaMemcpy(v->val, &val, sizeof(T), cudaMemcpyHostToDevice);
        CHECK_DEVICE(err);
    } else {
        // on the host
        v->val = MALLOC(T, 1, false);
        v->val[0] = val;
    }
    return v;
}

template <typename T>
scalar<T>* copyToDevice(scalar<T>* v)
{
    assert(!v->on_the_device);

    // alocate scalar on the device memory
    scalar<T>* v_d = init<T>(0, true);

    cudaError_t err = cudaMemcpy(v_d->val, v->val, sizeof(T), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);

    return v_d;
}

template <typename T>
scalar<T>* copyToHost(scalar<T>* v_d)
{
    assert(v_d->on_the_device);

    // alocate scalar on the host memory
    scalar<T>* v = init<T>(0, false);
    cudaError_t err = cudaMemcpy(v->val, v_d->val, sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    return v;
}

template <typename T>
void free(scalar<T>* v)
{
    if (v) {
        if (v->on_the_device) {
            CUDA_FREE(v->val);
        } else {
            FREE(v->val);
        }
        FREE(v);
    }
}

template <typename T>
void print(scalar<T>* v)
{
    scalar<T>* v_;

    if (v->on_the_device) {
        v_ = Scalar::copyToHost<T>(v);
    } else {
        v_ = v;
    }

    std::cout << v_->val[0] << "\n";

    if (v->on_the_device) {
        Scalar::free<T>(v_);
    }
}

// like copyToHost but with less overhead
template <typename T>
T* getvalueFromDevice(scalar<T>* v_d)
{
    assert(v_d->on_the_device);

    // alocate scalar on the host memory
    T* v = MALLOC(int, 1, true);
    cudaError_t err = cudaMemcpy(v, v_d->val, sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    return v;
}
}

namespace Scalar {
template scalar<itype>* init<itype>(itype, bool);
template scalar<vtype>* init<vtype>(vtype, bool);

template scalar<itype>* copyToDevice<itype>(scalar<itype>*);
template scalar<vtype>* copyToDevice<vtype>(scalar<vtype>*);

template scalar<itype>* copyToHost<itype>(scalar<itype>*);
template scalar<vtype>* copyToHost<vtype>(scalar<vtype>*);

template void free<itype>(scalar<itype>*);
template void free<vtype>(scalar<vtype>*);

template void print<itype>(scalar<itype>*);
template void print<vtype>(scalar<vtype>*);

template itype* getvalueFromDevice<itype>(scalar<itype>*);
}
