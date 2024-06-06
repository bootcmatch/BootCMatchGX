#pragma once

#include <iostream>
#include "utility/utils.h"
#include "utility/setting.h"

template <typename T>
struct scalar{
  bool on_the_device;
  T *val;
};

namespace Scalar{

  template <typename T>
  scalar<T>* init(T val, bool on_the_device);

  template <typename T>
  scalar<T>* copyToDevice(scalar<T> *v);

  template <typename T>
  scalar<T>* copyToHost(scalar<T> *v_d);

  template <typename T>
  void free(scalar<T> *v);

  template <typename T>
  void print(scalar<T> *v);

  // like copyToHost but with less overhead
  template <typename T>
  T* getvalueFromDevice(scalar<T> *v_d);
}
