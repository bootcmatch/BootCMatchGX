#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "utility/setting.h"

vtype Anorm(cublasHandle_t handle, CSR* A, vector<vtype>* v);
