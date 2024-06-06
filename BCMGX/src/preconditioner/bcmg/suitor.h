#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "utility/handles.h"

#define SUITOR_EPS 3.885780586188048e-10

vector<int>* approx_match_gpu_suitor(handles* h, CSR* A, CSR* W, vector<itype>* M, vector<double>* ws, vector<int>* mutex);
