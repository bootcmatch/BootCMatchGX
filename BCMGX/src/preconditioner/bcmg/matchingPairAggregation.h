#pragma once

#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "utility/handles.h"

void matchingPairAggregation(handles* h, CSR* A, vector<vtype>* w, CSR** _P, CSR** _R, bool used_by_solver = true);
