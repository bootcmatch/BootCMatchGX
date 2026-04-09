#pragma once

#include "utility/setting.h"

long* getmct(gsstype* Col, int nnz, int f, int l, int* uvs, long** bitcol, int* bitcolsize, int* nonuniquesize, int num_thr);
long* getmct_4shrink(long* Col, int nnz, int f, int l, int first_or_last, int* uvs, long** bitcol, int* bitcolsize, int* nonuniquesize, int* post_local, int num_thr);
