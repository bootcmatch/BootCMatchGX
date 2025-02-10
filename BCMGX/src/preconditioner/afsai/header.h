#ifndef _HEADER_H
#define _HEADER_H
#include "utility/precision.h"

#include <stdio.h>
#include <stdlib.h>
#if defined(OMPAVAIL)
#include <omp.h>
#endif
#include <math.h>
#include <string.h>
#include <time.h>

#define IDEF 1234567890
#define RDEF 1.23456789
#define TRUE 1
#define FALSE 0
#define DEBUG 0

#define BLOCKDIMENSION 256
#define WARPDIMENSION 32
#define MAXBLOCKS 2147483647
#define NB 100000

#define CSR_TYPE 0
#define HYB_TYPE 1

#define AUTO_MODE 0
#define USER_MODE 1
#define MAX_MODE 2

#endif
