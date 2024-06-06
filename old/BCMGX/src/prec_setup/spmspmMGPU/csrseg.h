
#pragma once
#define REAL double

typedef struct {
	int fr;
	int lr;
	int *row;
	int *col;
	REAL *val;
} csrlocinfo;
