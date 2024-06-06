#pragma once

#define REAL double

struct csrlocinfo {
    int fr;
    int lr;
    int* row;
    int* col;
    REAL* val;
};
