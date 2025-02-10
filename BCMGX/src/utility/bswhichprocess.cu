#include "bswhichprocess.h"

int bswhichprocess(gsstype* P_n_per_process, int nprocs, gsstype e)
{
    unsigned int low, high, medium;
    low = 0;
    high = nprocs;
    while (low < high) {
        medium = (high + low) / 2;
        if (e > P_n_per_process[medium]) {
            low = medium + 1;
        } else {
            high = medium;
        }
    }
    return low;
}

int bswhichprocess(gstype* P_n_per_process, int nprocs, gsstype e)
{
    unsigned int low, high, medium;
    low = 0;
    high = nprocs;
    while (low < high) {
        medium = (high + low) / 2;
        if (e > P_n_per_process[medium]) {
            low = medium + 1;
        } else {
            high = medium;
        }
    }
    return low;
}
