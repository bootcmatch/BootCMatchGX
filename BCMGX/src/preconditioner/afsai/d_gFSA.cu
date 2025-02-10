#include "utility/precision.h"
#include <stdio.h>

#if !defined(MIN)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

__global__ void d_gatherFullSysAdapt(int nrows, int rowsinpa, int orow, int mmax, unsigned int wkrsppt, int* d_mrow_A, int* d_mrow_old_A, int* d_ia_A, int* d_ja_A, int* d_IWN, REALafsai* d_coef_A, REALafsai* d_full_A, REALafsai* d_rhs, int* done)
{
    //      full_A[i][j] per la riga irow => d_full_sA[irow*mmax*mmax+i*mmax+j]
    //      rhs[i] per la riga irow => d_rhs[irow*mmax+i]
    //      IWN[i] per la riga irow => d_IWN[irow*nrows+i]=(d_IWN+irow*nrows)[i]

    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int irow = tid + orow;

    int mrow, i, ii, jj, row, endrow, jendbloc;

    if ((irow < nrows) && (tid < rowsinpa)) {
        if (!done[irow]) {
            if (d_mrow_A[irow] == d_mrow_old_A[irow]) {
                done[irow] = 1;
            } else {
                jendbloc = irow;
                mrow = d_mrow_A[irow];

                for (i = 0; i < mrow; i++) {
                    d_rhs[i * rowsinpa + (tid)] = 0.;
                    for (ii = i; ii < mrow; ii++) {
                        d_full_A[(((((ii * (ii + 1)) / 2)) + i) * rowsinpa) + (tid)] = 0.0;
                    }
                }

                for (i = 0; i < mrow; i++) {
                    ii = i;
                    row = (d_IWN + irow * mmax)[i];
                    jj = d_ia_A[row];
                    endrow = d_ia_A[row + 1];
                    while (ii < mrow) {
                        while (d_ja_A[jj] < (d_IWN + irow * mmax)[ii] && jj < endrow) {
                            jj++;
                        }
                        if (jj == endrow) {
                            break;
                        }
                        if (d_ja_A[jj] == (d_IWN + irow * mmax)[ii]) {
                            d_full_A[(((((ii * (ii + 1)) / 2)) + i) * rowsinpa) + (tid)] = d_coef_A[jj]; // F90 style
                        }
                        ii++;
                    }
                }

                ii = 0;
                jj = d_ia_A[irow];
                if (d_ja_A[jj] >= jendbloc) {
                    return;
                } // empty rhs

                while (d_ja_A[jj] < jendbloc) {
                    if ((d_IWN + irow * mmax)[ii] > d_ja_A[jj]) {
                        jj++;
                    } else if (d_ja_A[jj] == (d_IWN + irow * mmax)[ii]) {
                        d_rhs[ii * rowsinpa + (tid)] = -d_coef_A[jj];
                        jj++;
                        ii++;
                        if (ii >= mrow) {
                            break;
                        }
                    } else {
                        ii++;
                        if (ii >= mrow) {
                            break;
                        }
                    }
                }
            }
        }
    }
}
