#include "utility/precision.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#if !defined(WARP_SZ)
#define WARP_SZ 32
#endif
#define MAXINT 0x7fffffff
#define FULLMASK 0xFFFFFFFF
#undef OLDSHUFFLE

/**
 * @brief Performs a binary search for a value within a sorted array.
 * 
 * This device function implements a binary search algorithm to find the index of the 
 * given value in the sorted array `array[]`. If the value is not present, it returns
 * the position where the value should be inserted.
 * 
 * @param array A sorted array of integers.
 * @param size The number of elements in the array.
 * @param value The value to search for in the array.
 * 
 * @return The index where the value is found or where it should be inserted.
 */
__device__ int binsearch(int array[], unsigned int size, int value)
{
    unsigned int low, high, medium;
    low = 0;
    high = size + 1;
    while (low < high) {
        medium = (high + low) / 2;
        if (array[medium] < value) {
            low = medium + 1;
        } else {
            high = medium;
        }
    }
    return low;
}

/**
 * @brief Computes the minimum value among a set of values using warp-wide synchronization.
 * 
 * This device function computes the minimum value among the input value `val` and its 
 * values from other threads in the same warp, using shuffle-based synchronization.
 * 
 * @param val The value to compare with other values in the warp.
 * 
 * @return The minimum value within the warp.
 */
__device__ int computemin(int val)
{
    int scratchi;
#if !defined(OLDSHUFFLE)
    unsigned int mask = FULLMASK;
#endif
    for (int offset = WARP_SZ >> 1; offset > 0; offset /= 2) {
#if !defined(OLDSHUFFLE)
        scratchi = __shfl_xor_sync(mask, val, offset);
#else
        scratchi = __shfl_xor(val, offset, WARP_SZ);
#endif
        if (scratchi < val) {
            val = scratchi;
        }
    }
    return val;
}

/**
 * @brief Computes the next power of 2 greater than or equal to a given value.
 * 
 * This device function calculates the next power of 2 greater than or equal to the 
 * input integer `x` using bitwise operations.
 * 
 * @param x The input value.
 * 
 * @return The smallest power of 2 greater than or equal to `x`.
 */
__device__ unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/**
 * @brief Performs a parallel sorting and updating process for rows in a matrix.
 * 
 * This inline device function performs a sorting procedure on the values in `R_vec`, 
 * associates them with their respective indices in `I_vec`, and updates them based on 
 * specific criteria. It uses warp-level synchronization to ensure correct ordering 
 * and atomic operations for updating the values.
 * 
 * @param n The total number of elements to be sorted.
 * @param ncut The cutoff value for sorting.
 * @param R_vec A pointer to the vector of values to be sorted.
 * @param I_vec A pointer to the vector of indices associated with `R_vec`.
 */
__inline__ __device__ void RISortSplit(int n, int ncut, REALafsai* R_vec, int* I_vec)
{
    int lane = threadIdx.x % WARP_SZ;
    int scratchi, maxind;
    int first = 0;
    int last = n;
    int done = 0;
    REALafsai scratchf = 0., maxval;
    while (1) {
#if !defined(OLDSHUFFLE)
        if (__ballot_sync(FULLMASK, done) == FULLMASK) {
            break;
        }
#endif
        if (!done) {
            maxind = first;
            maxval = FABS(R_vec[maxind]);
            for (int i = lane + first + 1; i < last; i += WARP_SZ) {
                if (FABS(R_vec[i]) > maxval) {
                    maxval = FABS(R_vec[i]);
                    maxind = i;
                }
            }
        }
#if !defined(OLDSHUFFLE)
        unsigned int maskxshfl = __ballot_sync(FULLMASK, !done);
#endif
        if (!done) {
            for (int offset = WARP_SZ / 2; offset > 0; offset /= 2) {
#if !defined(OLDSHUFFLE)
                scratchf = __shfl_down_sync(maskxshfl, maxval, offset, (last < WARP_SZ) ? last : WARP_SZ);
                scratchi = __shfl_down_sync(maskxshfl, maxind, offset, (last < WARP_SZ) ? last : WARP_SZ);
#else
                scratchf = __shfl_down(maxval, offset, (last < WARP_SZ) ? last : WARP_SZ);
                scratchi = __shfl_down(maxind, offset, (last < WARP_SZ) ? last : WARP_SZ);
#endif
                if (FABS(scratchf) > maxval) {
                    maxval = FABS(scratchf);
                    maxind = scratchi;
                }
            }
            if (lane == 0) {
                scratchf = R_vec[first];
                scratchi = I_vec[first];
                R_vec[first] = R_vec[maxind];
                I_vec[first] = I_vec[maxind];
                R_vec[maxind] = scratchf;
                I_vec[maxind] = scratchi;
            }
            first++;
            if (first == ncut) {
                done = 1;
            }
        }
    }
}

/**
 * @brief CUDA kernel for processing and updating matrix rows in parallel.
 * 
 * This kernel performs various operations on matrix rows, including element-wise 
 * computations, sorting, and updating of the `WI` and `WR` arrays. It uses warp-wide 
 * synchronization for efficient parallel execution and atomic operations for thread-safety.
 * 
 * The kernel processes a subset of matrix rows and applies the necessary computations 
 * based on input coefficients, right-hand side values, and other parameters.
 * 
 * @param nrows The total number of rows in the matrix.
 * @param lastrow The last row index to process.
 * @param rowsinpa The number of rows in parallel.
 * @param orow The starting row index.
 * @param mmax The maximum number of matrix elements.
 * @param wkrsppt The number of elements processed by each thread.
 * @param mrow_A A pointer to the array containing the row indices.
 * @param lfil The maximum allowable number of elements per row.
 * @param ia A pointer to the array representing the row-pointer structure.
 * @param ja A pointer to the array representing the column indices of non-zero elements.
 * @param coef A pointer to the array of coefficients for each matrix element.
 * @param rhs A pointer to the array of right-hand side values.
 * @param IWN A pointer to the array storing the updated indices.
 * @param WI A pointer to the array storing the updated column indices.
 * @param WR A pointer to the array storing the updated right-hand side values.
 * @param done A pointer to an array indicating whether a row has been processed.
 */
__launch_bounds__(1024, 2)
    __global__ void d_kapGrad_merge(int nrows, int lastrow, int rowsinpa, int orow, int mmax,
        unsigned int wkrsppt, int mrow_A[], int lfil, int* ia, int* ja,
        REALafsai* coef, REALafsai* rhs, int* IWN, int* WI, REALafsai* WR,
        int* done)
{

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int i, j, iirow;
    unsigned int jjcol, jjcol2;
    // Local row
    int irow = (tid / WARP_SZ) + orow;
    int shift1 = irow * mmax;
    int shift2 = 2 * (irow % rowsinpa) * wkrsppt;
    int ind_iirow, len_iirow;
    int ind2_iirow, len2_iirow = 0;
    int indWR_loc;
    int ncut;
    unsigned int scratch;
    REALafsai fac;
    REALafsai fac2;

    int lane = threadIdx.x % WARP_SZ;
    int wid = threadIdx.x / WARP_SZ;
    int ioc, ish, pioc;
    int ioc2 = 0, pioc2;
    __shared__ int sWI[WARP_SZ * WARP_SZ];
    __shared__ REALafsai sWR[WARP_SZ * WARP_SZ];
    if (irow < lastrow && ((irow - orow) < rowsinpa) && !done[irow]) {
        int jendbloc = irow;
        unsigned int mrow = mrow_A[irow];
        indWR_loc = 0;
        // Get the degree zero entries (A pattern) and the higher degree entries
        {
            i = lane;
            if (i <= mrow) {
                iirow = lane ? (IWN + shift1)[i - 1] : irow; /* first lane loads irow */
                fac = lane ? (rhs + shift1)[i - 1] : v(1.0);
                ind_iirow = ia[iirow];
                len_iirow = ia[iirow + 1] - ind_iirow;
            } else {
                len_iirow = 0;
            }
            (sWR + wid * WARP_SZ)[i] = v(0.0);
            if (mrow > (WARP_SZ - 1)) {
                i += WARP_SZ;
                if (i <= mrow) {
                    iirow = (IWN + shift1)[i - 1];
                    fac2 = (rhs + shift1)[i - 1];
                    ind2_iirow = ia[iirow];
                    len2_iirow = ia[iirow + 1] - ind2_iirow;
                }
            }
            ish = ioc = 0;
            pioc = -1;
            pioc2 = -1;
            while (1) {
#if !defined(OLDSHUFFLE)
                __syncwarp();
#endif
                if (pioc < ioc) {
                    jjcol = (ioc < len_iirow) ? (ja + ind_iirow)[ioc] : MAXINT;
                    pioc = ioc;
                }
                if (pioc2 < ioc2) {
                    jjcol2 = (ioc2 < len2_iirow) ? (ja + ind2_iirow)[ioc2] : MAXINT;
                    pioc2 = ioc2;
                }
                scratch = (jjcol < jjcol2) ? jjcol : jjcol2;

                if (scratch >= jendbloc) {
                    scratch = MAXINT;
                }
#if !defined(OLDSHUFFLE)
                unsigned int jmin = (mrow > 0) ? computemin(scratch) : __shfl_sync(FULLMASK, scratch, 0);
#else
                unsigned int jmin = (mrow > 0) ? computemin(scratch) : __shfl(scratch, 0);
#endif
                if (jmin == MAXINT) {
                    break;
                }
                if (lane == ish) {
                    (sWI + wid * WARP_SZ)[ish] = jmin;
                }

                if (jmin == jjcol) {
                    atomicAdd(sWR + wid * WARP_SZ + ish, coef[ind_iirow + ioc] * fac);
                    ioc++;
                }
                if (jmin == jjcol2) {
                    atomicAdd(sWR + wid * WARP_SZ + ish, coef[ind2_iirow + ioc2] * fac2);
                    ioc2++;
                }

                ish++;
                if (ish == WARP_SZ) {
#if !defined(OLDSHUFFLE)
                    __syncwarp();
#endif
                    (WI + shift2 + indWR_loc)[lane] = (sWI + wid * WARP_SZ)[lane];
                    (WR + shift2 + indWR_loc)[lane] = (sWR + wid * WARP_SZ)[lane];
                    (sWR + wid * WARP_SZ)[lane] = v(0.0);
                    indWR_loc += WARP_SZ;
                    ish = 0;
                }
            }
#if !defined(OLDSHUFFLE)
            __syncwarp();
#endif
            if (lane < ish) {
                (WI + shift2 + indWR_loc)[lane] = (sWI + wid * WARP_SZ)[lane];
                (WR + shift2 + indWR_loc)[lane] = (sWR + wid * WARP_SZ)[lane];
            }
            indWR_loc += ish;
        }
#if !defined(OLDSHUFFLE)
        __syncwarp();
#else
        __syncthreads();
#endif

        // Remove already computed entries
        for (i = lane; i < mrow; i += WARP_SZ) {
            j = binsearch((WI + shift2), indWR_loc - 1, (IWN + shift1)[i]);
            if ((WI + shift2)[j] == (IWN + shift1)[i]) {
                (WR + shift2)[j] = 0.0;
            }
        }

        ncut = (lfil > indWR_loc) ? indWR_loc : lfil;
        int cncut = 0, ldone;
        if (ncut > 0) {
            RISortSplit(indWR_loc, ncut, (WR + shift2), &((WI + shift2)[0]));
        }
        while (cncut < ncut) {
            // Convergence check
            if ((WR + shift2 + cncut)[0] == 0.0) {
                break;
            }
            int newi = (WI + shift2 + cncut)[0];
            if (lane == 0) {
                (IWN + irow * mmax)[mrow + cncut] = newi;
            }
            i = mrow + cncut - WARP_SZ + lane;
            ldone = 0;
            while (1) {
                if (i < 0) {
                    ldone = 1;
                } else {
                    scratch = (IWN + irow * mmax)[i];
                }
                if (__ballot_sync(FULLMASK, ldone) == FULLMASK) {
                    break;
                }
                unsigned int maskxbal = __ballot_sync(FULLMASK, (!ldone && (scratch > newi)));
                if (!ldone && (scratch > newi)) {
                    (IWN + irow * mmax)[i + 1] = scratch;
                    // select the leader
                    int leader = __ffs(maskxbal) - 1;
                    // leader does the update
                    if (lane == leader) {
                        (IWN + irow * mmax)[i] = newi;
                    }
                    if (leader > 0) {
                        ldone = 1;
                    }
                }
#if !defined(OLDSHUFFLE)
                __syncwarp(FULLMASK);
#endif
                i -= WARP_SZ;
            }
            cncut++;
        }

        if (lane == 0) {
            mrow += (cncut);
            mrow_A[irow] = mrow;
        }
    }
}
