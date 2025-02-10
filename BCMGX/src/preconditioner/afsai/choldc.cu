#include "choldc.h"

__global__ void cudacholdc1(int n, int initVal, int* position, int* iat, int* ja, REALafsai* coef, int* iat_Filter, int* ja_Filter, REALafsai* coef_Filter)
{

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int p;

    if (idx < n) {
        printf("in cudacholdc1\n");
        p = iat[position[idx]];

        ja_Filter[idx] = ja[p];
        coef_Filter[idx] = rsqrt(coef[p]);
        iat_Filter[idx] = idx + initVal;
        if (idx == (n - 1)) {
            iat_Filter[idx + 1] = idx + initVal + 1;
        }
    }

    return;
}

__global__ void cudacholdcMix(int na[], int number, REALafsai l[], REALafsai x[], int* done, int orow)
{

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j, k, ioffi = 0, ioffj, ooffi = 0, ooffj;
    REALafsai sum;
    int n;

    if (idx < number && !done[idx + orow]) {
        n = na[idx + orow];
        for (i = 0; i < n; i++) {
            ioffj = 0;
            ooffj = ooffi;
            for (j = i; j < n; j++) {
                sum = l[idx + number * (i + ooffj)];
                for (k = 0; k < i; k++) {
                    sum -= l[idx + number * (ooffi + k)] * l[idx + number * (ooffj + k)];
                }
                if (i == j) {
                    if (sum <= 0.0) {
                        return;
                    }
                    l[idx + number * (ooffj + i)] = 1. / sqrt(sum);
                } else {
                    l[idx + number * (ooffj + i)] = sum * l[idx + number * (ooffi + i)];
                }
                ioffj += (n - j);
                ooffj += (j + 1);
            }
            ioffi += (n - i);
            ooffi += (i + 1);
        }
        ooffi = 0;
        for (i = 0; i < n; i++) {
            sum = x[idx + number * i];
            for (j = i - 1; j >= 0; j--) {
                sum -= l[idx + number * (ooffi + j)] * x[idx + number * j];
            }
            x[idx + number * i] = sum * l[idx + number * (ooffi + i)];
            ooffi += (i + 1);
        }
        ooffi = (n * (n - 1) / 2);
        for (i = n - 1; i >= 0; i--) {
            sum = x[idx + number * i];
            ooffj = ooffi + i;
            for (j = i + 1; j < n; j++) {
                sum -= l[idx + number * (ooffj + i + 1)] * x[idx + number * j];
                ooffj += (j + 1);
            }
            x[idx + number * i] = sum * l[idx + number * (ooffi + i)];
            ooffi -= i;
        }
    }
}

/*
 * Determina la decomposta di Cholesky e la sua soluzione.
 * Per sistemi inferiori o uguale a 256 si assegna un blocco per sistema.
 * Diversamente per sistemi maggiori di 256 si assegna un thread per sistema e tali sistemi sono memoriazzati
 * 'mixati'.
 */
void choldc(int number, int n[], REALafsai* l, REALafsai* x, int* done, int orow)
{
    int nBlocks;
    int nThreads = BLOCKDIMENSION;
    nBlocks = (number + nThreads - 1) / nThreads;
    cudacholdcMix<<<nBlocks, nThreads>>>(n, number, l, x, done, orow);
    MY_CHECK_ERROR("cudacholdcMix");
}

/*
 * Risolve sistemi di dimensione pari a 1.
 */
void choldc1(int number, int* position, int* iat, int* ja, REALafsai* coef, int* iat_Filter, int* ja_Filter, REALafsai* coef_Filter, int initVal)
{
    int n = 1;
    int nThreads = BLOCKDIMENSION;
    int nBlocks = (number + nThreads - 1) / nThreads;
    cudacholdc1<<<nBlocks, nThreads>>>(number, initVal, position, iat, ja, coef, iat_Filter, ja_Filter, coef_Filter);
    MY_CHECK_ERROR("cudacholdc1");
}
