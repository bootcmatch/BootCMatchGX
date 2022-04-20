#define BIN_NUM 7
#define PWARP 4
#define IMB_PWMIN 32
#define B_PWMIN 16
#define IMB_MIN 512
#define B_MIN 256
#define IMB_PW_SH_SIZE 2048
#define B_PW_SH_SIZE 1024
#define IMB_SH_SIZE 1024
#define B_SH_SIZE 512

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <nsparse.h>
#include <nsparse_asm.h>

/* SpGEMM Specific Parameters */
#define HASH_SCAL 107 // Set disjoint number to COMP_SH_SIZE
#define ONSTREAM
#define SHFLMASK 0xFFFFFFFF
#define DBINFO binfo->fr, binfo->lr, binfo->row, binfo->col

void init_bin(sfBIN *bin, int M)
{
    int i;
    bin->stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * BIN_NUM);
    for (i = 0; i < BIN_NUM; i++) {
        cudaStreamCreate(&(bin->stream[i]));
    }
  
    bin->bin_size = (int *)malloc(sizeof(int) * BIN_NUM);
    bin->bin_offset = (int *)malloc(sizeof(int) * BIN_NUM);
    checkCudaErrors(cudaMalloc((void **)&(bin->d_row_perm), sizeof(int) * M));
    checkCudaErrors(cudaMalloc((void **)&(bin->d_row_nz), sizeof(int) * (M + 1)));
    checkCudaErrors(cudaMalloc((void **)&(bin->d_max), sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&(bin->d_bin_size), sizeof(int) * BIN_NUM));
    checkCudaErrors(cudaMalloc((void **)&(bin->d_bin_offset), sizeof(int) * BIN_NUM));
    i = 0;
    bin->max_intprod = 0;
    bin->max_nz = 0;
}

void release_bin(sfBIN bin)
{
    int i;
    cudaFree(bin.d_row_nz);
    cudaFree(bin.d_row_perm);
    cudaFree(bin.d_max);
    cudaFree(bin.d_bin_size);
    cudaFree(bin.d_bin_offset);

    free(bin.bin_size);
    free(bin.bin_offset);
    for (i = 0; i < BIN_NUM; i++) {
        cudaStreamDestroy(bin.stream[i]);
    }
    free(bin.stream);
}

__global__ void set_intprod_num(int *d_arpt, int *d_acol,
                                const int* __restrict__ d_brpt, int binfo_fr, int binfo_lr, int *binfo_row, int *binfo_col,
                                int *d_row_intprod, int *d_max_intprod,
                                int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    int nz_per_row = 0;
    int j;
    for (j = d_arpt[i]; j < d_arpt[i + 1]; j++) {
	if((binfo_col==NULL) || d_acol[j]<binfo_fr || d_acol[j]>binfo_lr) { 
       	     nz_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
	} else {
       	     nz_per_row += binfo_row[d_acol[j]-binfo_fr + 1] - binfo_row[d_acol[j]-binfo_fr];
	}
    }
    d_row_intprod[i] = nz_per_row;
    atomicMax(d_max_intprod, nz_per_row);
}

__global__ void set_bin(int *d_row_nz, int *d_bin_size, int *d_max,
                        int M, int min, int mmin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    int nz_per_row = d_row_nz[i];

    atomicMax(d_max, nz_per_row);

    int j = 0;
    for (j = 0; j < BIN_NUM - 2; j++) {
        if (nz_per_row <= (min << j)) {
            if (nz_per_row <= (mmin)) {
                atomicAdd(d_bin_size + j, 1);
            }
            else {
                atomicAdd(d_bin_size + j + 1, 1);
            }
            return;
        }
    }
    atomicAdd(d_bin_size + BIN_NUM - 1, 1);
}

__global__ void init_row_perm(int *d_permutation, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M) {
        return;
    }
  
    d_permutation[i] = i;
}

__global__ void set_row_perm(int *d_bin_size, int *d_bin_offset,
                             int *d_max_row_nz, int *d_row_perm,
                             int M, int min, int mmin)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M) {
        return;
    }

    int nz_per_row = d_max_row_nz[i];
    int dest;
  
    int j = 0;
    for (j = 0; j < BIN_NUM - 2; j++) {
        if (nz_per_row <= (min << j)) {
            if (nz_per_row <= mmin) {
                dest = atomicAdd(d_bin_size + j, 1);
                d_row_perm[d_bin_offset[j] + dest] = i;
            }
            else {
                dest = atomicAdd(d_bin_size + j + 1, 1);
                d_row_perm[d_bin_offset[j + 1] + dest] = i;
            }
            return;
        }
    }
    dest = atomicAdd(d_bin_size + BIN_NUM - 1, 1);
    d_row_perm[d_bin_offset[BIN_NUM - 1] + dest] = i;

}

void set_max_bin(int *d_arpt, int *d_acol, int *d_brpt, csrlocinfo *binfo, sfBIN *bin, int M)
{
    int i;
    int GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin->bin_size[i] = 0;
        bin->bin_offset[i] = 0;
    }
  
    cudaMemcpy(bin->d_bin_size, bin->bin_size, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(bin->d_max, &(bin->max_intprod), sizeof(int), cudaMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    set_intprod_num<<<GS, BS>>>(d_arpt, d_acol, d_brpt, DBINFO, bin->d_row_nz, bin->d_max, M);
    cudaMemcpy(&(bin->max_intprod), bin->d_max, sizeof(int), cudaMemcpyDeviceToHost);
  
    if (bin->max_intprod > IMB_PWMIN) {
        set_bin<<<GS, BS>>>(bin->d_row_nz, bin->d_bin_size, bin->d_max, M, IMB_MIN, IMB_PWMIN);
  
        cudaMemcpy(bin->bin_size, bin->d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(bin->d_bin_size, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin->bin_offset[i + 1] = bin->bin_offset[i] + bin->bin_size[i];
        }
        cudaMemcpy(bin->d_bin_offset, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        set_row_perm<<<GS, BS>>>(bin->d_bin_size, bin->d_bin_offset, bin->d_row_nz, bin->d_row_perm, M, IMB_MIN, IMB_PWMIN);
    }
    else {
        bin->bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_size[i] = 0;
        }
        bin->bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_offset[i] = M;
        }
        init_row_perm<<<GS, BS>>>(bin->d_row_perm, M);
    }
}


void set_min_bin(sfBIN *bin, int M)
{
    int i;
    int GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin->bin_size[i] = 0;
        bin->bin_offset[i] = 0;
    }
  
    cudaMemcpy(bin->d_bin_size, bin->bin_size, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(bin->d_max, &(bin->max_nz), sizeof(int), cudaMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    set_bin<<<GS, BS>>>(bin->d_row_nz, bin->d_bin_size,
                        bin->d_max,
                        M, B_MIN, B_PWMIN);
  
    cudaMemcpy(&(bin->max_nz), bin->d_max, sizeof(int), cudaMemcpyDeviceToHost);
    if (bin->max_nz > B_PWMIN) {
        cudaMemcpy(bin->bin_size, bin->d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(bin->d_bin_size, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin->bin_offset[i + 1] = bin->bin_offset[i] + bin->bin_size[i];
        }
        cudaMemcpy(bin->d_bin_offset, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
  
        set_row_perm<<<GS, BS>>>(bin->d_bin_size, bin->d_bin_offset, bin->d_row_nz, bin->d_row_perm, M, B_MIN, B_PWMIN);
    }

    else {
        bin->bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_size[i] = 0;
        }
        bin->bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++) {
            bin->bin_offset[i] = M;
        }
        BS = 1024;
        GS = div_round_up(M, BS);
        init_row_perm<<<GS, BS>>>(bin->d_row_perm, M);
    }
}

__global__ void init_value(real *d_val, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz) {
        return;
    }
    d_val[i] = 0;
}

__global__ void init_check(int *d_check, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz) {
        return;
    }
    d_check[i] = -1;
}

#define BCOLARG int binfo_fr, int binfo_lr, int *binfo_row, int *binfo_col,

#define SELROW(a)	    if((binfo_col==NULL) || (a)<binfo_fr || (a)>binfo_lr) { \
	            brow=(int *)d_brpt; \
                    bacol=(int *)d_bcol; \
		    rowoff=0; \
	    } else { \
		    brow=binfo_row; \
		    bacol=binfo_col; \
		    rowoff=binfo_fr; \
	    } 


__global__ void set_row_nz_bin_pwarp(const int *d_arpt, const int *d_acol,
                                     const int* __restrict__ d_brpt,
                                     const int* __restrict__ d_bcol, BCOLARG
                                     const int *d_row_perm,
                                     int *d_row_nz,
                                     int bin_offset, int M) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / PWARP;
    int tid = i % PWARP;
    int local_rid = rid % (blockDim.x / PWARP);
  
    int j, k;
    int soffset;
    int acol, bcol, key, hash, adr, nz, old;
    __shared__ int check[IMB_PW_SH_SIZE];
  
    soffset = local_rid * IMB_PWMIN;
    int ipwarp=i&(WARP-1);
    int ipmask=ipwarp/PWARP;
    unsigned int shflmask=__ballot_sync(SHFLMASK, ((ipmask*PWARP)<=ipwarp)&&(ipwarp<((ipmask+1)*PWARP)) );
  
    for (j = tid; j < IMB_PWMIN; j += PWARP) {
        check[soffset + j] = -1;
    }

    if (rid >= M) {
        return;
    }
    
    __syncthreads();
    rid = d_row_perm[rid + bin_offset];
    nz = 0;
    int *brow, *bacol, rowoff;
    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        acol = ld_gbl_int32(d_acol + j);
	SELROW(acol);
        for (k = brow[acol-rowoff]; k < brow[acol-rowoff + 1]; k++) {
	    bcol=bacol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (IMB_PWMIN - 1);
            adr = soffset + hash;
	       if (check[adr] != key) {
                                              //@@@@@ IL PROBLEMA E SE DUE Jcol UGUALI
                                              //@@@@@ SUPERANO ASSIEME IL CONTROLLO PRECEDENTE
                                              //@@@@@ E UNO SCRIVE IL SUO VALORE PRIMA CHE L'ALTRO
                                              //@@@@@ FACCIA IL CONTROLLO
                  // Add the key
                  while (1){
                     old = atomicCAS(check + adr, -1, key);
                     if (old == -1) {
			nz++;
                        break;
                     } else {
                        if (old != key){ //@@@@@ CONTROLLO AGGIUNTO PER EVITARE DUPLICAZIONE
                           hash = (hash + 1) & (IMB_PWMIN - 1);
                    	   adr = soffset + hash;
                        } else {
                           break;
                        }
                     }
                  }
               }
        }
    }
  
    for (j = PWARP / 2; j >= 1; j /= 2) {
        nz += __shfl_xor_sync(shflmask,nz, j);
    }

    if (tid == 0) {
        d_row_nz[rid] = nz;
    }
}

template <int SH_ROW>
__global__ void set_row_nz_bin_each(const int *d_arpt, const int *d_acol,
                                    const int* __restrict__ d_brpt,
                                    const int* __restrict__ d_bcol, BCOLARG
                                    const int *d_row_perm,
                                    int *d_row_nz, int bin_offset, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / WARP;
    int tid = i % WARP;
    int wid = rid % (blockDim.x / WARP);
    int j, k, l;
    int bcol, key, hash, old;
    int nz, adr;
    int acol, ccol;
    int soffset;

    soffset = wid * SH_ROW;

    __shared__ int check[IMB_SH_SIZE];
    for (j = tid; j < SH_ROW; j += WARP) {
        check[soffset + j] = -1;
    }
  
    if (rid >= M) {
        return;
    }
    __syncthreads();
  
    acol = 0;
    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    int *brow, *bacol, rowoff;
    for (j = d_arpt[rid]; j < d_arpt[rid + 1]; j += WARP) {
        if (j + tid < d_arpt[rid + 1]) acol = ld_gbl_int32(d_acol + j + tid);
        for (l = 0; l < WARP && j + l < d_arpt[rid + 1]; l++) {
            ccol = __shfl_sync(SHFLMASK,acol, l);
	    SELROW(ccol);
            for (k = brow[ccol-rowoff] + tid; k < brow[ccol-rowoff + 1]; k += WARP) {
		bcol=bacol[k];
                key = bcol;
                hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
                adr = soffset + hash;
                if (check[adr] != key) {
                                              //@@@@@ IL PROBLEMA E SE DUE Jcol UGUALI
                                              //@@@@@ SUPERANO ASSIEME IL CONTROLLO PRECEDENTE
                                              //@@@@@ E UNO SCRIVE IL SUO VALORE PRIMA CHE L'ALTRO
                                              //@@@@@ FACCIA IL CONTROLLO
                  // Add the key
                  while (1){
                     old = atomicCAS(check + adr, -1, key);
                     if (old == -1) {
		        nz++;
                        break;
                     } else {
                        if (old != key){ //@@@@@ CONTROLLO AGGIUNTO PER EVITARE DUPLICAZIONE
                           hash = (hash + 1) & (SH_ROW - 1);
                           adr = soffset + hash;
                        } else {
                           break;
                        }
                     }
                  }
                }
            }
        }
    }

    for (j = WARP / 2; j >= 1; j /= 2) {
        nz += __shfl_xor_sync(SHFLMASK,nz, j);
    }
    __syncthreads();
    
    if (tid == 0) {
        d_row_nz[rid] = nz;
    }
}

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb(const int *d_arpt, const int *d_acol,
                                       const int* __restrict__ d_brpt,
                                       const int* __restrict__ d_bcol, BCOLARG
                                       int *d_row_perm, int *d_row_nz,
                                       int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j, k;
    int bcol, key, hash, old;
    int nz, adr;
    int acol;

    __shared__ int check[SH_ROW];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        check[j] = -1;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();

    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    int *brow, *bacol, rowoff;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        SELROW(acol);
        for (k = brow[acol-rowoff] + tid; k < brow[acol-rowoff + 1]; k += WARP) {
	    bcol=bacol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            if (check[adr] != key) {
                                              //@@@@@ IL PROBLEMA E SE DUE Jcol UGUALI
                                              //@@@@@ SUPERANO ASSIEME IL CONTROLLO PRECEDENTE
                                              //@@@@@ E UNO SCRIVE IL SUO VALORE PRIMA CHE L'ALTRO
                                              //@@@@@ FACCIA IL CONTROLLO
                  // Add the key
                  while (1){
                     old = atomicCAS(check + adr, -1, key);
                     if (old == -1) {
                        nz++;
                        break;
                     } else {
                        if (old != key){ //@@@@@ CONTROLLO AGGIUNTO PER EVITARE DUPLICAZIONE
                           hash = (hash + 1) & (SH_ROW - 1);
                           adr = hash;
                        } else {
                           break;
                        }
                     }
                  }
            }
        }
    }

    for (j = WARP / 2; j >= 1; j /= 2) {
        nz += __shfl_xor_sync(SHFLMASK,nz, j);
    }

    __syncthreads();  
    if (threadIdx.x == 0) {
        check[0] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(check, nz);
    }
    __syncthreads();
  
    if (threadIdx.x == 0) {
        d_row_nz[rid] = check[0];
    }
}

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb_large(const int *d_arpt, const int *d_acol,
                                             const int* __restrict__ d_brpt,
                                             const int* __restrict__ d_bcol, BCOLARG
                                             int *d_row_perm, int *d_row_nz,
                                             int *d_fail_count, int *d_fail_perm,
                                             int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j, k;
    int bcol, key, hash, old;
    int adr;
    int acol;

    __shared__ int check[SH_ROW];
    __shared__ int snz[1];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        check[j] = -1;
    }
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();
  
    rid = d_row_perm[rid + bin_offset];
    int count = 0;
    int border = SH_ROW >> 1;
    int *brow, *bacol, rowoff;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        SELROW(acol);
        for (k = brow[acol-rowoff] + tid; k < brow[acol-rowoff + 1]; k += WARP) {
	    bcol=bacol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (count < border && snz[0] < border) {
                if (check[adr] == key) {
                    break;
                }
                else if (check[adr] == -1) {
                    old = atomicCAS(check + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(snz, 1);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                    adr = hash;
                    count++;
                }
            }
            if (count >= border || snz[0] >= border) {
                break;
            }
        }
        if (count >= border || snz[0] >= border) {
            break;
        }
    }
  
    __syncthreads();
    if (count >= border || snz[0] >= border) {
        if (threadIdx.x == 0) {
            int d = atomicAdd(d_fail_count, 1);
            d_fail_perm[d] = rid;
        }
    }
    else {
        if (threadIdx.x == 0) {
            d_row_nz[rid] = snz[0];
        }
    }
}

__global__ void set_row_nz_bin_each_gl(const int *d_arpt, const int *d_acol,
                                       const int* __restrict__ d_brpt,
                                       const int* __restrict__ d_bcol, BCOLARG
                                       const int *d_row_perm,
                                       int *d_row_nz, int *d_check,
                                       int max_row_nz, int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j, k;
    int bcol, key, hash, old;
    int nz, adr;
    int acol;
    int offset = rid * max_row_nz;

    __shared__ int snz[1];
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
  
    if (rid >= M) {
        return;
    }

    __syncthreads();
  
    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    int *brow, *bacol, rowoff;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        SELROW(acol);
        for (k = brow[acol-rowoff] + tid; k < brow[acol-rowoff + 1]; k += WARP) {
	    bcol=bacol[k];
            hash = (bcol * HASH_SCAL) % max_row_nz;
            adr = offset + hash;
            if (d_check[adr] != key) {
                                              //@@@@@ IL PROBLEMA E SE DUE Jcol UGUALI
                                              //@@@@@ SUPERANO ASSIEME IL CONTROLLO PRECEDENTE
                                              //@@@@@ E UNO SCRIVE IL SUO VALORE PRIMA CHE L'ALTRO
                                              //@@@@@ FACCIA IL CONTROLLO
                  // Add the key
                  while (1){
                     old = atomicCAS(d_check + adr, -1, key);
                     if (old == -1) {
                        nz++;
                        break;
                     } else {
                        if (old != key){ //@@@@@ CONTROLLO AGGIUNTO PER EVITARE DUPLICAZIONE
                    	   hash = (hash + 1) % max_row_nz;
            	           adr = offset + hash;
                        } else {
                           break;
                        }
                     }
                  }
            }
        }
    }
  
    for (j = WARP / 2; j >= 1; j /= 2) {
        nz += __shfl_xor_sync(SHFLMASK,nz, j);
    }
    __syncthreads();
  
    if (tid == 0) {
        atomicAdd(snz, nz);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        d_row_nz[rid] = snz[0];
    }
}

#define BVALCOLARG int binfo_fr, int binfo_lr, int *binfo_row, int *binfo_col, real *binfo_val,

#define SELVALROW(a)	    if((binfo_col==NULL) || (a)<binfo_fr || (a)>binfo_lr) { \
	            brow=(int *)d_brpt; \
                    bacol=(int *)d_bcol; \
                    baval=(real *)d_bval; \
		    rowoff=0; \
	    } else { \
		    brow=binfo_row; \
		    bacol=binfo_col; \
		    baval=binfo_val; \
		    rowoff=binfo_fr; \
	    } 


void set_row_nnz(int *d_arpt, int *d_acol,
                 int *d_brpt, int *d_bcol, csrlocinfo *,
                 int *d_crpt,
                 sfBIN *bin,
                 int M, int *nnz);
  
__global__ void calculate_value_col_bin_pwarp(const int *d_arpt,
                                              const int *d_acol,
                                              const real *d_aval,
                                              const int* __restrict__ d_brpt,
                                              const int* __restrict__ d_bcol,
                                              const real* __restrict__ d_bval, BVALCOLARG
                                              int *d_crpt,
                                              int *d_ccol,
                                              real *d_cval,
                                              const int *d_row_perm,
                                              int *d_nz,
                                              int bin_offset,
                                              int bin_size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / PWARP;
    int tid = i % PWARP;
    int local_rid = rid % (blockDim.x / PWARP);
    int j;
    __shared__ int shared_check[B_PW_SH_SIZE];
    __shared__ real shared_value[B_PW_SH_SIZE];
  
    int soffset = local_rid * (B_PWMIN);
  
    for (j = tid; j < (B_PWMIN); j += PWARP) {
        shared_check[soffset + j] = -1;
        shared_value[soffset + j] = 0;
    }
    int ipwarp=i&(WARP-1);
    int ipmask=ipwarp/PWARP;
    unsigned int shflmask=__ballot_sync(SHFLMASK, ((ipmask*PWARP)<=ipwarp)&&(ipwarp<((ipmask+1)*PWARP)) );
  
    if (rid >= bin_size) {
        return;
    }
    rid = d_row_perm[rid + bin_offset];

    if (tid == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads(); 
    int k;
    int acol, bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    real aval, bval;

    int *brow, *bacol, rowoff;
    real *baval;
    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        acol = ld_gbl_int32(d_acol + j);
        aval = ld_gbl_real(d_aval + j);
	SELVALROW(acol);
        for (k = brow[acol-rowoff]; k < brow[acol-rowoff + 1]; k++) {
	    bcol=bacol[k];
	    bval=baval[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & ((B_PWMIN) - 1);
            adr = soffset + hash;
	    if (shared_check[adr] != key) {
                                              //@@@@@ IL PROBLEMA E SE DUE Jcol UGUALI
                                              //@@@@@ SUPERANO ASSIEME IL CONTROLLO PRECEDENTE
                                              //@@@@@ E UNO SCRIVE IL SUO VALORE PRIMA CHE L'ALTRO
                                              //@@@@@ FACCIA IL CONTROLLO
                  // Add the key
                  while (1){
                     old = atomicCAS(shared_check + adr, -1, key);
                     if (old == -1 || old == key) {
                        atomic_fadd(shared_value + adr, aval * bval);
                        break;
                     } 
                     hash = (hash + 1) & (B_PWMIN - 1);
                     adr = soffset + hash;
                  }
            } else {
                    atomic_fadd(shared_value + adr, aval * bval);
	    }
        }
    }
    __syncwarp(); 
    int col; 
    real val;

    for (j = tid; j < (B_PWMIN); j += PWARP) {
	col = shared_check[soffset + j];
        val = shared_value[soffset + j];
//	__syncwarp(shflmask);
        if (shared_check[soffset + j] != -1) {
            index = atomicAdd(d_nz + rid, 1);
            shared_check[soffset + index] = col;
            shared_value[soffset + index] = val;
        }
//	__syncwarp(shflmask);
    }

    int nz = d_nz[rid];

    // Sorting for shared data
    int count, target;
    for (j = tid; j < nz; j += PWARP) {
        target = shared_check[soffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(shared_check[soffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[soffset + j];
        d_cval[offset + count] = shared_value[soffset + j];
    }
}


template <int SH_ROW>
__global__ void calculate_value_col_bin_each(const int *d_arpt,
                                             const int *d_acol,
                                             const real *d_aval,
                                             const int* __restrict__ d_brpt,
                                             const int* __restrict__ d_bcol,
                                             const real* __restrict__ d_bval, BVALCOLARG
                                             int *d_crpt,
                                             int *d_ccol,
                                             real *d_cval,
                                             const int *d_row_perm,
                                             int *d_nz,
                                             int bin_offset,
                                             int bin_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / WARP;
    int tid = i % WARP;
    int wid = rid % (blockDim.x / WARP);
    int j;
    __shared__ int shared_check[B_SH_SIZE];
    __shared__ real shared_value[B_SH_SIZE];
  
    int soffset = wid * SH_ROW;

    for (j = tid; j < SH_ROW; j += WARP) {
        shared_check[soffset + j] = -1;
        shared_value[soffset + j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }
    rid = d_row_perm[rid + bin_offset];

    if (tid == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();
    int lacol, acol;
    int k, l;
    int bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    real laval, aval, bval;

    lacol = 0;
    int *brow, *bacol, rowoff;
    real *baval;
    for (j = d_arpt[rid]; j < d_arpt[rid + 1]; j += WARP) {
        if (j + tid < d_arpt[rid + 1]) {
            lacol = ld_gbl_int32(d_acol + j + tid);
            laval = ld_gbl_real(d_aval + j + tid);
        }
        for (l = 0; l < WARP && j + l < d_arpt[rid + 1]; l++) {
            acol = __shfl_sync(SHFLMASK,lacol, l);
            aval = __shfl_sync(SHFLMASK,laval, l);
	    SELVALROW(acol);
	    for (k = brow[acol-rowoff]+tid; k < brow[acol-rowoff + 1]; k+=WARP) {
		bcol=bacol[k];
	        bval=baval[k];

                key = bcol;
                hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
                adr = soffset + hash;
	        if (shared_check[adr] != key) {
                                              //@@@@@ IL PROBLEMA E SE DUE Jcol UGUALI
                                              //@@@@@ SUPERANO ASSIEME IL CONTROLLO PRECEDENTE
                                              //@@@@@ E UNO SCRIVE IL SUO VALORE PRIMA CHE L'ALTRO
                                              //@@@@@ FACCIA IL CONTROLLO
                  // Add the key
                  while (1){
                     old = atomicCAS(shared_check + adr, -1, key);
                     if (old == -1 || old == key) {
                        atomic_fadd(shared_value + adr, aval * bval);
                        break;
                     } else {
                        hash = (hash + 1) & (SH_ROW - 1);
                        adr = soffset + hash;
                     }
                  }
           	} else {
                    atomic_fadd(shared_value + adr, aval * bval);
           	}
            }
        }
    }
    int col; 
    real val;
    for (j = tid; j < SH_ROW; j += WARP) {
	col=shared_check[soffset + j];
	val=shared_value[soffset + j];
//	__syncwarp();
        if (col != -1) {
            index = atomicAdd(d_nz + rid, 1);
            shared_check[soffset + index] = col;
            shared_value[soffset + index] = val;
        }
    }
    int nz = d_nz[rid];
    __syncthreads(); 
    /* Sorting for shared data */
    int count, target;
    for (j = tid; j < nz; j += WARP) {
        target = shared_check[soffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(shared_check[soffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[soffset + j];
        d_cval[offset + count] = shared_value[soffset + j];
    }
}

template <int SH_ROW>
__global__ void calculate_value_col_bin_each_tb(const int *d_arpt,
                                                const int *d_acol,
                                                const real *d_aval,
                                                const int* __restrict__ d_brpt,
                                                const int* __restrict__ d_bcol,
                                                const real* __restrict__ d_bval, BVALCOLARG
                                                int *d_crpt,
                                                int *d_ccol,
                                                real *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int bin_offset,
                                                int bin_size)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j;
    __shared__ int shared_check[SH_ROW];
    __shared__ real shared_value[SH_ROW];
  
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        shared_check[j] = -1;
        shared_value[j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }

    rid = d_row_perm[rid + bin_offset];

    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    int acol;
    int k;
    int bcol, hash, key;
    int offset = d_crpt[rid];
    int old, index;
    real aval, bval;

    int *brow, *bacol, rowoff;
    real *baval;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        aval = ld_gbl_real(d_aval + j);
        SELVALROW(acol);
        for (k = brow[acol-rowoff]+tid; k < brow[acol-rowoff + 1]; k+=WARP) {
	    bcol=bacol[k];
	    bval=baval[k];
	
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
	    if (shared_check[hash] != key) {
                                              //@@@@@ IL PROBLEMA E SE DUE Jcol UGUALI
                                              //@@@@@ SUPERANO ASSIEME IL CONTROLLO PRECEDENTE
                                              //@@@@@ E UNO SCRIVE IL SUO VALORE PRIMA CHE L'ALTRO
                                              //@@@@@ FACCIA IL CONTROLLO
                  // Add the key
                  while (1){
                     old = atomicCAS(shared_check + hash, -1, key);
                     if (old == -1 || old == key) {
                        atomic_fadd(shared_value + hash, aval * bval);
                        break;
                     } else {
                        //@@@@@ CONTROLLO AGGIUNTO PER EVITARE DUPLICAZIONE
                        hash = (hash + 1) & (SH_ROW - 1);
                     }
                  }
            } else {
                    atomic_fadd(shared_value + hash, aval * bval);
            }
        }
    }

    __syncthreads();
    int col;
    real val;
    if (threadIdx.x < WARP) {
        for (j = tid; j < SH_ROW; j += WARP) {
	    col=shared_check[j];
	    val=shared_value[j];
//	    __syncwarp();
            if (col != -1) {
                index = atomicAdd(d_nz + rid, 1);
                shared_check[index] = col;
                shared_value[index] = val;
            }
        }
    }
    __syncthreads();
    int nz = d_nz[rid];
    /* Sorting for shared data */
    int count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x) {
        target = shared_check[j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(shared_check[k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[j];
        d_cval[offset + count] = shared_value[j];
    }

}

__global__ void calculate_value_col_bin_each_gl(const int *d_arpt,
                                                const int *d_acol,
                                                const real *d_aval,
                                                const int* __restrict__ d_brpt,
                                                const int* __restrict__ d_bcol,
                                                const real* __restrict__ d_bval, BVALCOLARG
                                                int *d_crpt,
                                                int *d_ccol,
                                                real *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int *d_check,
                                                real *d_value,
                                                int max_row_nz,
                                                int bin_offset,
                                                int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    int wnum = blockDim.x / WARP;
    int j;
  
    if (rid >= M) {
        return;
    }

    int doffset = rid * max_row_nz;

    rid = d_row_perm[rid + bin_offset];
  
    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    int acol;
    int k;
    int bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    real aval, bval;

    int *brow, *bacol, rowoff;
    real *baval;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        acol = ld_gbl_int32(d_acol + j);
        aval = ld_gbl_real(d_aval + j);
        SELVALROW(acol);
        for (k = brow[acol-rowoff]+tid; k < brow[acol-rowoff + 1]; k+=WARP) {
	    bcol=bacol[k];
	    bval=baval[k];
      
            key = bcol;
            hash = (bcol * HASH_SCAL) % max_row_nz;
            adr = doffset + hash;
	    if (d_check[adr] != key) {
                                              //@@@@@ IL PROBLEMA E SE DUE Jcol UGUALI
                                              //@@@@@ SUPERANO ASSIEME IL CONTROLLO PRECEDENTE
                                              //@@@@@ E UNO SCRIVE IL SUO VALORE PRIMA CHE L'ALTRO
                                              //@@@@@ FACCIA IL CONTROLLO
                  // Add the key
                  while (1){
                     old = atomicCAS(d_check + adr, -1, key);
                     if (old == -1 || old == key ) {
                        atomic_fadd(d_value + adr, aval * bval);
                        break;
                     } else {
                        hash = (hash + 1) % max_row_nz;
                        adr = doffset + hash;
                     }
                  }
            } else {
                    atomic_fadd(d_value + adr, aval * bval);
            }
        }
    }
  
    __syncthreads();
    int max_row_nz_mwarp=((max_row_nz+(WARP-1))/WARP)*WARP;
    int loc_check;
    real loc_value;
    unsigned int maskpsum;
    if (threadIdx.x < WARP) {
#if 0
        for (j = tid; j < max_row_nz; j += WARP) {
            if (d_check[doffset + j] != -1) {
                index = atomicAdd(d_nz + rid, 1);
                d_check[doffset + index] = d_check[doffset + j];
                d_value[doffset + index] = d_value[doffset + j];
            }
        }
#else
        for (j = tid; j < max_row_nz_mwarp; j += WARP) {
	    if(j<max_row_nz) {
	       index=d_nz[rid];
	       loc_check=d_check[doffset + j];
	       loc_value=d_value[doffset + j];
	    }
	    maskpsum=__ballot_sync(SHFLMASK,((j<max_row_nz)&&d_check[doffset + j] != -1));
	    if(threadIdx.x==(WARP-1)) {
  	       atomicAdd(d_nz+rid,__popc(maskpsum));
	    }
            maskpsum&=((1<<threadIdx.x)-1);
            if ((j<max_row_nz)&&d_check[doffset + j] != -1) {
	        index += __popc(maskpsum);
                d_check[doffset + index] = loc_check;
                d_value[doffset + index] = loc_value;
            }
	    __syncwarp();
        }
#endif
    }
    __syncthreads();
    int nz = d_nz[rid];
  
    /* Sorting for shared data */
    int count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x) {
        target = d_check[doffset + j];
        count = 0;
        for (k = 0; k < nz; k++) {
            count += (unsigned int)(d_check[doffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = d_check[doffset + j];
        d_cval[offset + count] = d_value[doffset + j];
    }

}

void calculate_value_col_bin(int *d_arpt, int *d_acol, real *d_aval,
                             int *d_brpt, int *d_bcol, real *d_bval, csrlocinfo *binfo,
                             int *d_crpt, int *d_ccol, real *d_cval,
                             sfBIN *bin,
                             int M);
  
void spgemm_csrseg_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c, csrlocinfo *binfo)
{
  
    int M;
    sfBIN bin;
  
    M = a->M;
    c->M = M;
    c->N = b->N;
  
    /* Initialize bin */
    init_bin(&bin, M);

    /* Set max bin */
    set_max_bin(a->d_rpt, a->d_col, b->d_rpt, binfo,  &bin, M);
  
    checkCudaErrors(cudaMalloc((void **)&(c->d_rpt), sizeof(int) * (M + 1)));

    /* Count nz of C */
    set_row_nnz(a->d_rpt, a->d_col,
                b->d_rpt, b->d_col, binfo,
                c->d_rpt,
                &bin,
                M,
                &(c->nnz));

    /* Set bin */
    set_min_bin(&bin, M);
  
    checkCudaErrors(cudaMalloc((void **)&(c->d_col), sizeof(int) * c->nnz));
    checkCudaErrors(cudaMalloc((void **)&(c->d_val), sizeof(real) * c->nnz));
  
    /* Calculating value of C */
    calculate_value_col_bin(a->d_rpt, a->d_col, a->d_val,
                            b->d_rpt, b->d_col, b->d_val, binfo,
                            c->d_rpt, c->d_col, c->d_val,
                            &bin,
                            M);

    release_bin(bin);
}


void set_row_nnz(int *d_arpt, int *d_acol,
                 int *d_brpt, int *d_bcol, csrlocinfo *binfo,
                 int *d_crpt,
                 sfBIN *bin,
                 int M,
                 int *nnz)
{
                                                int i;
    int GS, BS;
    
    for (i = BIN_NUM - 1; i >= 0; i--) {
        if (bin->bin_size[i] > 0) {
            switch (i) {
            case 0:
                BS = 256;
                GS = div_round_up(bin->bin_size[i] * PWARP, BS);
                set_row_nz_bin_pwarp<<<GS, BS, 0, bin->stream[i]>>>
                    (d_arpt, d_acol,
                    d_brpt, d_bcol, DBINFO,
                                         bin->d_row_perm,
                    bin->d_row_nz,
                    bin->bin_offset[i],
                    bin->bin_size[i]);
                break;
            case 1 :
                             	BS = 64;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<512><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol, DBINFO,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 2 :
                             	BS = 128;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<1024><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol, DBINFO,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 3 :
                             	BS = 256;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<2048><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol, DBINFO,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 4 :
                             	BS = 512;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<4096><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol, DBINFO,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 5 :
                             	BS = 1024;
				            	GS = bin->bin_size[i];
            	set_row_nz_bin_each_tb<8192><<<GS, BS, 0, bin->stream[i]>>>
            	  (d_arpt, d_acol, d_brpt, d_bcol, DBINFO,
            	   bin->d_row_perm, bin->d_row_nz,
            	   bin->bin_offset[i], bin->bin_size[i]);
            	break;
            case 6 :
                                     	{
            	    int fail_count;
            	    int *d_fail_count, *d_fail_perm;
            	    fail_count = 0;
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_count, sizeof(int)));
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_perm, sizeof(int) * bin->bin_size[i]));
            	    cudaMemcpy(d_fail_count, &fail_count, sizeof(int), cudaMemcpyHostToDevice);
            	    BS = 1024;
            	    GS = bin->bin_size[i];
            	    set_row_nz_bin_each_tb_large<8192><<<GS, BS, 0, bin->stream[i]>>>
            	      (d_arpt, d_acol, d_brpt, d_bcol, DBINFO,
            	       bin->d_row_perm, bin->d_row_nz,
            	       d_fail_count, d_fail_perm,
            	       bin->bin_offset[i], bin->bin_size[i]);
            	    cudaMemcpy(&fail_count, d_fail_count, sizeof(int), cudaMemcpyDeviceToHost);
            	    if (fail_count > 0) {
              	        int max_row_nz = bin->max_intprod;
            	        size_t table_size = (size_t)max_row_nz * fail_count;
            	        int *d_check;
            	        checkCudaErrors(cudaMalloc((void **)&(d_check), sizeof(int) * table_size));
            	        BS = 1024;
            	        GS = div_round_up(table_size, BS);
            	        init_check<<<GS, BS, 0, bin->stream[i]>>>(d_check, table_size);
            	        GS = bin->bin_size[i];
	                    set_row_nz_bin_each_gl<<<GS, BS, 0, bin->stream[i]>>>
                     		  (d_arpt, d_acol, d_brpt, d_bcol, DBINFO,
		                   d_fail_perm, bin->d_row_nz, d_check,
             		                   max_row_nz, 0, fail_count);
                                  	                    cudaFree(d_check);
  	                }
	            cudaFree(d_fail_count);
	            cudaFree(d_fail_perm);
	        }
	        break;
	      default :
	          exit(0);
	      }
        }
      }
      cudaDeviceSynchronize();

    /* Set row pointer of matrix C */
    thrust::exclusive_scan(thrust::device, bin->d_row_nz, bin->d_row_nz + (M + 1), d_crpt, 0);
    cudaMemcpy(nnz, d_crpt + M, sizeof(int), cudaMemcpyDeviceToHost);
}

#define DBINFOV binfo->fr, binfo->lr, binfo->row, binfo->col, binfo->val

void calculate_value_col_bin(int *d_arpt, int *d_acol, real *d_aval,
			     int *d_brpt, int *d_bcol, real *d_bval, csrlocinfo *binfo,
			     int *d_crpt, int *d_ccol, real *d_cval,
			     sfBIN *bin,
			     int M)
{
                       int i;
  int GS, BS;
  for (i = BIN_NUM - 1; i >= 0; i--) {
    if (bin->bin_size[i] > 0) {
      switch (i) {
      case 0:
      BS = 256;
      GS = div_round_up(bin->bin_size[i] * PWARP, BS);
      calculate_value_col_bin_pwarp<<<GS, BS, 0, bin->stream[i]>>>
           (d_arpt, d_acol, d_aval,
	   d_brpt, d_bcol, d_bval, DBINFOV,
	   d_crpt, d_ccol, d_cval,
	   bin->d_row_perm, bin->d_row_nz,
	   bin->bin_offset[i], bin->bin_size[i]);
      break;
      case 1:
	  BS = 64;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<256><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval, DBINFOV,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
      case 2:
	  BS = 128;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<512><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval, DBINFOV,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
      case 3:
	  BS = 256;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<1024><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval, DBINFOV,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
      case 4:
	  BS = 512;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<2048><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval, DBINFOV,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
      case 5:
	  BS = 1024;
	  GS = bin->bin_size[i];
	  calculate_value_col_bin_each_tb<4096><<<GS, BS, 0, bin->stream[i]>>>
	    (d_arpt, d_acol, d_aval,
	     d_brpt, d_bcol, d_bval, DBINFOV,
	     d_crpt, d_ccol, d_cval,
	     bin->d_row_perm, bin->d_row_nz,
	     bin->bin_offset[i], bin->bin_size[i]);
	  break;
	case 6 :
	  {
	    int max_row_nz = bin->max_nz * 2;
	    int table_size = max_row_nz * bin->bin_size[i];
	    int *d_check;
	    real *d_value;
	    checkCudaErrors(cudaMalloc((void **)&(d_check), sizeof(int) * table_size));
	    checkCudaErrors(cudaMalloc((void **)&(d_value), sizeof(real) * table_size));
	    BS = 1024;
	    GS = div_round_up(table_size, BS);
	    init_check<<<GS, BS, 0, bin->stream[i]>>>(d_check, table_size);
	    init_value<<<GS, BS, 0, bin->stream[i]>>>(d_value, table_size);
	    GS = bin->bin_size[i];
	    calculate_value_col_bin_each_gl<<<GS, BS, 0, bin->stream[i]>>>
	      (d_arpt, d_acol, d_aval,
	       d_brpt, d_bcol, d_bval, DBINFOV,
	       d_crpt, d_ccol, d_cval,
	       bin->d_row_perm, bin->d_row_nz,
	       d_check, d_value, max_row_nz,
	       bin->bin_offset[i], bin->bin_size[i]);
	    cudaFree(d_check);
	    cudaFree(d_value);
	  }
	  break;
	default :
	  exit(0);
	}
      }
    }
  cudaDeviceSynchronize();
}