#include <iostream>
using namespace std;

#include <nsp.h>
#include <nsparse_asm.h> 

#define DEBUG 0

#if DEBUG
#define CHECK_CUDA() { \
  cudaError_t err = cudaGetLastError(); \
  if(err != cudaSuccess){ fprintf(stderr,"CUDA err at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} \
  err = cudaDeviceSynchronize(); \
  if(err != cudaSuccess){ fprintf(stderr,"CUDA sync err at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} \
}

#define CHECK_BS_GS() do { \
    cudaDeviceProp prop; \
    cudaGetDeviceProperties(&prop, 0); \
    printf("=== DBG (%s:%d) Kernel Launch Check ===\n", __FILE__, __LINE__); \
    printf("  Requested BS = %d\n", BS); \
    printf("  Requested GS = %lld\n", (long long)GS); \
    printf("  Device maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock); \
    printf("  Device maxGridSize = [%d, %d, %d]\n", \
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); \
    \
    if (BS <= 0) { \
        fprintf(stderr, "ERROR: BS <= 0 (%d)\n", BS); \
    } else if (BS > prop.maxThreadsPerBlock) { \
        fprintf(stderr, \
            "ERROR: BS (%d) exceeds maxThreadsPerBlock (%d)\n", \
            BS, prop.maxThreadsPerBlock); \
    } \
    \
    if (GS <= 0) { \
        fprintf(stderr, "ERROR: GS <= 0 (%lld)\n", (long long)GS); \
    } else if (GS > prop.maxGridSize[0]) { \
        fprintf(stderr, \
            "ERROR: GS (%lld) exceeds device maxGridDim.x (%d)\n", \
            (long long)GS, prop.maxGridSize[0]); \
    } \
    \
    /* Detect overflow from div_round_up() or PWARP multiplication */ \
    if (GS > (1LL<<40)) { \
        fprintf(stderr, \
            "WARNING: GS appears overflowed or bogus (%lld)\n", (long long)GS); \
    } \
    \
    printf("=============================================\n"); \
} while(0)

#define LOG_KERNEL(...) {\
    printf("Before invoking kernel at line %d\n", __LINE__);\
    __VA_ARGS__;\
    printf("After invoking kernel at line %d\n", __LINE__);\
}
#else
#define CHECK_CUDA()
#define CHECK_BS_GS()

#define LOG_KERNEL(...) {\
    __VA_ARGS__;\
}
#endif

// Counts how many significant bits are necessary to represent nn
inline int countBITS(int nn){

   int k = 1;
   int imax = 2;
   while (nn >= imax){
      imax *= 2;
      k++;
   }
   return k;

}

//////////////////////////////////////////////////////////////////////////////////////////


template <typename ind_type, typename ind_type2>
__device__ __forceinline__ void hashmap_symbolic_bit(ind_type &nz, ind_type *check, ind_type2 key, const int SH_ROW_1){

  int hash = (key * HASH_SCAL) & SH_ROW_1;
  if (check[hash] != key) {
     while(check[hash]!=key && check[hash]!=-1) hash = (hash + 1) & SH_ROW_1;
     if (check[hash] != key) {
        while(1){
           ind_type old = atomicCAS(check + hash, -1, key);
           if (old == -1 || old == key) {
              nz += (unsigned int)old >> 31;
              break;
           }else hash = (hash + 1) & SH_ROW_1;
        }
     }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////

template <typename ind_type, typename ind_type2>
__device__ __forceinline__ void hashmap_symbolic_mod(ind_type &nz, ind_type *check, ind_type2 key, const int SH_ROW){

  int hash = (key * HASH_SCAL) % SH_ROW;
  if (check[hash] != key) {
     while(check[hash]!=key && check[hash]!=-1) hash = (hash + 1) % SH_ROW;
     if (check[hash] != key) {
        while(1){
           ind_type old = atomicCAS(check + hash, -1, key);
           if (old == -1 || old == key) {
              nz += (unsigned int)old >> 31;
              break;
           }else hash = (hash + 1) % SH_ROW;
        }
     }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////


 template <typename ind_type, typename ind_type2, typename val_type>
__device__ __forceinline__ void hashmap_bit(ind_type *check, val_type *value, ind_type2 key, val_type val,const int SH_ROW_1){

  // int hash = (key * HASH_SCAL) & SH_ROW_1;
  // if (check[hash] != key) {
  //    while(1){
  //       ind_type old = myatomicCAS(check + hash, -1, key);
  //       if (old == -1 || old == key) break;
  //       else hash = (hash + 1) & (SH_ROW - 1);
  //    }
  // }
  // atomicAdd_block(value + hash, val);


  int hash = (key * HASH_SCAL) & SH_ROW_1;
  if (check[hash] != key) {
     while(check[hash]!=key && check[hash]!=-1) hash = (hash + 1) & SH_ROW_1;
     if (check[hash] != key) {
        while(1){
           ind_type old = atomicCAS(check + hash, -1, key);
           if (old == -1 || old == key) break;
           else hash = (hash + 1) & SH_ROW_1;
        }
     }
  }
  atomicAdd_block(value + hash, val);
}

//////////////////////////////////////////////////////////////////////////////////////////

template <typename ind_type, typename ind_type2, typename val_type>
 __device__ __forceinline__ void hashmap_mod_count(ind_type *sh_sums, ind_type *check, val_type *value, ind_type2 key, val_type val,const int SH_ROW){


   int hash = (key * HASH_SCAL) % SH_ROW;
   if (check[hash] != key) {
      while(check[hash]!=key && check[hash]!=-1) hash = (hash + 1) % SH_ROW;
      if (check[hash] != key) {
         while(1){
            ind_type old = atomicCAS(check + hash, -1, key);
            if (old == -1 || old == key) {
               if (old == -1) atomicAdd_block(sh_sums,1);
               break;
            }
            else hash = (hash + 1) % SH_ROW;
         }
      }
   }
   atomicAdd_block(value + hash, val);
}




//////////////////////////////////////////////////////////////////////////////////////////

void nsp_init_bin(sfBIN *bin, const int nrows_C) {

   // allocating streams
   bin->stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * BIN_NUM);
   for (int i = 0; i < BIN_NUM; i++) {
      cudaStreamCreate(&(bin->stream[i]));
   }

   // allocate host members
   //if ( bin->stream == NULL ) throw linsol_error("nsp_init_bin","stream");
   if ( bin->stream == NULL ){ 
        printf("ERROR: bin->stream is NULL\n");
   }

   bin->h_bin_size = (int *)malloc(sizeof(int) * BIN_NUM);
   bin->h_bin_offset = (int *)malloc(sizeof(int) * BIN_NUM);

   // allocate device members
   checkCudaErrors(cudaMalloc((void **)&(bin->d_row_nz), sizeof(int) * (nrows_C + 1)));
   checkCudaErrors(cudaMalloc((void **)&(bin->d_max), sizeof(int)));
   checkCudaErrors(cudaMalloc((void **)&(bin->d_bin_size), sizeof(int) * BIN_NUM));
   checkCudaErrors(cudaMalloc((void **)&(bin->d_bin_offset), sizeof(int) * BIN_NUM));
   checkCudaErrors(cudaMalloc((void **)&(bin->d_row_perm), sizeof(int) * nrows_C));

   #if DEBUG
   printf("d_row_perm can store %d elements\n", nrows_C);
   #endif

   // set shared memory infos
   bin->SHTB_cmp_max = SHTB / 12;
   bin->SHTB_set_max = bin->SHTB_cmp_max * 2; 
   bin->IMB_MIN = bin->SHTB_set_max / 16;
   bin->B_MIN   = bin->SHTB_cmp_max / 16;

}

//////////////////////////////////////////////////////////////////////////////////////////


void nsp_release_bin(sfBIN *bin) {
   // destroy streams
   for (int i = 0; i < BIN_NUM; i++) {
       cudaStreamDestroy(bin->stream[i]);
   }
   free(bin->stream);
   free(bin->h_bin_size);
   free(bin->h_bin_offset);
   
   // free device members
   cudaFree(bin->d_max);
   cudaFree(bin->d_row_nz);
   cudaFree(bin->d_row_perm);
   cudaFree(bin->d_bin_size);
   cudaFree(bin->d_bin_offset);
}



//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_intprod_num(int *iat_A, int *ja_A, int *iat_B,
                                    int *row_intprod, int nrows_C) {
   // retrieve row index
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= nrows_C) return;
   // initialize number of intermediate products
   int nz_per_row = 0;
   // compute number of intprod
   for (int j = iat_A[i]; j < iat_A[i+1]; j++) {
      int jcol_A = ja_A[j];
      nz_per_row += iat_B[jcol_A+1] - iat_B[jcol_A];
   }
   // store the number
   row_intprod[i] = nz_per_row;
}

//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_bin(int *row_nz, int *bin_size, int nrows_C) {

   // retrieve row index
   int rid = blockIdx.x * blockDim.x + threadIdx.x;

   if (rid >= nrows_C) return;

   // registers
   int nz_per_row = row_nz[rid];
   int loc_bin[BIN_NUM] = {0};

   if      (nz_per_row <= 64)          loc_bin[0]++;            // pwarp 
   else if (nz_per_row <= 128)         loc_bin[1]++;            // pwarp 
   else if (nz_per_row <= 256)         loc_bin[2]++;            // pwarp
   else if (nz_per_row <= 512)         loc_bin[3]++;            // pwarp
   else if (nz_per_row <= 1024)        loc_bin[4]++;            // tb
   else if (nz_per_row <= 2048)        loc_bin[5]++;            // tb
   else if (nz_per_row <= 4096)        loc_bin[6]++;            // tb
   else if (nz_per_row <= 8192)        loc_bin[7]++;            // tb
   else if (nz_per_row <= 12288)       loc_bin[8]++;          // tb
   else  loc_bin[9]++;       	       // chunk
   
   #pragma unroll
   for(int i=0;i<BIN_NUM-1;i++){
      atomicAdd(bin_size+i, loc_bin[i]);
   }

}
//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_row_perm(int *bin_size, int *bin_offset,
                                 int *max_row_nz, int *row_perm,
                                 int nrows_C) {

   // retrieve row index
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i >= nrows_C) return;

   // other registers
   int nz_per_row = max_row_nz[i];
   int dest;

   // BINNUM = 10
   if (nz_per_row <= 64){                    // pwarp 
      dest = atomicAdd(bin_size, 1);
      row_perm[bin_offset[0] + dest] = i;
   }
   else if (nz_per_row <= 128){              // pwarp 
      dest = atomicAdd(bin_size+1, 1);
      row_perm[bin_offset[1] + dest] = i;
   }
   else if (nz_per_row <= 256){              // pwarp
      dest = atomicAdd(bin_size+2, 1);
      row_perm[bin_offset[2] + dest] = i;
   }
   else if (nz_per_row <= 512){              // pwarp
      dest = atomicAdd(bin_size+3, 1);
      row_perm[bin_offset[3] + dest] = i;
   }
   else if (nz_per_row <= 1024){             // tb
      dest = atomicAdd(bin_size+4, 1);
      row_perm[bin_offset[4] + dest] = i;
   }
   else if (nz_per_row <= 2048){             // tb
      dest = atomicAdd(bin_size+5, 1);
      row_perm[bin_offset[5] + dest] = i;
   }
   else if (nz_per_row <= 4096){             // tb
      dest = atomicAdd(bin_size+6, 1);
      row_perm[bin_offset[6] + dest] = i;
   }
   else if (nz_per_row <= 8192){             // tb
      dest = atomicAdd(bin_size+7, 1);
      row_perm[bin_offset[7] + dest] = i;
   }
   else if (nz_per_row <= 12288){            // tb
      dest = atomicAdd(bin_size+8, 1);
      row_perm[bin_offset[8] + dest] = i;
   }
   else{                                     // large
      dest = atomicAdd(bin_size+9, 1);
      row_perm[bin_offset[9] + dest] = i;
   }

}

//////////////////////////////////////////////////////////////////////////////////////////


// Estimate size of C rows and set-up sfBIN
void nsp_set_max_bin( int *d_iat_A, int *d_ja_A, int *d_iat_B, sfBIN *bin, int nrows_C, int &DIRECT) {

   // set handles
   int *h_bin_offset = bin->h_bin_offset;
   int *h_bin_size   = bin->h_bin_size;
   int *d_row_nz     = bin->d_row_nz;
   int *d_bin_offset = bin->d_bin_offset;
   int *d_bin_size   = bin->d_bin_size;
   int *d_row_perm   = bin->d_row_perm;

   // initialize sfBIN structure to 0
   for (int i = 0; i < BIN_NUM; i++) {
        h_bin_size[i] = 0;
      h_bin_offset[i] = 0;
   }
   cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(int));

   // estimate size of C rows as number of intprod
   int BS = BLKSIZE_MxM;
   int GS = div_round_up(nrows_C,BS);

   nsp_set_intprod_num<<<GS,BS>>> (d_iat_A, d_ja_A, d_iat_B, d_row_nz, nrows_C);
   nsp_set_bin<<<GS,BS>>> (d_row_nz, d_bin_size, nrows_C);
   
   // copy group sizes from Device to Host
   cudaMemcpy(h_bin_size, d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);
   // if the largest bin is dominant (has > 15% of the rows) then don't permute the rows and use direct access
   int i = BIN_NUM - 1;
   while (h_bin_size[i] == 0) i--;

   if ((float)h_bin_size[i]/nrows_C > 0.15 ) { // add condition that it is not the chunk bins
      // nulify the use of other bins
      for (int j = 0; j < i; j++) h_bin_size[j] = 0;
      h_bin_size[i] = nrows_C;      // set up the largest bin to process all the rows
      d_row_perm = NULL;         // nulify the row permutation pointer
      DIRECT = 1;
   }else{
      // reset to 0 group sizes on the Device (recomputed later in set_row_perm)
      cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(int));
      // set-up host
      for (int i = 0; i < BIN_NUM - 1; i++) {
         h_bin_offset[i+1] = h_bin_offset[i] + int(h_bin_size[i]);
      }
      cudaMemcpy(d_bin_offset, h_bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
      nsp_set_row_perm<<<GS,BS>>>(d_bin_size,d_bin_offset,d_row_nz,d_row_perm,nrows_C);
   }
   // nulify the row_nz pointer to use atomic add in each_tb kernel
   if (i > 3) cudaMemset(d_row_nz, 0, nrows_C*sizeof(int));

   #if defined BENCHMARK
      for (int i = 0; i < BIN_NUM; i++) cout << h_bin_size[i] << " ";
      cout << endl;
   #endif

}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

template <const int pWARP,const int SH_ROW>
__global__ void nsp_set_row_nz_bin_mpwarp(int * __restrict__ iat_A, int * __restrict__ ja_A, 
                                         const int * __restrict__ iat_B, int * __restrict__ ja_B,
                                         int * __restrict__ row_perm, int * __restrict__ row_nz, 
                                         const int bin_offset,const int nrows) {

   // retrieve thread infos
   int mid  = (blockIdx.x * (blockDim.x / pWARP) + threadIdx.x / pWARP);
   int rid  = (row_perm == nullptr) ? mid : row_perm[mid + bin_offset];
   int tid  = threadIdx.x & (pWARP - 1);
   int wid  = threadIdx.x / pWARP;

   // registers
   int jr;
   int je,ke;
   int jcol_A;
   int key;
   int nz = 0;   // initialize number of non zeros

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*)sh_mem + wid * SH_ROW;

   // initialize hash table
   #pragma unroll 
   for (jr = tid; jr < SH_ROW; jr += pWARP) {
      check[jr] = -1;
   }

   // warp synchronization to ensure check initialization
   __syncwarp();

   if (mid < nrows) {
      // loop over A-row coefficients
      for (je = iat_A[rid]; je < iat_A[rid + 1]; je++) {
         // load from global memory without using the cache
         jcol_A = load_glob(ja_A + je);
         // loop over B-row coefficients
         for (ke = iat_B[jcol_A]+tid; ke < iat_B[jcol_A + 1]; ke+=pWARP) {
            // load from global memory using the cache
            key = ja_B[ke];
            hashmap_symbolic_bit(nz, check, key, SH_ROW-1);
         } // end loop over B-row coefficients
      } // end loop over A-row coefficients
   }
   // pwarp reduction of nz
   __syncwarp(MASKFULL);
   #pragma unroll 
   for( jr = pWARP>>1; jr>0; jr>>=1) {
      nz += __shfl_down_sync( MASKFULL, nz, jr, pWARP );
   }
   // store the final value
   if (tid == 0 && mid < nrows) row_nz[rid] = nz;
}


//////////////////////////////////////////////////////////////////////////////////////////

template <const int SH_ROW>
__global__ void nsp_set_row_nz_bin_each_tb(int * __restrict__ iat_A, int * __restrict__ ja_A, 
                                           const int * __restrict__ iat_B,const int * __restrict__ ja_B,
                                           int * __restrict__ row_perm, int * __restrict__ row_nz, const int bin_offset) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int nz = 0;
   int jcol_A;
   int key;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;

   // initialize hash table
   #pragma unroll 
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
   }

   // block synchronization to ensure check initialization
   __syncthreads();
   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {
      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
         // load from global memory using the cache
         key = ja_B[ke];
         hashmap_symbolic_bit(nz, check, key, SH_ROW-1);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients
   // warp reduction of nz
   __syncwarp(MASKFULL);
   #pragma unroll 
   for( jr = WARPSIZE>>1; jr>0; jr>>=1) {
      nz += __shfl_down_sync( MASKFULL, nz, jr, WARPSIZE );
   }
   // block reduction of nz using 1 thread for each warp
   // __syncthreads();
   if (tid == 0) atomicAdd(row_nz + rid, nz);

}



//////////////////////////////////////////////////////////////////////////////////////////

__global__ void nsp_set_row_nz_bin_each_tb_chunk(const int *iat_A,const int *ja_A,
                                                 const int * __restrict__ iat_B, const int * __restrict__ ja_B,
                                                 const int *row_perm, int *row_nz,
                                                 const int bin_offset,const int nrows_tb,
                                                 const int ncols_C,const int SH_ROW) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int ichunk;
   int nz;
   int jcol_A,jcol_B;
   int istrB,iendB;
   int key,hash,old;

   // block shared memory
   extern __shared__ int sh_mem[];
   #if defined LARGE_NCOLS
      int *check = (int*) sh_mem;
   #else
      int *check = (int*) sh_mem;
   #endif

   // initialize hash table
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
   }

   // initialize number of non zeros
   if (threadIdx.x == 0 ) row_nz[rid] = 0;

   // block synchronization to ensure check initialization
   __syncthreads();

   // start loop over B chunks
   ichunk = 0;
   istrB  = 0;
   iendB  = SH_ROW;
   while (1) {

      // initialize number of non-zeros for the chunk
      nz = 0;

      // loop over A-row coefficients
      for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

         // load from global memory without using the cache
         jcol_A = load_glob(ja_A + je);

         // loop over row terms of B
         for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {

            // load from global memory using the cache
            jcol_B = ja_B[ke];

            // check end of the chunk
            if ( jcol_B >= iendB ) goto cycle_loop_A;

            // check column index
            if ( jcol_B >= istrB ) {

               key  = jcol_B - istrB;
               hash = (key * HASH_SCAL) & (SH_ROW - 1);

               #if defined HASH_UPD
               if (check[hash] != key) {
                  while(1){
                     old = atomicCAS(check + hash, -1, key);
                     if (old == -1 || old == key) {
                        if (old == -1) nz++;
                        break; 
                     }else hash = (hash + 1) & (SH_ROW - 1);
                  }
               }
               #elif defined HASH_NSPARSE
                  // put the key inside the hash table
                  if (check[hash] != key) {
                     while (1){
                        old = atomicCAS(check + hash, -1, key);
                        if (old == -1) {
                           nz++;
                           break;
                        } else {
                           if (old != key){ 
                              hash = (hash + 1) & (SH_ROW - 1);
                           } else {
                              break;
                           }
                        }
                     }
                  }
               #else // default hash algorithm:
                  // put the key inside the hash table
                  while (1) {
                     if (check[hash] == key) {
                        break;
                     } else if (check[hash] == -1) {
                        old = atomicCAS(check + hash, -1, key);
                        if (old == -1) {
                           nz++;
                           break;
                        }
                     } else if (check[hash] != key) {
                        hash = (hash + 1) & (SH_ROW - 1);
                     }
                  }
               #endif

            } // end check column index

         } // end loop over row terms of B

         cycle_loop_A: ;

      } // end loop over A-row coefficients

      // warp reduction of nz
      // __syncwarp(MASKFULL);
      for( jr = WARPSIZE/2; jr > 0; jr /= 2) {
         nz += __shfl_down_sync( MASKFULL, nz, jr, WARPSIZE );
      }

      // block reduction of nz using 1 thread for each warp
      __syncthreads();
      if (threadIdx.x == 0) check[0] = 0;
      __syncthreads();
      if (tid == 0) atomicAdd(check, nz);
      __syncthreads();

      // store the final value
      if (threadIdx.x == 0) row_nz[rid] += check[0];
      __syncthreads();

      // check end of loop over B chunks
      if ( iendB >= ncols_C ) break;

      ichunk++;

      // update B indeces
      istrB  = iendB;
      iendB += SH_ROW;

      // initialize shared scratch
      for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
          check[jr] = -1;
      }

      // synchronize before next chunk cycle
      __syncthreads();

   } // end loop over B chunks

}

//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_row_nz_bin_each_tb_large(const int *iat_A,const int *ja_A,
                                                 const int * __restrict__ iat_B,const int * __restrict__ ja_B,
                                                 const int *row_perm, int *row_nz,
                                                 int *fail_count, int *fail_perm,
                                                 const int bin_offset, const int nrows_tb,const int SH_ROW ){

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int jcol_A;
   int key,hash,old;
   int count;
   int border; 
   int dest;

   // block shared memory
   extern __shared__ int sh_mem[];
   #if defined LARGE_NCOLS
      int *check = (int*) sh_mem;
   #else
      int *check = (int*) sh_mem;
   #endif
    __shared__ int snz[1];

   // initialize hash table
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
   }
   if (threadIdx.x == 0) snz[0] = 0;

   // block synchronization to ensure check initialization
   __syncthreads();

   // initialize registers
   count = 0;
   border = SH_ROW >> 1;

   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);

      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {

         // load from global memory using the cache
         key = ja_B[ke];

         hash = (key * HASH_SCAL) & (SH_ROW - 1);

         // put the key inside the hash table
         while (count < border && snz[0] < border) {
            
            if (check[hash] == key) {
               // key already added
               break; 
            } else if (check[hash] == -1) {
               // add the key
               old = atomicCAS(check + hash, -1, key);
               if (old == -1) {
                  atomicAdd(snz,1);
                  break;
               }
            } else if (check[hash] != key) {
               // find a free place to add the key
               hash = (hash + 1) & (SH_ROW - 1);
               count++;
            }
         }
   
         // check fail: gone outside hash
         if (count >= border || snz[0] >= border) break;

      } // end loop over B-row coefficients

      // check fail: gone outside hash
      if (count >= border || snz[0] >= border) break;

   } // end loop over A-row coefficients

   // block syncronization
   __syncthreads();

   // check compuatation fail
   if (count >= border || snz[0] >= border) {
      // store failed row index
      if (threadIdx.x == 0) {
         dest = atomicAdd(fail_count, 1);
         fail_perm[dest] = rid;
      }
   } else {
      // store row non-zeros
      if (threadIdx.x == 0) {
         row_nz[rid] = snz[0];
      }
   }

}

//////////////////////////////////////////////////////////////////////////////////////////

template <int SH_ROW>
__global__ void nsp_set_row_nz_bin_each_tb_max(int * __restrict__ iat_A, int * __restrict__ ja_A, 
                                           const int * __restrict__ iat_B,const int * __restrict__ ja_B,
                                           int * __restrict__ row_perm, int * __restrict__ row_nz, const int bin_offset) {

   // retrieve thread infos
   int rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int nz = 0;
   int jcol_A;
   int key;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;

   // initialize hash table
   #pragma unroll 
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
   }

   // block synchronization to ensure check initialization
   __syncthreads();

   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {
      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
         // load from global memory using the cache
         key = ja_B[ke];
         hashmap_symbolic_mod(nz, check, key, SH_ROW-1);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients

   // warp reduction of nz
   __syncwarp(MASKFULL);
   #pragma unroll 
   for( jr = WARPSIZE>>1; jr>0; jr>>=1) {
      nz += __shfl_down_sync( MASKFULL, nz, jr, WARPSIZE );
   }
   // block reduction of nz using 1 thread for each warp
   // __syncthreads();
   if (tid == 0) atomicAdd(row_nz + rid, nz);
}


//////////////////////////////////////////////////////////////////////////////////////////


void nsp_set_row_nnz( int *d_iat_A, int *d_ja_A, int *d_iat_B, int *d_ja_B, int *d_iat_C,
                      sfBIN *bin, int nrows_C, int ncols_C, int *nterm_C, int DIRECT) {

   // set handles
   int *h_bin_offset   = bin->h_bin_offset;
   int *h_bin_size     = bin->h_bin_size;
   int *d_row_perm     = (DIRECT) ? nullptr : bin->d_row_perm;
   int *d_row_nz       = bin->d_row_nz;

   // define varibles for GPU resources
   size_t shmemsize;
   int GS,BS,SH;

   // cub exclusive scan
   void     *d_temp_storage = NULL;
   size_t   temp_storage_bytes = 0;
   cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_row_nz, d_iat_C, nrows_C+1);
   cudaMalloc(&d_temp_storage, temp_storage_bytes);

   // loop over groups
   for (int i = BIN_NUM - 1; i >= 0; i--) {
      // check sizes
      if (h_bin_size[i] > 0) {
         // select group kernel
         switch (i) {
            case 0:  // <= 64
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = 64 * BS / 8;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_mpwarp<8,64><<<h_bin_size[i]/(BS/8)+1, BS, shmemsize, bin->stream[i]>>>
                                       (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                        d_row_perm,d_row_nz,
                                        h_bin_offset[i],h_bin_size[i]);
               break;
            case 1:  // <= 128
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = 128 * BS / 16;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_mpwarp<16,128><<<h_bin_size[i]/(BS/16)+1, BS, shmemsize, bin->stream[i]>>>
                                       (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                        d_row_perm,d_row_nz,
                                        h_bin_offset[i],h_bin_size[i]);
               break;
            case 2 : // <= 256
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = 256 * BS / 32;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_mpwarp<32,256><<<h_bin_size[i]/(BS/32)+1, BS, shmemsize, bin->stream[i]>>>
                                       (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                        d_row_perm,d_row_nz,
                                        h_bin_offset[i],h_bin_size[i]);
               break;
            case 3 : // <= 512
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               GS = h_bin_size[i];
               SH = 512;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<512><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;
            case 4 : // <= 1024
               BS = 128;
               GS = h_bin_size[i];
               SH = 1024;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<1024><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;
            case 5 : // <= 2048
               BS = 256;
               GS = h_bin_size[i];
               SH = 2048;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<2048><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,h_bin_offset[i]);
               break;
            case 6 : // <= 4096
               BS = 512;
               GS = h_bin_size[i];
               SH = 4096;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<4096><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;
            case 7 : // <= 8192
               #if CC == 86
                  BS = 768;
               #else
                  BS = 1024;
               #endif
               GS = h_bin_size[i];
               SH = 8192;
               shmemsize = SH * sizeof(int);
               nsp_set_row_nz_bin_each_tb<8192><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;
            case 8 : // <= 12288
               #if CC == 86
                  BS = 768;
               #else
                  BS = 1024;
               #endif
               GS = h_bin_size[i];
               SH = 12288;
               shmemsize = SH * sizeof(int);
	       nsp_set_row_nz_bin_each_tb_max<12288><<<GS, BS, shmemsize, bin->stream[i]>>>
                                         (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                          d_row_perm,d_row_nz,
                                          h_bin_offset[i]);
               break;

            case 9 :
               // start scope case 9 (to allocate inside a switch)
               {
                  // prepare auxiliary variables for large rows
                  int  h_fail_count = 0;
                  int *d_fail_count = NULL;
            	  checkCudaErrors(cudaMalloc((void **)&d_fail_count, sizeof(int)));
                  //cudaError_t cudaError = cudaMalloc((void **)&d_fail_count, sizeof(int));
                  //CheckCudaError("nsp_set_row_nnz","allocating d_fail_count",cudaError);

                  cudaMemcpy(d_fail_count, &h_fail_count, sizeof(int), cudaMemcpyHostToDevice);

                  int* d_vec_fail_perm;
				  checkCudaErrors(cudaMalloc((void **)&d_vec_fail_perm, sizeof(int) * h_bin_size[i]));

                  int *d_fail_perm = d_vec_fail_perm;
				  
                  // set GPU resources
                  #if CC == 86
                     BS = 768;
                  #else
                     BS = 1024;
                  #endif
                  GS = h_bin_size[i];
                  SH = bin->SHTB_set_max;
                  #if defined LARGE_NCOLS
                     shmemsize = SH * sizeof(int);
                  #else
                     shmemsize = SH * sizeof(int);
                  #endif
                  // try to compute all the rows using the standard kernel
                  nsp_set_row_nz_bin_each_tb_large<<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                                   d_row_perm,d_row_nz,
                                                   d_fail_count,d_fail_perm,
                                                   h_bin_offset[i],h_bin_size[i],SH);
 
                  // check if the computation failed for some rows
                  cudaMemcpy(&h_fail_count, d_fail_count, sizeof(int), cudaMemcpyDeviceToHost);
                  if (h_fail_count > 0) {
                     // compute rows using the chunk kernel
                     GS = h_fail_count;
                     nsp_set_row_nz_bin_each_tb_chunk<<<GS, BS, shmemsize, bin->stream[i]>>>
                                                           (d_iat_A,d_ja_A,d_iat_B,d_ja_B,
                                                            d_fail_perm,d_row_nz,
                                                            0,h_fail_count,ncols_C,SH);
                  }
                  // remove auxiliary variables for large rows
                  cudaFree(d_fail_count);
                  //d_vec_fail_perm.resize(0);
				  cudaFree(d_vec_fail_perm);
               } // end scope case 9
               break;
            default :
			   printf("nsp_set_row_nnz -- kernel not implemented yet");
               //throw linsol_error ("nsp_set_row_nnz","kernel not implemented yet");          
               break;

         } // end select case
      } // end check group size
   } // end loop over groups

   // syncronize device
   cudaDeviceSynchronize();

   // Set row pointer of matrix C
   // thrust::exclusive_scan(thrust::device, d_row_nz, d_row_nz + (nrows_C + 1), d_iat_C, 0);
   cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_row_nz, d_iat_C, nrows_C + 1);

   cudaMemcpy(nterm_C, d_iat_C + nrows_C, sizeof(int), cudaMemcpyDeviceToHost);

   cudaFree(d_temp_storage);

}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


__global__ void nsp_set_bin_min(int *row_nz, int *bin_size,
                            int nrows_C, int ncols_C) {

   // retrieve row index
   int rid = blockIdx.x * blockDim.x + threadIdx.x;

   int loc_bin[BIN_NUM] = {0};

   if (rid >= nrows_C) return;

   int nz_per_row = row_nz[rid];

   float density = (float)nz_per_row / ncols_C * 100.;
   if ( (ncols_C <= MAX_SH_DENSE && density > MIN_DENSITY) || (density > MIN_DENSITY_chunk) )
                                  loc_bin[BIN_NUM-1]++;  // dense bin
   else if (nz_per_row <= 11)     loc_bin[0]++;          // pwarp 
   else if (nz_per_row <= 22)     loc_bin[1]++;          // pwarp
   else if (nz_per_row <= 44)     loc_bin[2]++;          // pwarp
   else if (nz_per_row <= 90)     loc_bin[3]++;          // warp
   else if (nz_per_row <= 180)    loc_bin[4]++;          // tb
   else if (nz_per_row <= 360)    loc_bin[5]++;          // tb
   else if (nz_per_row <= 720)    loc_bin[6]++;          // tb
   else if (nz_per_row <= 1536)   loc_bin[7]++;          // tb
   else if (nz_per_row <= 3584)   loc_bin[8]++;          // tb
   // else if (nz_per_row <= 8192)   loc_bin[9]++;          // tb
   else                           loc_bin[BIN_NUM-2]++;  // dynamic

   #pragma unroll
   for(int i=0;i<BIN_NUM;i++){
      atomicAdd(bin_size+i, loc_bin[i]);
   }
}


//////////////////////////////////////////////////////////////////////////////////////////

__global__ void nsp_set_row_perm_min(int *bin_size, int *bin_offset,
                                 int *max_row_nz, int *row_perm,
                                 int nrows_C, int ncols_C) {

   // retrieve row index
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i >= nrows_C) return;

   // other registers
   int nz_per_row = max_row_nz[i];
   int dest;

   // BINNUM = 11
   float density = (float)nz_per_row / ncols_C * 100.;
   if ( (ncols_C <= MAX_SH_DENSE && density > MIN_DENSITY) || (density > MIN_DENSITY_chunk) ){     // dense bin
      dest = atomicAdd(bin_size +  BIN_NUM-1, 1);
      row_perm[bin_offset[BIN_NUM - 1] + dest] = i;
   }
   else if (nz_per_row <= 11){                                                                     // pwarp 
      dest = atomicAdd(bin_size, 1);
      row_perm[bin_offset[0] + dest] = i;
   }
   else if (nz_per_row <= 22){                                                                     // pwarp 
      dest = atomicAdd(bin_size + 1, 1);
      row_perm[bin_offset[1] + dest] = i;
   }
   else if (nz_per_row <= 44){                                                                     // pwarp 
      dest = atomicAdd(bin_size + 2, 1);
      row_perm[bin_offset[2] + dest] = i;
   }
   else if (nz_per_row <= 90){                                                                     // warp 
      dest = atomicAdd(bin_size + 3, 1);
      row_perm[bin_offset[3] + dest] = i;
   }
   else if (nz_per_row <= 180){                                                                    // tb 
      dest = atomicAdd(bin_size + 4, 1);
      row_perm[bin_offset[4] + dest] = i;
   }
   else if (nz_per_row <= 360){                                                                    // tb
      dest = atomicAdd(bin_size + 5, 1);
      row_perm[bin_offset[5] + dest] = i;
   }
   else if (nz_per_row <= 720){                                                                    // tb
      dest = atomicAdd(bin_size + 6, 1);
      row_perm[bin_offset[6] + dest] = i;
   }
   else if (nz_per_row <= 1536){                                                                   // tb
      dest = atomicAdd(bin_size + 7, 1);
      row_perm[bin_offset[7] + dest] = i;
   }
   else if (nz_per_row <= 3584){                                                                   // tb
      dest = atomicAdd(bin_size + 8, 1);
      row_perm[bin_offset[8] + dest] = i;
   }
   // else if (nz_per_row <= 8192){                                                                   // tb
   //    dest = myatomicAdd(bin_size + 9, 1);
   //    row_perm[bin_offset[9] + dest] = i;
   // }
   else{                                                                                           // dynamic
      dest = atomicAdd(bin_size + BIN_NUM-2, 1);
      row_perm[bin_offset[BIN_NUM-2] + dest] = i;
   }

}

//////////////////////////////////////////////////////////////////////////////////////////

void nsp_set_min_bin( sfBIN *bin, int nrows_C, int ncols_C, int &DIRECT, int &ifrac, float &frac) {

   // set handles
   int *h_bin_offset = bin->h_bin_offset;
   int *h_bin_size   = bin->h_bin_size;
   int *d_row_nz     = bin->d_row_nz;
   int *d_bin_offset = bin->d_bin_offset;
   int *d_bin_size   = bin->d_bin_size;
   int *d_row_perm   = bin->d_row_perm;

   // initialize sfBIN structure to 0
   for (int i = 0; i < BIN_NUM; i++) {
      h_bin_size[i]   = 0;
      h_bin_offset[i] = 0;
   }
   cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(int));

   // Compute size of C rows
   int BS = BLKSIZE_MxM;
   int GS = div_round_up(nrows_C,BS);
   nsp_set_bin_min<<<GS,BS>>>(d_row_nz,d_bin_size,nrows_C,ncols_C);

   // copy group sizes from Device to Host
   cudaMemcpy(h_bin_size, d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);

   // if the largest bin is dominant (has > 15% of the rows) then don't permute the rows and use direct access
   int i = BIN_NUM - 1;
   while (h_bin_size[i] == 0) i--;
   frac  = (float)h_bin_size[i]/nrows_C;
   ifrac = i;
   if ((float)h_bin_size[i]/nrows_C > 0.15 ) { // add condition that it is not the chunk bins
      // nulify the use of other bins
      for (int j = 0; j < i; j++) h_bin_size[j] = 0;
      h_bin_size[i] = nrows_C;      // set up the largest bin to process all the rows
      d_row_perm = nullptr;         // nulify the row permutation pointer
      DIRECT = 1;
   }else{
      DIRECT = 0;
      // reset to 0 group sizes on the Device (recomputed later in set_row_perm)
      cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(int));
      // set-up host
      for (int i = 0; i < BIN_NUM - 1; i++) {
         h_bin_offset[i+1] = h_bin_offset[i] + int(h_bin_size[i]);
      }
      cudaMemcpy(d_bin_offset, h_bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
      nsp_set_row_perm_min<<<GS,BS>>>(d_bin_size,d_bin_offset,d_row_nz,d_row_perm,nrows_C,ncols_C);
      // sort_permutations(d_row_nz, d_row_perm, nrows_C, BIN_NUM, d_bin_offset);
      // sort_permutations(d_row_nz, d_row_perm, nrows_C, 4, d_bin_offset+3);
   }

   #if defined BENCHMARK
      for (int i = 0; i < BIN_NUM; i++) cout << h_bin_size[i] << " ";
      cout << endl;
      // FILE *fid = fopen("bin.txt","a");
      // for (int i = 0; i < BIN_NUM; i++) fprintf(fid,"%d ",h_bin_size[i]);
      // fprintf(fid,"\n");
      // fclose(fid);
   #endif

}

//////////////////////////////////////////////////////////////////////////////////////////

#include "getMaskByWarpID.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

#include "dev_compact.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

#include "nsp_calculate_value_col_bin_mpwarp.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

#include "nsp_calculate_value_col_bin_each_warp.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

#include "nsp_calculate_value_col_bin_each_tb.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

#include "nsp_calculate_value_col_bin_each_tb_outsort.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

#include "nsp_calculate_value_col_bin_each_tb_chunk_dynamic.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

void prefixSumExclusive(int *in, int nelems, int initVal) {

   thrust::device_ptr<int> d_thrust_in = thrust::device_pointer_cast(in);

	try{
		thrust::exclusive_scan(d_thrust_in, d_thrust_in + nelems+1, d_thrust_in, 0);

	}catch(std::bad_alloc &e){

		printf("Error prefixSumExclusive\n");        
		// exit(EXIT_FAILURE);
      return;

	}

}

//////////////////////////////////////////////////////////////////////////////////////////

__global__ void nsp_calc_C_dense(const int *iat_A,const int *ja_A, const double *coef_A,
                                 const int * __restrict__ iat_B, const int * __restrict__ ja_B, const double * __restrict__ coef_B,
                                 const int *iat_C, int *ja_C, double *coef_C,
                                 const int *row_perm, int *row_nz,int ncols_C,
                                 const int bin_offset, const int nrows_tb, const int SH_ROW) {

   /*
      Use bitarray "check" to count the added elements into shared table called "value".
      We use bitarray to avoid the possibility of accidental zeros in the "value" from reduction.
   */

   // retrieve thread infos
   int rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (warpSize - 1);
   int wid  = threadIdx.x / warpSize;
   int wnum = blockDim.x / warpSize;

   extern __shared__ int sh_mem[];
   unsigned char *check = (unsigned char*)sh_mem;
   double *value = (double*) &(check[SH_ROW]);

   // initialize shared table
   for (int jr = threadIdx.x; jr < ncols_C; jr += blockDim.x) {
       value[jr] = 0.;
       check[jr] = 0x00;
   }

   // block synchronization to ensure initialization
   __syncthreads();

   // loop over A-row coefficients
   for (int je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {
      // load from global memory without using the cache
      int jcol_A = load_glob(ja_A + je);
      double    cval_A = load_glob(coef_A + je);
      // loop over B-row coefficients
      for (int ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += warpSize) {
         // load from global memory using the cache
         int key =   ja_B[ke];
         double cval_B = coef_B[ke];
         // bitmask "check" and update "value"
         check[key] |= 0x01; 
         atomicAdd(value + key, cval_A * cval_B);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients

   // Thread-Block synchronization
   __syncthreads();
   // New compaction takes the dense row called "value" and bitarray "check" as input and stores the output into csr coef_C[] and ja_C[] 
   int offset = iat_C[rid];
   dev_compactVal(ncols_C,check, value, (int*)&(value[SH_ROW]), coef_C+offset, ja_C+offset, tid, wid, wnum);
}


//////////////////////////////////////////////////////////////////////////////////////////

#include "nsp_calculate_value_col_chunk_B_dense.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

__global__ void set_row_size(int *iat_A, int *d_row_perm, int nn, int bin_shift, int *d_A_col_offsets){

   int tid = threadIdx.x;  
      
   if (d_row_perm == nullptr){
      while(tid < nn){
         d_A_col_offsets[tid] = iat_A[tid+1] - iat_A[tid]; 
         tid += blockDim.x;
      }
   }
   else{
      while(tid < nn){
         int ind = d_row_perm[bin_shift + tid];
         d_A_col_offsets[tid] = iat_A[ind+1] - iat_A[ind]; 
         tid += blockDim.x;
      }
   }
}


//////////////////////////////////////////////////////////////////////////////////////////

#include "nsp_calculate_value_col_bin.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

void nsp_spgemm_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c){
    sfBIN bin;
    int nrows_C = a->M;
    int ncols_C = b->N;
    c->M = nrows_C;
    c->N = ncols_C;
  
    // Initialize bin 
	nsp_init_bin (&bin, nrows_C);

    // Set max bin 
	int DIRECT = 0; 
	nsp_set_max_bin ( a->d_rpt, a->d_col, b->d_rpt, &bin, nrows_C, DIRECT);
  
    checkCudaErrors(cudaMalloc((void **)&(c->d_rpt), sizeof(int) * (nrows_C + 1)));

    // Count nz of C
	c->nnz = 0;
	nsp_set_row_nnz( a->d_rpt, a->d_col, b->d_rpt, b->d_col, c->d_rpt,
	                 &bin, nrows_C, ncols_C, &(c->nnz), DIRECT );
		
  
    checkCudaErrors(cudaMalloc((void **)&(c->d_col), sizeof(int) * c->nnz));
    checkCudaErrors(cudaMalloc((void **)&(c->d_val), sizeof(real) * c->nnz));
    
	// Set bin
    int ifrac;
    float frac;
    // compute the exact size of C rows and update sfBIN
    nsp_set_min_bin(&bin, nrows_C, ncols_C, DIRECT, ifrac, frac);
  
  
    // Calculating value of C 
    nsp_calculate_value_col_bin(a->d_rpt, a->d_col, a->d_val,
							    b->d_rpt, b->d_col, b->d_val,
								c->d_rpt, c->d_col, c->d_val,
								&bin, nrows_C, ncols_C, c->nnz, DIRECT);

	nsp_release_bin(&bin);
}
