
//----------------------------------------------------------------------------------------

// Compute C rows using TB/ROW and shared memory
template <int SH_ROW, typename IDXcol>
__global__ void nsp_calculate_value_col_bin_each_tb( const iExt   * __restrict__ iat_A,
                                                     const IDXcol * __restrict__ ja_A,
                                                     const rExt   * __restrict__ coef_A,
                                                     const iExt   * __restrict__ iat_B,
                                                     const IDXcol * __restrict__ ja_B,
                                                     const rExt   * __restrict__ coef_B,
                                                     const iExt   * __restrict__ iat_C,
                                                           IDXcol * __restrict__ ja_C,
                                                           rExt   * __restrict__ coef_C,
                                                     const iReg   * __restrict__ row_perm,
                                                     const iReg   * __restrict__ row_nz,
                                                     const iReg bin_offset,
                                                     const iReg nrows_tb ) {

   // retrieve thread infos
   const iReg tid  = threadIdx.x & (WARPSIZE - 1);
   const iReg wid  = threadIdx.x / WARPSIZE;
   const iReg wnum = blockDim.x  / WARPSIZE;

   // Local row index (with permutation if needed)
   const iReg rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];

   // registers
   iReg jr,kr;
   iExt je,ke;
   IDXcol jcol_A;
   rExt cval_A,val;
   IDXcol key;
   iReg nz;
   iReg index_0,index,index_out;
   iReg aggregate,ii;
   iExt offset;
   unsigned int count;

   // block shared memory
   extern __shared__ IDXcol sh_mem[];
   IDXcol *check = (IDXcol*) sh_mem;
   rExt *value = (rExt*) (&sh_mem[SH_ROW]);

   // initialize hash table
   #pragma unroll
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
      check[jr] = -1;
      value[jr] = 0.;
   }

   // block synchronization to ensure initialization
   __syncthreads();

   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      cval_A = load_glob(coef_A + je);

      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {

         // load from global memory using the cache
         key =   ja_B[ke];
         val = coef_B[ke] * cval_A;

         hashmap_bit(check, value, key, val, SH_ROW-1);

      } // end loop over B-row coefficients

   } // end loop over A-row coefficients

   // Thread-Block synchronization
   __syncthreads();

   // Compact the hash table using the first warp
   if (threadIdx.x < WARPSIZE) {
      nz = 0;
      #pragma unroll
      for (jr = tid; jr < SH_ROW; jr += WARPSIZE) {
         key = check[jr];
         val = value[jr];

         // read shmem with warp synchronization to prevent race with writes if index_0
         __syncwarp(MASKFULL);

         index_0 = (key<0) ? 0 : 1;
         index = index_0;
         #pragma unroll
         for (offset = 1; offset < WARPSIZE; offset <<= 1) {
           ii = __shfl_up_sync(MASKFULL,index, offset);
           if (tid >= offset) {
             index += ii;
           }
         }
         index_out = index - index_0;
         aggregate = __shfl_sync(MASKFULL,index,WARPSIZE-1);
         if (index_0) {
            check[nz+index_out] = key;
            value[nz+index_out] = val;
         }
         nz += aggregate;
      }
   }
   __syncthreads();

   // get the number of non-zeros
   nz = row_nz[rid];

   // Sorting hash table and store data in global memory
   offset = iat_C[rid];
   for (jr = threadIdx.x; jr < nz; jr += blockDim.x) {
      key = check[jr];
      count = 0;
      for (kr = 0; kr < nz; kr++) {
         count += (unsigned int)(check[kr] - key) >> 31;
      }
        ja_C[offset + iExt(count)] = key;
      coef_C[offset + iExt(count)] = value[jr];
   }

}

//----------------------------------------------------------------------------------------
