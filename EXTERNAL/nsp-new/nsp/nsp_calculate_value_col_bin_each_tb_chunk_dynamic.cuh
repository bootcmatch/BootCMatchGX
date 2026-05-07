#define mymax(a,b) ((a)>(b)?(a):(b))

//----------------------------------------------------------------------------------------

// Compute C rows using TB/ROW and shared memory. B and C are devided into column chunks.
// Rows of B (and A) are partitioned into chunks to manage very dense rows that do not fit
// into shared memory.
// SH_ROW is the size of the hash table. For efficiency the SH_ROW should be a multiple of BS.
// BS is the block size (so that no threads idle in jr-loops).
// OVERFILL is used to determine when the hash table becomes full and must be at least BS
// less the hash table size. This is done to avoid a barrier inside the loop over A rows.
// The -(WARPSIZE+1) accounts for sh_sums and sh_nz, and -1 ensures the hash will never get full.
template <int SH_ROW, int BS, typename IDXcol, int OVERFILL = SH_ROW - BS - (WARPSIZE+1) -1>
__global__ void nsp_calculate_value_col_bin_each_tb_chunk_dynamic( const iExt   * __restrict__ iat_A,
                                                                   const IDXcol * __restrict__ ja_A,
                                                                   const rExt   * __restrict__ coef_A,
                                                                   const iExt   * __restrict__ iat_B,
                                                                   const IDXcol * __restrict__ ja_B,
                                                                   const rExt   * __restrict__ coef_B,
                                                                   const iExt   * __restrict__ iat_C,
                                                                         IDXcol * __restrict__ ja_C,
                                                                         rExt   * __restrict__ coef_C,
                                                                   const iReg   * __restrict__ row_perm,
                                                                   const iReg   bin_offset,
                                                                   const iReg   nrows_tb,
                                                                   const IDXcol ncols_C ) {

   /*
      - this kernel is aimed at sparse and big size rows. The idea is to use a smaller number of chunks
        to process a row of C. If the row is dense then this kernel is not efficient, instead use
        nsp_calculate_value_col_chunk_B_dense.cuh. The density to use this kernel is set to 30%. However,
        it should be configured more precisely.

      - while hashing we count the number of elements added to the hash table and we initialize the
        column_length (maximal number of B columns traversed). When the hash table becomes overfull,
        we decrease column_length.
   */

   // retrieve thread infos
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // Local row index (with permutation if needed)
   iReg rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];

   // registers
   iReg jr;
   iExt je,ke;
   IDXcol jcol_A,jcol_B;
   rExt cval_A,cval_B;
   IDXcol istrB,iendB;
   IDXcol key;
   iExt offset;
   iReg nz;
   IDXcol column_length;
   int ichunk;
   unsigned int count;

   // block shared memory
   extern __shared__ IDXcol sh_mem[];
   IDXcol *check = (IDXcol*) sh_mem;
   rExt *value   = (rExt*) (&sh_mem[SH_ROW]);
   iReg *sh_sums = (iReg*) &(value[SH_ROW]);
   iReg *sh_nz   = sh_sums + WARPSIZE;

   // initialize hash table
   #pragma unroll
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
       value[jr] = 0.;
   }

   // initialize number of non zeros
   if (threadIdx.x == 0 ) {
      *sh_nz = 0;
   }

   // block synchronization to ensure check initialization
   __syncthreads();

   // get term offset
   offset = iat_C[rid];

   // start loop over B chunks
   ichunk = 0;
   istrB  = 0;

   //                   (                min number of chunks                    )
   column_length  = mymax(ncols_C / ( ((iat_C[rid+1] - offset - 1) / SH_ROW ) + 1 ) / 2 ,SH_ROW) ;

   iendB = column_length;

   while (1) {

      // loop over A-row coefficients
      for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

         // load from global memory without using the cache
         jcol_A = load_glob(ja_A + je);
         cval_A = load_glob(coef_A + je);

         // get the offset for a term of the row of A that correponds to the row of B

         // loop over row terms of B
         for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
            // load from global memory using the cache
            jcol_B = ja_B[ke];

            // check end of the chunk
            if (jcol_B >= iendB ) break;
            if (jcol_B >= istrB){

               cval_B = coef_B[ke];
               key  = jcol_B - istrB;

               hashmap_mod_count(sh_nz, check, value, key, cval_A * cval_B, SH_ROW);

            } // end check column index

            if (*sh_nz > OVERFILL && column_length > SH_ROW) break;

         } // end loop over row terms of B

         if (*sh_nz > OVERFILL && column_length > SH_ROW) break;

      } // end loop over A-row coefficients

      // Thread-Block synchronization
      __syncthreads();

      // check if the hash table is not overfull, else rerun with a smaller column_length
      if (*sh_nz > OVERFILL && column_length > SH_ROW){
         iendB -= column_length;
         column_length /= 2;
         column_length = mymax(SH_ROW,column_length);
         iendB += column_length;

         goto table_init;
      }

      // compact the hash table and store the number of nonzero into sh_sums
      dev_compactKeyVal_inplace(SH_ROW,istrB,sh_sums,check,value,sh_sums);
      __syncthreads();
      nz = *sh_sums;

      // Sorting for shared data and copy to global memory
      for (jr = threadIdx.x; jr < nz; jr += blockDim.x) {
         key = check[jr];
         count = 0;
         for (iReg kr = 0; kr < nz; kr++) {
            count += (unsigned int)(check[kr] - key) >> 31;
         }
           ja_C[offset + count] = key;
         coef_C[offset + count] = value[jr];
      }
      __syncthreads();

      // check end of loop over B chunks
      if ( iendB >= ncols_C ) break;

      ichunk++;

      // update B indeces
      istrB  = iendB;
      iendB += column_length;

      // update term offset
      offset += nz;

      table_init: ;

      // initialize shared scratch
      #pragma unroll
      for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
         check[jr] = -1;
         value[jr] = 0.;
      }
      if (threadIdx.x == 0) *sh_nz = 0;

      // synchronize before next chunk cycle
      __syncthreads();

   } // end loop over B chunks

}

//----------------------------------------------------------------------------------------
