
//----------------------------------------------------------------------------------------

// Compute C rows using TB/ROW and shared memory. B and C are divided into column chunks.
template <int BS, int SH_ROW, typename IDXcol>
__global__ void nsp_calculate_value_col_chunk_B_dense( const iExt   * __restrict__ iat_A,
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
                                                       const iReg nrows_tb,
                                                       const IDXcol ncols_C,
                                                       const iExt   * __restrict__ d_A_col_offsets,
                                                             iExt   * __restrict__ d_A_col_chunks ) {

   /*
      since we parse the rows of B in chunks size of the shared table, we can use the bitmask as "check" array
      - at each iteration of loop over B terms we store the last visited index of B added into the hash table
      - in the next chunk iteraion use the offset to go directly to the non visited elements of B
      - you can use less memory: ideally, the scratch space needed is the number of blocks per SM times the
        size of the biggest rows of A
      - be careful to not exceed the GPU global memory
   */

   // retrieve thread infos
   const int tid  = threadIdx.x & (WARPSIZE - 1);
   const int wid  = threadIdx.x / WARPSIZE;
   const int wnum = BS / WARPSIZE;

   // Local row index (with permutation if needed)
   const iReg rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];

   // registers
   iReg jr;
   iExt je, ke;
   iExt offset;
   IDXcol key;
   IDXcol istrB, iendB;
   IDXcol jcol_A, jcol_B;
   rExt cval_A, cval_B;
   iExt A_row_offset;
   iReg B_col_offset;
   iExt iat_B_ind;
   unsigned max_lane;
   bool flag;
   iExt off;

   // block shared memory
   extern __shared__ IDXcol sh_mem[];
   unsigned char *check = (unsigned char*)sh_mem;
   rExt *value = (rExt*) &(check[SH_ROW]);
   iReg *sh_sums = (iReg*) &(value[SH_ROW]);
   iReg *sh_nz = sh_sums + WARPSIZE;

   // initialize shared table and bitmask
   #pragma unroll
   for (jr = threadIdx.x; jr < SH_ROW; jr += BS) {
       value[jr] = 0.;
       check[jr] = 0x00;
   }

   // block synchronization to ensure check initialization
   __syncthreads();

   // get the offset between different rows of C
   offset = iat_C[rid];

   // get the offset between different rows of A
   A_row_offset = d_A_col_offsets[blockIdx.x];

   // start loop over B chunks
   istrB  = 0;
   iendB  = SH_ROW;

   while (1) {

      // loop over A-row coefficients
      for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

         // load from global memory without using the cache
         jcol_A = load_glob(  ja_A + je);
         cval_A = load_glob(coef_A + je);

         // get the offset for a term of the row of A that correponds to the row of B
         B_col_offset = A_row_offset + je - iat_A[rid];
         iat_B_ind = iat_B[jcol_A];

         // init flag to identify max non-exited lane
         flag = false;

         // loop over row terms of B
         for (ke = d_A_col_chunks[B_col_offset] + iat_B_ind + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {

            // load from global memory using the cache
            jcol_B = ja_B[ke];

            // check end of the chunk
            if ( jcol_B >= iendB ) break;
            else{
               key = jcol_B - istrB;
               cval_B = coef_B[ke];

               // add the keys to bitmask and update the values in shared table
               check[key] = 0x01;
               myatomicAdd(value + key, cval_A * cval_B);

               // update thread's B column offsets (max non-exited lane will do the update)
               max_lane = __activemask();
               max_lane = 31 - __clz(max_lane);
               off = ke - iat_B_ind + 1;
               flag = (tid == max_lane) ? true : false;

            } // end check column index

         } // end loop over row terms of B

         // update B column offsets (max non-exited lane will do the update)
         if (flag) myatomicMax(d_A_col_chunks + B_col_offset, off);

      } // end loop over A-row coefficients

      // thread-block synchronization
      __syncthreads();

      // compact the shared table
      dev_compactVal_chunk_dense(SH_ROW, tid, wid, wnum, istrB, check, value,
                                 coef_C + offset, ja_C + offset, sh_nz, sh_sums);

      // thread-block synchronization
      __syncthreads();

      // update the offset by adding the number of nonzeros of the previous chunk
      offset += *sh_nz;

      // check end of loop over B chunks
      if ( iendB >= ncols_C ) break;

      // update B indices
      istrB  = iendB;
      iendB += SH_ROW;

      // initialize the shared table and the bitmask
      #pragma unroll
      for (iReg jr = threadIdx.x; jr < SH_ROW; jr += BS) {
         value[jr] = 0.;
         check[jr] = 0x00;
      }

      // synchronize before the next chunk cycle
      __syncthreads();

   } // end loop over B chunks

}

//----------------------------------------------------------------------------------------
