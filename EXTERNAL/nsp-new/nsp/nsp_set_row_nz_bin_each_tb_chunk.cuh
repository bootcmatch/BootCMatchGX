
//----------------------------------------------------------------------------------------

// Compute the number of non-zero terms in C rows using TB/ROW and shared memory.
// B and C are devided into column chunks.
template <typename IDXcol>
__global__ void nsp_set_row_nz_bin_each_tb_chunk( const iExt   * __restrict__ iat_A,
                                                  const IDXcol * __restrict__ ja_A,
                                                  const iExt   * __restrict__ iat_B,
                                                  const IDXcol * __restrict__ ja_B,
                                                  const iReg   * __restrict__ row_perm,
                                                        iReg   * __restrict__ row_nz,
                                                  const iReg   bin_offset,
                                                  const iReg   nrows_tb,
                                                  const IDXcol ncols_C,
                                                  const iReg   SH_ROW) {

   // retrieve thread infos
   const iReg tid  = threadIdx.x & (WARPSIZE - 1);
   const iReg wid  = threadIdx.x / WARPSIZE;
   const iReg wnum = blockDim.x / WARPSIZE;

   // Local row index (with permutation if needed)
   const iReg rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];

   // registers
   iReg jr;
   iExt je,ke;
   iReg ichunk;
   iReg nz;
   IDXcol jcol_A,jcol_B;
   IDXcol istrB,iendB;
   IDXcol key,hash,old;

   // block shared memory
   extern __shared__ IDXcol sh_mem[];
   IDXcol *check = (IDXcol*) sh_mem;

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
               hash = (key * HASH_SCAL) % SH_ROW;

               if (check[hash] != key) {
                  while(1){
                     old = myatomicCAS(check + hash, -1, key);
                     if (old == -1 || old == key) {
                        if (old == -1) nz++;
                        break; 
                     } else hash = (hash + 1) % SH_ROW;
                  }
               }

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
      if (tid == 0) myatomicAdd(check, nz);
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

//----------------------------------------------------------------------------------------
