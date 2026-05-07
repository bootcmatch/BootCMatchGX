
//----------------------------------------------------------------------------------------

// Compute dense C mat using global memory
template <typename IDXcol>
__global__ void nsp_calc_C_dense( const iExt   * __restrict__ iat_A,
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
                                  const IDXcol ncols_C,
                                  const iReg   bin_offset,
                                  const iReg   nrows_tb,
                                  const iReg   SH_ROW ) {

   /*
      Use bitarray "check" to count the added elements into shared table called "value".
      We use bitarray to avoid the possibility of accidental zeros in the "value" from reduction.
   */

   // retrieve thread infos
   int tid  = threadIdx.x & (warpSize - 1);
   int wid  = threadIdx.x / warpSize;
   int wnum = blockDim.x / warpSize;

   // Local row index (with permutation if needed)
   iReg rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];

   // registers
   iReg jr;
   iExt je,ke;
   IDXcol jcol_A;
   rExt cval_A,cval_B;
   IDXcol key;
   iExt offset;

   // block shared memory
   extern __shared__ IDXcol sh_mem[];
   unsigned char *check = (unsigned char*)sh_mem;
   rExt *value = (rExt*) &(check[SH_ROW]);
   iReg *sh_sums = (iReg*) &(value[SH_ROW]);

   // initialize shared table
   for (jr = threadIdx.x; jr < ncols_C; jr += blockDim.x) {
       value[jr] = 0.;
       check[jr] = 0x00;
   }

   // block synchronization to ensure initialization
   __syncthreads();

   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      cval_A = load_glob(coef_A + je);

      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += warpSize) {

         // load from global memory using the cache
         key    = ja_B[ke];
         cval_B = coef_B[ke];

         // bitmask "check" and update "value"
         check[key] = 0x01;
         myatomicAdd(value + key, cval_A * cval_B);

      } // end loop over B-row coefficients

   } // end loop over A-row coefficients

   // Thread-Block synchronization
   __syncthreads();

   // New compaction takes the dense row called "value" and bitarray "check" as input and stores the
   // output into csr coef_C[] and ja_C[]
   offset = iat_C[rid];
   dev_compactVal(ncols_C, tid, wid, wnum, check, value, coef_C+offset, ja_C+offset, sh_sums);

}

//----------------------------------------------------------------------------------------
