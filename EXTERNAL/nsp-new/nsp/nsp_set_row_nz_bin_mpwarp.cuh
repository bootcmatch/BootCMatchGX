
//----------------------------------------------------------------------------------------

// Compute the number of non-zero terms in C rows using pWARP/ROW and shared memory
template <int pWARP, int SH_ROW, typename IDXcol>
__global__ void nsp_set_row_nz_bin_mpwarp( const iExt   * __restrict__ iat_A,
                                           const IDXcol * __restrict__ ja_A,
                                           const iExt   * __restrict__ iat_B,
                                           const IDXcol * __restrict__ ja_B,
                                           const iReg   * __restrict__ row_perm,
                                                 iReg   * __restrict__ row_nz,
                                           const iReg bin_offset,
                                           const iReg nrows ) {

   // retrieve thread infos
   const iReg mid = (blockIdx.x * (blockDim.x / pWARP) + threadIdx.x / pWARP);
   const iReg tid = threadIdx.x & (pWARP - 1);
   const iReg wid = threadIdx.x / pWARP;

   // Local row index (with permutation if needed)
   iReg rid;
   if (mid < nrows) {
      rid = (row_perm == nullptr) ? mid : row_perm[mid + bin_offset];
   }

   // Registers
   iReg jr;
   iExt je,ke;
   IDXcol jcol_A;
   IDXcol key;
   iReg nz = 0;   // initialize number of non zeros
   const int SH_ROW_1 = SH_ROW - 1;

   // Block shared memory
   extern __shared__ IDXcol sh_mem[];
   IDXcol *check = (IDXcol*)sh_mem + wid * SH_ROW;

   // Initialize hash table
   #pragma unroll 
   for (jr = tid; jr < SH_ROW; jr += pWARP) {
      check[jr] = -1;
   }

   // Warp synchronization to ensure check initialization
   __syncwarp();

   if (mid < nrows) {

      // Loop over A-row coefficients
      for (je = iat_A[rid]; je < iat_A[rid + 1]; je++) {

         // Load from global memory without using the cache
         jcol_A = load_glob(ja_A + je);

         // Loop over B-row coefficients
         for (ke = iat_B[jcol_A]+tid; ke < iat_B[jcol_A + 1]; ke+=pWARP) {

            // Load from global memory using the cache
            key = ja_B[ke];

            // Insert nz in the hash
            hashmap_symbolic_bit(nz, check, key, SH_ROW_1);

         } // End loop over B-row coefficients

      } // End loop over A-row coefficients

   }

   // pWARP reduction of nz
   __syncwarp();
   #pragma unroll 
   for( jr = pWARP>>1; jr>0; jr>>=1) {
      nz += __shfl_down_sync( MASKFULL, nz, jr, pWARP );
   }
 
   // Store the final value
   if (tid == 0 && mid < nrows) row_nz[rid] = nz;

}

//----------------------------------------------------------------------------------------
