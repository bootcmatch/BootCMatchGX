
//----------------------------------------------------------------------------------------

// Compute the number of non-zero terms in C rows using TB/ROW and shared memory
template <int SH_ROW, typename IDXcol>
__global__ void nsp_set_row_nz_bin_each_tb_max( const iExt   * __restrict__ iat_A,
                                                const IDXcol * __restrict__ ja_A,
                                                const iExt   * __restrict__ iat_B,
                                                const IDXcol * __restrict__ ja_B,
                                                const iReg   * __restrict__ row_perm,
                                                      iReg   * __restrict__ row_nz,
                                                const iReg bin_offset ) {

   // retrieve thread infos
   const iReg tid  = threadIdx.x & (WARPSIZE - 1);
   const iReg wid  = threadIdx.x / WARPSIZE;
   const iReg wnum = blockDim.x / WARPSIZE;

   // Local row index (with permutation if needed)
   const iReg rid  = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];

   // registers
   iReg jr;
   iExt je,ke;
   iReg nz = 0;
   IDXcol jcol_A;
   IDXcol key;

   // block shared memory
   extern __shared__ IDXcol sh_mem[];
   IDXcol *check = (IDXcol*) sh_mem;

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
   if (tid == 0) myatomicAdd(row_nz + rid, nz);

}

//----------------------------------------------------------------------------------------
