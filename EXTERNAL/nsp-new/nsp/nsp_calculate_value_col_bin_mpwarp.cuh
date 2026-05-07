#include "nsp/getMaskByWarpID.cuh"

//----------------------------------------------------------------------------------------

// Compute C rows using pWARP/ROW and shared memory
template <int pWARP, int SH_ROW, typename IDXcol>
__global__ void nsp_calculate_value_col_bin_mpwarp( const iExt   * __restrict__ iat_A,
                                                    const IDXcol * __restrict__ ja_A,
                                                    const rExt   * __restrict__ coef_A,
                                                    const iExt   * __restrict__ iat_B,
                                                    const IDXcol * __restrict__ ja_B,
                                                    const rExt   * __restrict__ coef_B,
                                                    const iExt   * __restrict__ iat_C,
                                                          IDXcol * __restrict__ ja_C,
                                                          rExt   * __restrict__ coef_C,
                                                    const iReg   * __restrict__ row_perm,
                                                    const iReg bin_offset,
                                                    const iReg nrows ) {

   constexpr int SHIFT = __builtin_ctz(pWARP);

   // Retrieve thread infos
   const iReg mid  = (blockIdx.x * blockDim.x + threadIdx.x) / pWARP;
   const unsigned int lane = threadIdx.x & (WARPSIZE-1);
   const unsigned int gid = lane >> SHIFT;
   const unsigned int tid  = threadIdx.x & (pWARP - 1);
   const iReg wid  = threadIdx.x / pWARP;
   const iReg wnum = blockDim.x / pWARP;

   if (mid >= nrows) return;

   // Local row index (with permutation if needed)
   const iReg rid = (row_perm == nullptr) ? mid : row_perm[mid + bin_offset];

   // registers
   iReg jr,kr;
   iExt je,ke;
   iReg nz;
   iReg index_0,index,index_out;
   iReg aggregate,ii;
   IDXcol jcol_A;
   rExt cval_A,val;
   IDXcol key;
   iExt offset;
   unsigned int count;
   unsigned int mask, last;

   // block shared memory
   extern __shared__ IDXcol sh_mem[];
   IDXcol *check = (IDXcol*) sh_mem;
   rExt *value = (rExt*) (&check[wnum*SH_ROW]);

   // Compute the mask
   mask = getMaskByWarpID(pWARP,gid);
   last = (gid<<SHIFT) + pWARP - 1;

   // initialize hash table
   check = check + wid * SH_ROW;
   value = value + wid * SH_ROW;

   #pragma unroll
   for (jr = tid; jr < SH_ROW; jr += pWARP) {
      check[jr] = -1;
      value[jr] = 0.;
   }

   // warp synchronization to ensure initialization
   __syncwarp(mask);

   // loop over A-row coefficients
   for (je = iat_A[rid]; je < iat_A[rid + 1]; je++) {

      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      cval_A = load_glob(coef_A + je);

      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke+=pWARP) {

         // load from global memory using the cache
         key = ja_B[ke];
         val = coef_B[ke] * cval_A;

         hashmap_bit(check, value, key, val, SH_ROW-1);

      } // end loop over B-row coefficients

   } // end loop over A-row coefficients

   // warp synchronization
   __syncwarp(mask);

   // Compact hash table
   nz = 0;
   for (jr = tid; jr < SH_ROW; jr += pWARP) {
      key = check[jr];
      val = value[jr];

      // read shmem with warp synchronization to prevent race with writes if index_0
      __syncwarp(mask);

      index_0 = (key<0) ? 0:1;
      index = index_0;
      #pragma unroll
      for (offset = 1; offset < pWARP; offset <<= 1) {
        ii = __shfl_up_sync(mask,index, offset);
        if (tid >= offset) {
          index += ii;
        }
      }
      index_out = index - index_0;
      aggregate = __shfl_sync(mask,index,last);
      if (index_0) {
         check[nz+index_out] = key;
         value[nz+index_out] = val;
      }
      nz += aggregate;
   }

   // write shmem (if index_0) with warp synchronization to prevent race with reads below
   __syncwarp(mask);

   // Sorting hash table and store data in global memory
   offset = iat_C[rid];
   for (jr = tid; jr < nz; jr += pWARP) {
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
