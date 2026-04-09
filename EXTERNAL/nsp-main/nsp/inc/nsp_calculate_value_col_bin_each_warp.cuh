template <int SH_ROW>
__global__ void nsp_calculate_value_col_bin_each_warp(const int *iat_A,const int *ja_A,const double *coef_A,
                                                   const int * __restrict__ iat_B, const int * __restrict__ ja_B,const double * __restrict__ coef_B,
                                                   const int *iat_C, int *ja_C, double *coef_C,
                                                   const int *row_perm, int *row_nz,
                                                   const int bin_offset,const int nrows_tb) {

   // check that we are inside the bounds
   const int mid = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
   if (mid >= nrows_tb) return;

   // retrieve thread infos
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // Local row index (with permutation if needed)
   int rid = (row_perm == nullptr) ? mid : row_perm[mid + bin_offset];

   // registers
   int jr,kr;
   int je,ke;
   int jcol_A;
   double cval_A,val;
   int key;
   int nz = 0;
   int offset;
   unsigned int count;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;
   double *value = (double*) (&check[wnum*SH_ROW]);
   check = check + wid * (SH_ROW);
   value = value + wid * (SH_ROW);

   typedef cub::WarpScan<uint8_t> WarpScanT;
   __shared__ typename WarpScanT::TempStorage temp_storage;

   // initialize hash table
   #pragma unroll
   for (jr = tid; jr < SH_ROW; jr += WARPSIZE) {
      check[jr] = -1;
      value[jr] = 0.;
   }

   // warp synchronization to ensure initialization
   __syncwarp();

   // loop over A-row coefficients
   for (je = iat_A[rid]; je < iat_A[rid + 1]; je++) {
      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);
      cval_A = load_glob(coef_A +je);

      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
         // load from global memory using the cache
         key =   ja_B[ke];
         val = coef_B[ke] * cval_A;
         hashmap_bit(check, value, key, val, SH_ROW-1);
      } // end loop over B-row coefficients
   } // end loop over A-row coefficients

   // Thread-Warp synchronization
   __syncwarp();

   // Compact the hash table using cub
   #pragma unroll
   for (jr = tid; jr < SH_ROW; jr += WARPSIZE) {
      key = check[jr];
      val = value[jr];
      uint8_t index = (key<0) ? 0:1;
      uint8_t warp_aggregate;
      WarpScanT(temp_storage).ExclusiveSum(index, index, warp_aggregate);
      if (key != -1){
         check[nz+index] = key;
         value[nz+index] = val;
      }
      nz+=warp_aggregate;
   }

   // __syncwarp();

   // Sorting hash table and store data in global memory
   offset = iat_C[rid];
   for (jr = tid; jr < nz; jr += WARPSIZE) {
      key = check[jr];
      count = 0;
      for (kr = 0; kr < nz; kr++) {
         count += (unsigned int)(check[kr] - key) >> 31;
      }
         ja_C[offset + int(count)] = key;
      coef_C[offset + int(count)] = value[jr];
   }
}
