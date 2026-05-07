/*----------------------------------------------------------------------------------------
// Compacts an array of nn elements stored in Key removing entries with a negative value
// The function works in place and needs (WARPSIZE+1) positions in the shared memory
//
// Variables:
//
// nn:      number of entries to compact
// nc:      number of compacted entries
// Key:     in/out set of entries to compact
// sh_sums: scratch vector (typically in shared memory) of dimension WARPSIZE+1
----------------------------------------------------------------------------------------*/

template <typename IDXcol>
static __device__ __forceinline__ void dev_compactKeyVal_inplace( const iReg nn,
                                                                  const IDXcol col_offset,
                                                                        iReg   * __restrict__ nc,
                                                                        IDXcol * __restrict__ Key,
                                                                        rExt   * __restrict__ Data,
                                                                        iReg   * sh_sums ) {

   // Thread and warp info
   int id = threadIdx.x;
   int lane_id = id%WARPSIZE;
   int warp_id = id/WARPSIZE;
   int blksz = blockDim.x;
   int nwarps_in_blk = (blksz + WARPSIZE - 1) / WARPSIZE;

   // Other variables
   int i,jj,last_id;
   iReg pos,locSum,warpSum,blockSum;
   IDXcol key_value;
   rExt data_value;

   // First thread inits the first total running sum to zero
   if (id == 0) sh_sums[WARPSIZE] = 0;

   // Loop over all elements using 1 thread per element
   for (jj = id; jj < nn; jj += blksz){

      // Record "key_value" and decide wether it will deserve or not a position
      key_value = Key[jj];
      data_value = Data[jj];
      pos = (key_value<0) ? 0:1;
     
      // Compute prefix sum in each warp
      #pragma unroll
      for (i = 1; i < WARPSIZE; i *= 2){
          locSum = __shfl_up_sync(MASKFULL, pos, i, WARPSIZE);
          if (lane_id >= i) pos += locSum;
      }

      // Write the sum of the warp into the sh_sums array
      if (lane_id == WARPSIZE-1) sh_sums[warp_id] = pos;
      __syncthreads();

      // First warp computes the blockSum
      if (warp_id == 0){
         warpSum = (lane_id < nwarps_in_blk) ? sh_sums[lane_id]:0;
         #pragma unroll
         for (i = 1; i < WARPSIZE; i *= 2){
            locSum = __shfl_up_sync(MASKFULL,warpSum,i,WARPSIZE);
            if (lane_id >= i) warpSum += locSum;
         }
         sh_sums[lane_id] = warpSum;
      }
      __syncthreads();

      // Store in blockSum the running sum correspondig to this warp
      blockSum = (warp_id > 0) ? sh_sums[warp_id-1] : 0;
      // Add blockSum and the total running sum to determine the position of each entry
      pos += blockSum + sh_sums[WARPSIZE];

      // Store key_value back in Key
      if (key_value >= 0){
         Key[pos-1] = key_value + col_offset;
         Data[pos-1] = data_value;
      }
      __syncthreads();

      // Store the new total running sum value
      if (id == blksz-1) sh_sums[WARPSIZE] = pos;
   }

   // Last active thread in the last cycle dumps the final number of entries
   last_id = nn%blksz;
   last_id = (!last_id) ? blksz-1:last_id-1;
   if (id == last_id) *nc = pos;

}

//----------------------------------------------------------------------------------------

// compact the array Key[] and store the values in coef_C[] and its indices in ja_C[]
template <typename IDXcol>
static __device__ __forceinline__ void dev_compactVal( const iReg nn,
                                                       const int tid,
                                                       const int wid,
                                                       const int wnum,
                                                       const unsigned char * __restrict__ check,
                                                       const rExt   * __restrict__ Key,
                                                             rExt   * __restrict__ coef_C,
                                                             IDXcol * __restrict__ ja_C,
                                                             iReg   * sh_sums ) {
   // Other variables
   int i,jj;
   iReg pos,locSum,warpSum,blockSum;
   unsigned char bit_check;
   rExt key_value;

   // First thread inits the first total running sum to zero
   if (threadIdx.x == 0) sh_sums[WARPSIZE] = 0;

   // Loop over all elements using 1 thread per element
   for (jj = threadIdx.x; jj < nn; jj += blockDim.x){

      // Record "key_value" and decide wether it will deserve or not a position
      key_value = Key[jj];
      bit_check = check[jj];
      pos = (bit_check == 0x00) ? 0:1;
     
      // Compute prefix sum in each warp
      #pragma unroll
      for (i = 1; i < WARPSIZE; i *= 2){
          locSum = __shfl_up_sync(MASKFULL, pos, i, WARPSIZE);
          if (tid >= i) pos += locSum;
      }

      // Write the sum of the warp into the sh_sums array
      if (tid == WARPSIZE-1) sh_sums[wid] = pos;
      __syncthreads();

      // First warp computes the blockSum
      if (wid == 0){
         warpSum = (tid < wnum) ? sh_sums[tid]:0;
         #pragma unroll
         for (i = 1; i < WARPSIZE; i *= 2){
            locSum = __shfl_up_sync(MASKFULL,warpSum,i,WARPSIZE);
            if (tid >= i) warpSum += locSum;
         }
         sh_sums[tid] = warpSum;
      }
      __syncthreads();

      // Store in blockSum the running sum correspondig to this warp
      blockSum = (wid > 0) ? sh_sums[wid-1]:0;
      // Add blockSum and the total running sum to determine the position of each entry
      pos += blockSum + sh_sums[WARPSIZE];

      // Store key_value back in Key
      if (bit_check != 0x00) {
         coef_C[pos-1] = key_value;
         ja_C[pos-1]   = jj;
      }
      __syncthreads();

      // Store the new total running sum value
      if (threadIdx.x == blockDim.x-1) sh_sums[WARPSIZE] = pos;

   }

}

//----------------------------------------------------------------------------------------

template <typename IDXcol>
static __device__ __forceinline__ void dev_compactVal_chunk_dense( const iReg nn,
                                                                   const int tid,
                                                                   const int wid,
                                                                   const int wnum,
                                                                   const IDXcol istrB,
                                                                   const unsigned char * __restrict__ check,
                                                                   const rExt   * __restrict__ Data,
                                                                         rExt   * __restrict__ coef,
                                                                         IDXcol * __restrict__ ja,
                                                                         iReg   * nc,
                                                                         iReg   * sh_sums ) {

   // Other variables
   int i,jj,last_id;
   iReg pos,locSum,warpSum,blockSum;
   unsigned char bit_check;
   rExt data_value;

   // First thread inits the first total running sum to zero
   if (threadIdx.x == 0) sh_sums[WARPSIZE] = 0;

   // Loop over all elements using 1 thread per element
   for (jj = threadIdx.x; jj < nn; jj += blockDim.x){

      // Record "bit_check" and decide wether it will deserve or not a position
      data_value = Data[jj];
      bit_check = check[jj];
      pos = (bit_check == 0x00) ? 0:1;
     
      // Compute prefix sum in each warp
      #pragma unroll
      for (i = 1; i < WARPSIZE; i *= 2){
          locSum = __shfl_up_sync(MASKFULL, pos, i, WARPSIZE);
          if (tid >= i) pos += locSum;
      }

      // Write the sum of the warp into the sh_sums array
      if (tid == WARPSIZE-1) sh_sums[wid] = pos;
      __syncthreads();

      // First warp computes the blockSum
      if (wid == 0){
         warpSum = (tid < wnum) ? sh_sums[tid]:0;
         #pragma unroll
         for (i = 1; i < WARPSIZE; i *= 2){
            locSum = __shfl_up_sync(MASKFULL,warpSum,i,WARPSIZE);
            if (tid >= i) warpSum += locSum;
         }
         sh_sums[tid] = warpSum;
      }
      __syncthreads();

      // Store in blockSum the running sum correspondig to this warp
      blockSum = (wid > 0) ? sh_sums[wid-1]:0;
      // Add blockSum and the total running sum to determine the position of each entry
      pos += blockSum + sh_sums[WARPSIZE];

      // Store key_value back in Key
      if (bit_check != 0x00){
         ja[pos-1] = jj + istrB;
         coef[pos-1] = data_value;
      }
      __syncthreads();

      // Store the new total running sum value
      if (threadIdx.x == blockDim.x-1) sh_sums[WARPSIZE] = pos;
   }

   // Last active thread in the last cycle dumps the final number of entries
   last_id = nn & (blockDim.x-1);
   last_id = (!last_id) ? blockDim.x-1:last_id-1;
   if (threadIdx.x == last_id) *nc = pos;

}

//----------------------------------------------------------------------------------------
