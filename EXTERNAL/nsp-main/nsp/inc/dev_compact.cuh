static __device__ __forceinline__ void dev_compactVal(const int nn,const unsigned char *check,const double *Key, int *sh_sums, double *coef_C, int *ja_C,const int tid,const int wid, const int wnum){

   // Other variables
   int locSum,warpSum,blockSum;

   // First thread inits the first total running sum to zero
   if (threadIdx.x == 0) sh_sums[warpSize] = 0;

   // Loop over all elements using 1 thread per element
   for (int jj = threadIdx.x; jj < nn; jj += blockDim.x){

      // Record "key_value" and decide wether it will deserve or not a position
      double        key_value =   Key[jj];
      unsigned char bit_check = check[jj];
      int pos = (bit_check == 0x00) ? 0:1;
     
      // Compute prefix sum in each warp
      #pragma unroll
      for (int i = 1; i < warpSize; i *= 2){
          locSum = __shfl_up_sync(MASKFULL,pos, i, warpSize);
          if (tid >= i) pos += locSum;
      }

      // Write the sum of the warp into the sh_sums array
      if (tid == warpSize-1) sh_sums[wid] = pos;
      __syncthreads();

      // First warp computes the blockSum
      if (wid == 0){
         warpSum = (tid < wnum) ? sh_sums[tid]:0;
         #pragma unroll
         for (int i = 1; i < warpSize; i *= 2){
            locSum = __shfl_up_sync(MASKFULL,warpSum,i,warpSize);
            if (tid >= i) warpSum += locSum;
         }
         sh_sums[tid] = warpSum;
      }
      __syncthreads();

      // Store in blockSum the running sum correspondig to this warp
      blockSum = (wid > 0) ? sh_sums[wid-1]:0;
      // Add blockSum and the total running sum to determine the position of each entry
      pos += blockSum + sh_sums[warpSize];

      // Store key_value back in Key
      if (bit_check != 0x00) {
         coef_C[pos-1] = key_value;
           ja_C[pos-1] = jj;
      }
      __syncthreads();
      // Store the new total running sum value
      if (threadIdx.x == blockDim.x-1) sh_sums[warpSize] = pos;

   }

}

//////////////////////////////////////////////////////////////////////////////////////////

static __device__ __forceinline__ void dev_compactKeyVal_inplace(const int nn, int *nc, int *Key, double *Data,
                                  int *sh_sums,int col_offset){

   // Thread and warp info
   int id = threadIdx.x;
   int lane_id = id%warpSize;
   int warp_id = id/warpSize;
   int blksz = blockDim.x;
   int nwarps_in_blk = (blksz + warpSize - 1) / warpSize;

   // Other variables
   int i,jj;
   int pos,key_value,locSum,warpSum,blockSum,last_id;
   double data_value;

   // First thread inits the first total running sum to zero
   if (id == 0) sh_sums[warpSize] = 0;

   // Loop over all elements using 1 thread per element
   for (jj = id; jj < nn; jj += blksz){

      // Record "key_value" and decide wether it will deserve or not a position
      key_value = Key[jj];
      data_value = Data[jj];
      pos = (key_value<0) ? 0:1;
     
      // Compute prefix sum in each warp
      #pragma unroll
      for (i = 1; i < warpSize; i *= 2){
          locSum = __shfl_up_sync(MASKFULL,pos, i, warpSize);
          if (lane_id >= i) pos += locSum;
      }

      // Write the sum of the warp into the sh_sums array
      if (lane_id == warpSize-1) sh_sums[warp_id] = pos;
      __syncthreads();

      // First warp computes the blockSum
      if (warp_id == 0){
         warpSum = (lane_id < nwarps_in_blk) ? sh_sums[lane_id]:0;
         #pragma unroll
         for (i = 1; i < warpSize; i *= 2){
            locSum = __shfl_up_sync(MASKFULL,warpSum,i,warpSize);
            if (lane_id >= i) warpSum += locSum;
         }
         sh_sums[lane_id] = warpSum;
      }
      __syncthreads();

      // Store in blockSum the running sum correspondig to this warp
      blockSum = (warp_id > 0) ? sh_sums[warp_id-1]:0;
      // Add blockSum and the total running sum to determine the position of each entry
      pos += blockSum + sh_sums[warpSize];

      // Store key_value back in Key
      if (key_value >= 0){
         Key[pos-1] = key_value + col_offset;
         Data[pos-1] = data_value;
      }
      __syncthreads();

      // Store the new total running sum value
      if (id == blksz-1) sh_sums[warpSize] = pos;
   }

   // Last active thread in the last cycle dumps the final number of entries
   last_id = nn%blksz;
   last_id = (!last_id) ? blksz-1:last_id-1;
   if (id == last_id) *nc = pos;

}

//////////////////////////////////////////////////////////////////////////////////////////

// template <const int BS,const int nn, const int wnum>
// __device__ __forceinline__ void dev_compactVal_chunk_dense(const unsigned char *check,const double *value,
//                               int *sh_sums,int istrB,double *coef, int *ja,const int tid,const int wid){

//    // First thread inits the first total running sum to zero
//    if (threadIdx.x == 0) sh_sums[WARPSIZE] = 0;

//    // Loop over all elements using 1 thread per element
//    for (uint16_t jj = threadIdx.x; jj < nn; jj += BS){

//       // Record "key" and decide whether it will deserve or not a position
//       unsigned char key = check[jj];
//       int           pos = (key == 0x00) ? 0:1; // could be int or uint16_t
     
//       // Compute prefix sum in each warp
//       #pragma unroll
//       for (int i = 1; i < WARPSIZE; i <<= 1){
//          int locSum = __shfl_up_sync(MASKFULL,pos,i,WARPSIZE);
//          if (tid >= i) pos += locSum;
//       }

//       // Write the sum of the warp into the sh_sums array
//       if (tid == WARPSIZE-1) sh_sums[wid] = pos;
//       __syncthreads();

//       // First warp computes the blockSum
//       if (wid == 0){
//          int warpSum = (tid < wnum) ? sh_sums[tid]:0;
//          #pragma unroll
//          for (int i = 1; i < WARPSIZE; i <<= 1){
//             int locSum = __shfl_up_sync(MASKFULL,warpSum,i,WARPSIZE);
//             if (tid >= i) warpSum += locSum;
//          }
//          sh_sums[tid] = warpSum;
//       }
//       __syncthreads();

//       // add warp and block running sums
//       pos += (wid > 0) ? sh_sums[WARPSIZE] + sh_sums[wid-1] : sh_sums[WARPSIZE];

//       // Store key_value back in Key
//       if (key != 0x00){
//            ja[pos-1] = jj + istrB;
//          coef[pos-1] = value[jj];
//       }
//       __syncthreads();

//       // Store the new total running sum value
//       if (threadIdx.x == BS-1) sh_sums[WARPSIZE] = pos;
//    }

// }

static __device__ __forceinline__ void dev_compactVal_chunk_dense(const int nn, int *nc,const unsigned char *check,const double *Data,
                                  int *sh_sums,int istrB,double *coef, int *ja,const int tid,const int wid, const int wnum){

   // variables
   int pos,locSum,warpSum,blockSum,last_id;

   // First thread inits the first total running sum to zero
   if (threadIdx.x == 0) sh_sums[warpSize] = 0;

   // Loop over all elements using 1 thread per element
   for (int jj = threadIdx.x; jj < nn; jj += blockDim.x){

      // Record "bit_check" and decide wether it will deserve or not a position
      double data_value = Data[jj];
      unsigned char bit_check = check[jj];
      pos = (bit_check == 0x00) ? 0:1;
     
      // Compute prefix sum in each warp
      #pragma unroll
      for (int i = 1; i < warpSize; i *= 2){
          locSum = __shfl_up_sync(MASKFULL,pos, i, warpSize);
          if (tid >= i) pos += locSum;
      }

      // Write the sum of the warp into the sh_sums array
      if (tid == warpSize-1) sh_sums[wid] = pos;
      __syncthreads();

      // First warp computes the blockSum
      if (wid == 0){
         warpSum = (tid < wnum) ? sh_sums[tid]:0;
         #pragma unroll
         for (int i = 1; i < warpSize; i *= 2){
            locSum = __shfl_up_sync(MASKFULL,warpSum,i,warpSize);
            if (tid >= i) warpSum += locSum;
         }
         sh_sums[tid] = warpSum;
      }
      __syncthreads();

      // Store in blockSum the running sum correspondig to this warp
      blockSum = (wid > 0) ? sh_sums[wid-1]:0;
      // Add blockSum and the total running sum to determine the position of each entry
      pos += blockSum + sh_sums[warpSize];

      // Store key_value back in Key
      if (bit_check != 0x00){
         ja[pos-1] = jj + istrB;
         coef[pos-1] = data_value;
      }
      __syncthreads();

      // Store the new total running sum value
      if (threadIdx.x == blockDim.x-1) sh_sums[warpSize] = pos;
   }

   // Last active thread in the last cycle dumps the final number of entries
   last_id = nn & (blockDim.x-1);
   last_id = (!last_id) ? blockDim.x-1:last_id-1;
   if (threadIdx.x == last_id) *nc = pos;

}


