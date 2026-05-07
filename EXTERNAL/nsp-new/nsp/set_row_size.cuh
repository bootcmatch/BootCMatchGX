#pragma once

//----------------------------------------------------------------------------------------

__global__ void set_row_size( const iExt * __restrict__ iat_A,
                              const iReg * __restrict__ d_row_perm,
                              const iReg nn,
                              const iReg bin_shift,
                                    iExt * __restrict__ d_A_col_offsets ) {

   int tid = threadIdx.x;  
      
   if (d_row_perm == nullptr){
      while(tid < nn){
         d_A_col_offsets[tid] = iat_A[tid+1] - iat_A[tid]; 
         tid += blockDim.x;
      }
   }
   else{
      while(tid < nn){
         int ind = d_row_perm[bin_shift + tid];
         d_A_col_offsets[tid] = iat_A[ind+1] - iat_A[ind]; 
         tid += blockDim.x;
      }
   }
   // Unneeded but set to prevent initcheck error in prefixSumExclusive
   if (tid == nn) d_A_col_offsets[nn] = 0;
}

//----------------------------------------------------------------------------------------
