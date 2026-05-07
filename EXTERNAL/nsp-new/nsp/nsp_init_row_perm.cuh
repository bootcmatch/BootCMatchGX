
//----------------------------------------------------------------------------------------

// initialize permutation array
__global__ void nsp_init_row_perm( iReg * __restrict__ permutation, const iReg nrows_C ) {

   // retrieve row index
   iReg i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i >= nrows_C) return;

   permutation[i] = i;

}

//----------------------------------------------------------------------------------------
