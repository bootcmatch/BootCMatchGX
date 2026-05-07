
//----------------------------------------------------------------------------------------

// Estimate size of C rows as number of intprod
template <typename IDXcol>
__global__ void nsp_set_intprod_num( const iReg nrows_C,
                                     const iExt   * __restrict__ iat_A,
                                     const IDXcol * __restrict__ ja_A,
                                     const iExt   * __restrict__ iat_B,
                                           iReg   * __restrict__ row_intprod ) {

   // retrieve row index
   iReg i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i >= nrows_C) return;

   // initialize number of intermediate products
   iReg nz_per_row = 0;

   // compute number of intprod
   for (iExt j = iat_A[i]; j < iat_A[i+1]; j++) {
      IDXcol jcol_A = ja_A[j];
      nz_per_row += iat_B[jcol_A+1] - iat_B[jcol_A];
   }

   // store the number
   row_intprod[i] = nz_per_row;

}

//----------------------------------------------------------------------------------------
