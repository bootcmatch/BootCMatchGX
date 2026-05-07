//----------------------------------------------------------------------------------------

// set-up permutation array for symbolic computation
__global__ void nsp_set_row_perm(       iReg * __restrict__ bin_size,
                                  const iReg * __restrict__ bin_offset,
                                  const iReg * __restrict__ max_row_nz,
                                        iReg * __restrict__ row_perm,
                                  const iReg nrows_C,
                                  const int max_nnz_shared ) {

   // retrieve row index
   iReg i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i >= nrows_C) return;

   // other registers
   iReg nz_per_row = HASH_FAC_SYMB*max_row_nz[i];
   iReg dest;

   if (nz_per_row <= max_nnz_shared){

      if (nz_per_row <= 64){
         // mpwarp
         dest = myatomicAdd(bin_size, 1);
         row_perm[bin_offset[0] + dest] = i;
      }
      else if (nz_per_row <= 128){
         // mpwarp
         dest = myatomicAdd(bin_size+1, 1);
         row_perm[bin_offset[1] + dest] = i;
      }
      else if (nz_per_row <= 256){
         // mpwarp
         dest = myatomicAdd(bin_size+2, 1);
         row_perm[bin_offset[2] + dest] = i;
      }
      else if (nz_per_row <= 512){
         // tb
         dest = myatomicAdd(bin_size+3, 1);
         row_perm[bin_offset[3] + dest] = i;
      }
      else if (nz_per_row <= 1024){
         // tb
         dest = myatomicAdd(bin_size+4, 1);
         row_perm[bin_offset[4] + dest] = i;
      }
      else if (nz_per_row <= 2048){
         // tb
         dest = myatomicAdd(bin_size+5, 1);
         row_perm[bin_offset[5] + dest] = i;
      }
      else if (nz_per_row <= 4096){
         // tb
         dest = myatomicAdd(bin_size+6, 1);
         row_perm[bin_offset[6] + dest] = i;
      }
      else if (nz_per_row <= 8192){
         // tb
         dest = myatomicAdd(bin_size+7, 1);
         row_perm[bin_offset[7] + dest] = i;
      }
      else if (nz_per_row <= 12288){
         // tb_max
         dest = myatomicAdd(bin_size+8, 1);
         row_perm[bin_offset[8] + dest] = i;
      }
      else{
         // tb_chunk
         dest = myatomicAdd(bin_size+9, 1);
         row_perm[bin_offset[9] + dest] = i;
      }
   }
   else{
      // tb_chunk
      dest = myatomicAdd(bin_size+9, 1);
      row_perm[bin_offset[9] + dest] = i;
   }

}

//----------------------------------------------------------------------------------------

// set-up permutation array for calculation of values
__global__ void nsp_set_row_perm_min(iReg *bin_size, iReg *bin_offset, iReg *max_row_nz,
                                     iReg *row_perm, iReg nrows_C, iReg ncols_C,
                                     const int max_nnz_shared) {

   // retrieve row index
   iReg i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i >= nrows_C) return;

   // other registers
   iReg nz_per_row = HASH_FAC_COMP*max_row_nz[i];
   rExt density = (rExt)nz_per_row / ncols_C * 100.;
   iReg dest;

   // BINNUM = 11
   if (nz_per_row <= max_nnz_shared){

      if ( (ncols_C <= MAX_SH_DENSE && density > MIN_DENSITY) || (density > MIN_DENSITY_chunk) ){
         // dense bin
         dest = myatomicAdd(bin_size +  BIN_NUM-1, 1);
         row_perm[bin_offset[BIN_NUM - 1] + dest] = i;
      }
      else if (nz_per_row <= 16){
         // mpwarp
         dest = myatomicAdd(bin_size, 1);
         row_perm[bin_offset[0] + dest] = i;
      }
      else if (nz_per_row <= 32){
         // mpwarp
         dest = myatomicAdd(bin_size + 1, 1);
         row_perm[bin_offset[1] + dest] = i;
      }
      else if (nz_per_row <= 64){
         // mpwarp
         dest = myatomicAdd(bin_size + 2, 1);
         row_perm[bin_offset[2] + dest] = i;
      }
      else if (nz_per_row <= 128){
         // warp
         dest = myatomicAdd(bin_size + 3, 1);
         row_perm[bin_offset[3] + dest] = i;
      }
      else if (nz_per_row <= 256){
         // tb
         dest = myatomicAdd(bin_size + 4, 1);
         row_perm[bin_offset[4] + dest] = i;
      }
      else if (nz_per_row <= 512){
         // tb
         dest = myatomicAdd(bin_size + 5, 1);
         row_perm[bin_offset[5] + dest] = i;
      }
      else if (nz_per_row <= 1024){
         // tb
         dest = myatomicAdd(bin_size + 6, 1);
         row_perm[bin_offset[6] + dest] = i;
      }
      else if (nz_per_row <= 2048){
         // tb
         dest = myatomicAdd(bin_size + 7, 1);
         row_perm[bin_offset[7] + dest] = i;
      }
      else if (nz_per_row <= 4096){
         // tb_outsort
         dest = myatomicAdd(bin_size + 8, 1);
         row_perm[bin_offset[8] + dest] = i;
      }
      else{
         // tb_chunk
         dest = myatomicAdd(bin_size + BIN_NUM-2, 1);
         row_perm[bin_offset[BIN_NUM-2] + dest] = i;
      }

   }
   else{
      // tb_chunk
      dest = myatomicAdd(bin_size + BIN_NUM-2, 1);
      row_perm[bin_offset[BIN_NUM-2] + dest] = i;
   }

}

//----------------------------------------------------------------------------------------
