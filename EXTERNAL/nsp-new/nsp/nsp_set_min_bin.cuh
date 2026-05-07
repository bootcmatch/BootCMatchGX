#include <cub/cub.cuh>

//----------------------------------------------------------------------------------------

// Compute the exact size of C rows and update sfBIN
void nsp_set_min_bin( sfBIN * bin,
                      const iReg nrows_C,
                      const iReg ncols_C,
                            int  & DIRECT,
                            iReg & ifrac,
                            rExt & frac ) {

   // Initilize error flag
   cudaError_t cudaError = cudaSuccess;

   // set handles
   int max_nnz_shared = bin->max_nnz_shared_comp;
   iReg *h_bin_offset = bin->h_bin_offset.data();
   iReg *h_bin_size   = bin->h_bin_size.data();
   iReg *d_row_nz     = bin->d_row_nz.data();
   iReg *d_bin_offset = bin->d_bin_offset.data();
   iReg *d_bin_size   = bin->d_bin_size.data();
   iReg *d_row_perm   = bin->d_row_perm.data();

   // initialize sfBIN structure to 0
   for (iReg i = 0; i < BIN_NUM; i++) {
      h_bin_size[i]   = 0;
      h_bin_offset[i] = 0;
   }
   cudaError = cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(iReg));
   CheckCudaError(ERROR_INFO,"cudaMemset on d_bin_size failed", cudaError);

   // Compute size of C rows
   iReg BS = BLKSIZE_MxM;
   iReg GS = div_round_up(nrows_C,BS);
   LaunchCudaKernel(nsp_set_bin_min<<<GS,BS>>>(d_row_nz,d_bin_size,nrows_C,ncols_C,max_nnz_shared));

   // copy group sizes from Device to Host
   cudaError = cudaMemcpy(h_bin_size, d_bin_size, sizeof(iReg) * BIN_NUM, cudaMemcpyDeviceToHost);
   CheckCudaError(ERROR_INFO,"copying bin_size D2H failed", cudaError);

   // if the largest bin is dominant (has > 15% of the rows) then don't permute the rows and use direct access
   int i = BIN_NUM - 1;
   while (h_bin_size[i] == 0) i--;
   frac  = (rExt)h_bin_size[i]/nrows_C;
   ifrac = i;
   if ((rExt)h_bin_size[i]/nrows_C > 0.15 ) { // add condition that it is not the chunk bins
      // nulify the use of other bins
      for (iReg j = 0; j < i; j++) h_bin_size[j] = 0;
      h_bin_size[i] = nrows_C;      // set up the largest bin to process all the rows
      d_row_perm = nullptr;         // nulify the row permutation pointer
      // set flag
      DIRECT = 1;
   } else {
      // reset to 0 group sizes on the Device (recomputed later in set_row_perm)
      cudaError = cudaMemset(d_bin_size, 0, BIN_NUM * sizeof(iReg));
      CheckCudaError(ERROR_INFO,"cudaMemset on d_bin_size failed", cudaError);
      // set-up host
      for (iReg i = 0; i < BIN_NUM - 1; i++) {
         h_bin_offset[i+1] = h_bin_offset[i] + iReg(h_bin_size[i]);
      }
      cudaError = cudaMemcpy(d_bin_offset, h_bin_offset, sizeof(iReg) * BIN_NUM, cudaMemcpyHostToDevice);
      CheckCudaError(ERROR_INFO,"copying bin_offset H2D failed", cudaError);
      LaunchCudaKernel(nsp_set_row_perm_min<<<GS,BS>>>(d_bin_size,d_bin_offset,d_row_nz,d_row_perm,
                                                       nrows_C,ncols_C,max_nnz_shared));
      // set flag
      DIRECT = 0;
   }

   #if defined BENCHMARK_MXM
      for (iReg i = 0; i < BIN_NUM; i++) cout << h_bin_size[i] << " ";
      cout << endl;
      // FILE *fid = fopen("bin.txt","a");
      // for (iReg i = 0; i < BIN_NUM; i++) fprintf(fid,"%d ",h_bin_size[i]);
      // fprintf(fid,"\n");
      // fclose(fid);
   #endif

}

//----------------------------------------------------------------------------------------
