
#pragma once
#include "nsp/prefixSumExclusive.h"

#define BS_OUTSORT 512
#define BS_CHUNK 512
#define BS_DENSE 512

//----------------------------------------------------------------------------------------

// compute C
template<typename IDXcol>
void nsp_calculate_value_col_bin( const iExt   * __restrict__ d_iat_A,
                                  const IDXcol * __restrict__ d_ja_A,
                                  const rExt   * __restrict__ d_coef_A,
                                  const iExt   * __restrict__ d_iat_B,
                                  const IDXcol * __restrict__ d_ja_B,
                                  const rExt   * __restrict__ d_coef_B,
                                  const iExt   * __restrict__ d_iat_C,
                                        IDXcol * __restrict__ d_ja_C,
                                        rExt   * __restrict__ d_coef_C,
                                        sfBIN  * bin,
                                  const iExt   nrows_C,
                                  const IDXcol ncols_C,
                                  const iExt   nterm_C,
                                  const int    DIRECT ) {

   // Initilize error flag
   cudaError_t cudaError = cudaSuccess;

   // set handles
   iReg *h_bin_offset = bin->h_bin_offset.data();
   iReg *h_bin_size   = bin->h_bin_size.data();
   iReg *d_row_perm   = (DIRECT) ? nullptr : bin->d_row_perm.data();
   iReg *d_row_nz     = bin->d_row_nz.data();

   // define varibles for GPU resources
   int maxthread,maxNblk;
   size_t SHthread,shmemsize;
   iReg GS,BS,SH;

   int warpSize = ChronosDevice.get_warpSize();
   int maxBlkMP = ChronosDevice.get_maxBlocksPerMultiProcessor();
   int maxThrMP = ChronosDevice.get_maxThreadsPerMultiProcessor();
   int maxThrBlk = ChronosDevice.get_maxThreadsPerBlock();
   long int ShMemMP = ChronosDevice.get_sharedMemPerMultiprocessor();
   long int ShMemBlk = ChronosDevice.get_sharedMemPerBlock();

   //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   #ifdef DEBUG_MXM
   type_MPI_int rank;
   MPI_Comm_rank(Chronos.Get_currComm(), &rank);
   #endif
   //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

   // loop over groups
   for (iReg i = BIN_NUM - 1; i >= 0; i--) {

      // check sizes
      if (h_bin_size[i] > 0) {

         // select group kernel
         switch (i) {

            case 0: // <= 16
               // Memory per thread
               SHthread = (16/4)*( sizeof(IDXcol) + sizeof(rExt) );
               // Max number of threads
               maxthread = min(maxThrMP,int(ShMemMP/SHthread));
               // Determine block size giving max occupancy
               BS = (maxthread/(maxBlkMP*warpSize))*warpSize;
               // Shared memory per block
               shmemsize = SHthread * BS;
               // Grid size
               GS = h_bin_size[i]/(BS/4)+1;
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld, stream %d\n",
                      rank,__FUNCTION__,i,"mpwarp",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_calculate_value_col_bin_mpwarp<4,16>
                  <<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                   d_row_perm,h_bin_offset[i],h_bin_size[i]));
               break;

            case 1: // <= 32
               // Memory per thread
               SHthread = (32/8)*( sizeof(IDXcol) + sizeof(rExt) );
               // Max number of threads
               maxthread = min(maxThrMP,int(ShMemMP/SHthread));
               // Determine block size giving max occupancy
               BS = (maxthread/(maxBlkMP*warpSize))*warpSize;
               // Shared memory per block
               shmemsize = SHthread * BS;
               // Grid size
               GS = h_bin_size[i]/(BS/8)+1;
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld, stream %d\n",
                      rank,__FUNCTION__,i,"mpwarp",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_calculate_value_col_bin_mpwarp<8,32>
                  <<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                   d_row_perm,h_bin_offset[i],h_bin_size[i]));
               break;

            case 2 : // <= 64
               // Memory per thread
               SHthread = (64/16)*( sizeof(IDXcol) + sizeof(rExt) );
               // Max number of threads
               maxthread = min(maxThrMP,int(ShMemMP/SHthread));
               // Determine block size giving max occupancy
               BS = (maxthread/(maxBlkMP*warpSize))*warpSize;
               // Shared memory per block
               shmemsize = SHthread * BS;
               // Grid size
               GS = h_bin_size[i]/(BS/16)+1;
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld, stream %d\n",
                      rank,__FUNCTION__,i,"mpwarp",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_calculate_value_col_bin_mpwarp<16,64>
                  <<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                   d_row_perm,h_bin_offset[i],h_bin_size[i]));
               break;

            case 3 : // <= 128
               // Memory per thread
               SHthread = (128/32)*( sizeof(IDXcol) + sizeof(rExt) );
               // Max number of threads
               maxthread = min(maxThrMP,int(ShMemMP/SHthread));
               // Determine block size giving max occupancy
               BS = (maxthread/(maxBlkMP*warpSize))*warpSize;
               // Shared memory per block
               shmemsize = SHthread * BS;
               // Grid size
               GS = h_bin_size[i]/(BS/32)+1;
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld, stream %d\n",
                      rank,__FUNCTION__,i,"mpwarp",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_calculate_value_col_bin_mpwarp<32,128>
                  <<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                   d_row_perm,h_bin_offset[i],h_bin_size[i]));
               break;

            case 4 : // <= 256
               // Memory per block
               shmemsize = 256*( sizeof(IDXcol) + sizeof(rExt) );
               // Max number of blocks per MP
               maxNblk = min(int(ShMemMP/shmemsize),maxBlkMP);
               // Block size
               BS = min(maxThrMP/maxNblk,maxThrBlk);
               BS = BS<WARPSIZE ? WARPSIZE : (BS/WARPSIZE)*WARPSIZE;
               // Grid size
               GS = h_bin_size[i];
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld, stream %d\n",
                      rank,__FUNCTION__,i,"tb",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_calculate_value_col_bin_each_tb<256>
                  <<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                   d_row_perm,d_row_nz,h_bin_offset[i],h_bin_size[i]));
               break;

            case 5 : // <= 512
               // Memory per block
               shmemsize = 512*( sizeof(IDXcol) + sizeof(rExt) );
               // Max number of blocks per MP
               maxNblk = min(int(ShMemMP/shmemsize),maxBlkMP);
               // Block size
               BS = min(maxThrMP/maxNblk,maxThrBlk);
               BS = BS<WARPSIZE ? WARPSIZE : (BS/WARPSIZE)*WARPSIZE;
               // Grid size
               GS = h_bin_size[i];
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld, stream %d\n",
                      rank,__FUNCTION__,i,"tb",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_calculate_value_col_bin_each_tb<512>
                  <<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                   d_row_perm,d_row_nz,h_bin_offset[i],h_bin_size[i]));
               break;

            case 6 : // <= 1024
               // Memory per block
               shmemsize = 1024*( sizeof(IDXcol) + sizeof(rExt) );
               // Max number of blocks per MP
               maxNblk = min(int(ShMemMP/shmemsize),maxBlkMP);
               // Block size
               BS = min(maxThrMP/maxNblk,maxThrBlk);
               BS = BS<WARPSIZE ? WARPSIZE : (BS/WARPSIZE)*WARPSIZE;
               // Grid size
               GS = h_bin_size[i];
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld, stream %d\n",
                      rank,__FUNCTION__,i,"tb",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_calculate_value_col_bin_each_tb<1024>
                  <<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                   d_row_perm,d_row_nz,h_bin_offset[i],h_bin_size[i]));
               break;

            case 7 : // <= 2048
               // Memory per block
               shmemsize = 2048*( sizeof(IDXcol) + sizeof(rExt) );
               // Max number of blocks per MP
               maxNblk = min(int(ShMemMP/shmemsize),maxBlkMP);
               // Block size
               BS = min(maxThrMP/maxNblk,maxThrBlk);
               BS = BS<WARPSIZE ? WARPSIZE : (BS/WARPSIZE)*WARPSIZE;
               // Grid size
               GS = h_bin_size[i];
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld, stream %d\n",
                      rank,__FUNCTION__,i,"tb",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_calculate_value_col_bin_each_tb<2048>
                  <<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                   d_row_perm,d_row_nz,h_bin_offset[i],h_bin_size[i]));
               break;

            case 8 : // <= 4096
               // Memory per block
               shmemsize = 4096*( sizeof(IDXcol) + sizeof(rExt) );
               // Max number of blocks per MP
               maxNblk = min(int(ShMemMP/shmemsize),maxBlkMP);
               // Block size
               BS = min(maxThrMP/maxNblk,maxThrBlk);
               BS = BS<WARPSIZE ? WARPSIZE : (BS/WARPSIZE)*WARPSIZE;
               // Grid size
               GS = h_bin_size[i];
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld, stream %d\n",
                      rank,__FUNCTION__,i,"tb_outsort",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_calculate_value_col_bin_each_tb_outsort<4096>
                  <<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                   d_row_perm,d_row_nz,h_bin_offset[i],h_bin_size[i]));

               // sort the arrays on global memory using cub library
               {
               SH = 4096;
               IDXcol nSigBits = countBITS(ncols_C);

               // Dangerous: templating BS prevents setting it to meet shmem requirements. Hence,
               // too many resources could be requested.
               LaunchCudaKernel(nsp_calc_val_sort_rows<BS_OUTSORT><<<GS, BS_OUTSORT, shmemsize, bin->stream[i]>>>
                  (nSigBits,SH,h_bin_offset[i],d_row_perm,d_iat_C,d_ja_C,d_coef_C));
               }

               break;

            case 9 :
               // (Maximum) memory per block
               SH = ShMemBlk / ( sizeof(IDXcol) + sizeof(rExt) );
               shmemsize = SH * ( sizeof(IDXcol) + sizeof(rExt) );
               // Grid size
               GS = h_bin_size[i];
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               #ifdef DEBUG_MXM
               printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld (SH %d), stream %d\n",
                      rank,__FUNCTION__,i,"tb_chunk_dynamic",GS,BS_CHUNK,shmemsize,SH,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               // try to compute all the rows using the standard kernel (hasmap_mod)
               if (SH >= 4096) {
                  LaunchCudaKernel(nsp_calculate_value_col_bin_each_tb_chunk_dynamic<4096-BS_CHUNK,BS_CHUNK>
                        <<<GS, BS_CHUNK, shmemsize, bin->stream[i]>>>
                        (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                         d_row_perm,h_bin_offset[i],h_bin_size[i],ncols_C));
               } else {
                  LaunchCudaKernel(nsp_calculate_value_col_bin_each_tb_chunk_dynamic<3072-BS_CHUNK,BS_CHUNK>
                        <<<GS, BS_CHUNK, shmemsize, bin->stream[i]>>>
                        (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                         d_row_perm,h_bin_offset[i],h_bin_size[i],ncols_C));
               }

               break;

            case 10 :
               // Grid size
               GS = h_bin_size[i];

               if (ncols_C <= MAX_SH_DENSE){
                  // Unfortunately, we have to use the size of shared table as the multiple of the blocksize.
                  // Otherwise, the shared table initialization cannot succeed
                  SH = ((ncols_C - 1) / BS_DENSE + 1) * BS_DENSE;
                  shmemsize = SH * (sizeof(rExt)+sizeof(unsigned char)) + (WARPSIZE + 1)*sizeof(iReg);
                  if (shmemsize > (size_t)ShMemBlk) {
                     throw linsol_error(ERROR_INFO,"excessive shared memory requirements, update MAX_SH_DENSE");
                  }
                  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                  #ifdef DEBUG_MXM
                  printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld (SH %d), stream %d\n",
                         rank,__FUNCTION__,i,"C_dense",GS,BS_DENSE,shmemsize,SH,i);
                  #endif
                  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		            LaunchCudaKernel(nsp_calc_C_dense<<<GS, BS_DENSE, shmemsize, bin->stream[i]>>>
                       (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,
                        d_row_perm,d_row_nz,ncols_C,h_bin_offset[i],h_bin_size[i],SH));
               }
               else{
                  // allocate A_col_offsets on the device
                  VEC_GPU<iExt> d_vec_A_col_offsets;
                  try {
                     d_vec_A_col_offsets.resize(h_bin_size[i]+1);
                  } catch (linsol_error) {
                     throw linsol_error(ERROR_INFO,"allocating d_vec_A_col_offsets");
                  }
                  iExt *d_A_col_offsets = d_vec_A_col_offsets.data();

                  // set d_A_col_offsets - to hold the offsets between the rows of A that fit the chunk bin
                  LaunchCudaKernel(set_row_size<<<1,1024, 0, bin->stream[i]>>>
                     (d_iat_A, d_row_perm,h_bin_size[i], h_bin_offset[i], d_A_col_offsets));

                  // find the cumulative sum of the entries of d_A_col_offsets using thrust
                  prefixSumExclusive(d_A_col_offsets, h_bin_size[i] + 1, 0, bin->stream[i]);

                  // copy A_bin_terms D2H
                  iExt A_bin_terms;
                  cudaError = cudaMemcpy( &A_bin_terms, &(d_A_col_offsets[h_bin_size[i]]), sizeof(iExt),
                                          cudaMemcpyDeviceToHost );
                  CheckCudaError(ERROR_INFO,"copying A_bin_terms D2H",cudaError);

                  // allocate A_bin_col on the device and initialize the array of d_A_bin_col offsets to
                  // store the end of the previous chunk (set to 0 for atomicMax)
                  VEC_GPU<iExt> d_vec_A_bin_col;
                  try {
                     d_vec_A_bin_col.assign(A_bin_terms, 0);
                  } catch (linsol_error) {
                     throw linsol_error(ERROR_INFO,"allocating d_vec_A_bin_col");
                  }
                  iExt *d_A_bin_col = d_vec_A_bin_col.data();

                  SH = MAX_SH_DENSE;
                  shmemsize = SH * (sizeof(rExt)+sizeof(unsigned char)) + (WARPSIZE + 1)*sizeof(iReg);
                  if (shmemsize > (size_t)ShMemBlk) {
                     throw linsol_error(ERROR_INFO,"excessive shared memory requirements, update MAX_SH_DENSE");
                  }
                  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                  #ifdef DEBUG_MXM
                  printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, shmemsize %5ld (SH %d), stream %d\n",
                         rank,__FUNCTION__,i,"chunk_B_dense",GS,BS_DENSE,shmemsize,SH,i);
                  #endif
                  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                  LaunchCudaKernel(nsp_calculate_value_col_chunk_B_dense<BS_DENSE,MAX_SH_DENSE>
                     <<<GS, BS_DENSE, shmemsize, bin->stream[i]>>>
                     (d_iat_A,d_ja_A,d_coef_A,d_iat_B,d_ja_B,d_coef_B,d_iat_C,d_ja_C,d_coef_C,d_row_perm,
                      d_row_nz,h_bin_offset[i],h_bin_size[i],ncols_C,d_A_col_offsets,d_A_bin_col));
                  }

               break;

            default :
               throw linsol_error(ERROR_INFO,"kernel not implemented yet");
               break;

         } // end select case

      // }

      } // end check group size

   } // end loop over groups

   // syncronize device
   cudaError = cudaDeviceSynchronize();
   CheckCudaError(ERROR_INFO,"runtime failure",cudaError);

}

//----------------------------------------------------------------------------------------
