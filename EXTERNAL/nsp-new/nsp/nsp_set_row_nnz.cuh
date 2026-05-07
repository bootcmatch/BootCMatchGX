
#pragma once

#include "nsp/prefixSumExclusive.h"

//----------------------------------------------------------------------------------------

// compute the exact size of C rows using the hash table
template<typename IDXcol>
void nsp_set_row_nnz( const iExt   * __restrict__ d_iat_A,
                      const IDXcol * __restrict__ d_ja_A,
                      const iExt   * __restrict__ d_iat_B,
                      const IDXcol * __restrict__ d_ja_B,
                            iExt   * __restrict__ d_iat_C,
                            sfBIN  * bin,
                      const iReg nrows_C,
                      const IDXcol ncols_C,
                            iExt   * __restrict__ nterm_C,
                      const iReg DIRECT ) {

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

            case 0:  // <= 64
               // Memory per thread
               SHthread = (64/8)*sizeof(IDXcol);
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
               LaunchCudaKernel(nsp_set_row_nz_bin_mpwarp<8,64><<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,h_bin_offset[i],h_bin_size[i]));
               break;

            case 1:  // <= 128
               // Memory per thread
               SHthread = (128/16)*sizeof(IDXcol);
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
               LaunchCudaKernel(nsp_set_row_nz_bin_mpwarp<16,128><<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,h_bin_offset[i],h_bin_size[i]));
               break;

            case 2 : // <= 256
               // Memory per thread
               SHthread = (256/32)*sizeof(IDXcol);
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
               LaunchCudaKernel(nsp_set_row_nz_bin_mpwarp<32,256><<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,h_bin_offset[i],h_bin_size[i]));
               break;

            case 3 : // <= 512
               // Memory per block
               shmemsize = 512*sizeof(IDXcol);
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
               LaunchCudaKernel(nsp_set_row_nz_bin_each_tb<512><<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,h_bin_offset[i]));
               break;

            case 4 : // <= 1024
               // Memory per block
               shmemsize = 1024*sizeof(IDXcol);
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
               LaunchCudaKernel(nsp_set_row_nz_bin_each_tb<1024><<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,h_bin_offset[i]));
               break;

            case 5 : // <= 2048
               // Memory per block
               shmemsize = 2048*sizeof(IDXcol);
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
               LaunchCudaKernel(nsp_set_row_nz_bin_each_tb<2048><<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,h_bin_offset[i]));
               break;

            case 6 : // <= 4096
               // Memory per block
               shmemsize = 4096*sizeof(IDXcol);
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
               LaunchCudaKernel(nsp_set_row_nz_bin_each_tb<4096><<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,h_bin_offset[i]));
               break;

            case 7 : // <= 8192
               // Memory per block
               shmemsize = 8192*sizeof(IDXcol);
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
               LaunchCudaKernel(nsp_set_row_nz_bin_each_tb<8192><<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,h_bin_offset[i]));
               break;

            case 8 : // <= 12288
               // Memory per block
               shmemsize = 12288*sizeof(IDXcol);
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
                      rank,__FUNCTION__,i,"tb_max",GS,BS,shmemsize,i);
               #endif
               //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
               LaunchCudaKernel(nsp_set_row_nz_bin_each_tb_max<12288><<<GS, BS, shmemsize, bin->stream[i]>>>
                  (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,h_bin_offset[i]));
               break;

            case 9 :
               // start scope case 9 (to allocate inside a switch)
               {
                  static int cacheConfigured = 0;
                  if (!cacheConfigured) {
                     cudaError = cudaFuncSetCacheConfig(nsp_set_row_nz_bin_each_tb_large<IDXcol>, cudaFuncCachePreferShared);
                     CheckCudaError(ERROR_INFO,"cudaFuncSetCacheConfig failed",cudaError);
                     cacheConfigured = 1;
                  }

                  // prepare auxiliary variables for large rows
                  iReg  h_fail_count = 0;
                  iReg *d_fail_count = nullptr;
                  cudaError = cudaMalloc((void **)&d_fail_count, sizeof(iReg));
                  CheckCudaError(ERROR_INFO,"allocating d_fail_count",cudaError);

                  // initialise d_fail_count to 0
                  cudaError = cudaMemcpy(d_fail_count, &h_fail_count, sizeof(iReg), cudaMemcpyHostToDevice);
                  CheckCudaError(ERROR_INFO,"copying fail_count H2D failed", cudaError);

                  VEC_GPU<iReg> d_vec_fail_perm;
                  try {
                     d_vec_fail_perm.resize(h_bin_size[i]);
                  } catch (linsol_error) {
                     throw linsol_error(ERROR_INFO,"allocating d_vec_fail_perm");
                  }
                  iReg *d_fail_perm = d_vec_fail_perm.data();

                  // Memory per block
                  // Retrieve static shared memory used in the kernel
                  cudaFuncAttributes a{};
                  cudaError = cudaFuncGetAttributes(&a, nsp_set_row_nz_bin_each_tb_large<IDXcol>);
                  CheckCudaError(ERROR_INFO,"getting nsp_set_row_nz_bin_each_tb_large attributes failed",
                                 cudaError);
                  // Compute maximum dynamic memory available
                  // SH = (ShMemBlk - a.sharedSizeBytes) / sizeof(IDXcol);
		  if(a.sharedSizeBytes==0) {
		     a.sharedSizeBytes=sizeof(iReg);
		  }
                  SH = (ShMemBlk - (a.sharedSizeBytes*maxBlkMP)) / sizeof(IDXcol);
                  shmemsize = SH * sizeof(IDXcol);
                  // Max number of blocks per MP
                  maxNblk = min(int(ShMemMP/shmemsize),maxBlkMP);
                  // Block size
                  BS = min(maxThrMP/maxNblk,maxThrBlk);
                  BS = BS<WARPSIZE ? WARPSIZE : (BS/WARPSIZE)*WARPSIZE;

#if 0		  
                  // 2026-04-24: Fix Massimo (begin)
                  // if (BS > 640) BS = 640;
                  // 2026-04-24: Fix Massimo (end)

                  // 2026-04-24: Fix Giacomo (begin)
                  //int maxBS = 0;
                  //cudaOccupancyMaxPotentialBlockSize(
                  //    nullptr,        // grid size (ignored)
                  //    &maxBS,
                  //    nsp_set_row_nz_bin_each_tb_large<IDXcol>,
                  //    shmemsize,
                  //    0               // no limit
                  //);
                  //maxBS = (maxBS / WARPSIZE) * WARPSIZE; // warp align

                  //if (maxBS == 0) { // This would cause cudaErrorInvalidConfiguration
                  // //  // permissive fallback
                  //   int maxThreadsRegs = regsPerSM / attr.numRegs;
                  //   maxThreadsRegs = (maxThreadsRegs / WARPSIZE) * WARPSIZE;
                  //   BS = min(maxThreadsRegs, maxThrBlk);
		  //
                  //   // just for safety
                  //    if (BS > 640) BS = 640;
                  //}
                  //else {
                  //    BS = min(BS, maxBS);
                  //}
                  // 2026-04-24: Fix Giacomo (end)

                  // // 2026-04-24: Fix Giacomo (begin)
                  int tryBS = maxThrBlk;
                  for (; tryBS >= WARPSIZE; tryBS -= WARPSIZE) {
                      int activeBlocks = 0;

                      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                         &activeBlocks,
                         nsp_set_row_nz_bin_each_tb_large<IDXcol>,
                         tryBS,
                         shmemsize
                      );

                      if (activeBlocks > 0) {
                         BS = tryBS;
                         break;
                      }
                  }
                  if (BS == 0) {
                      throw linsol_error(ERROR_INFO, "No valid block size found");
                  }
                  // // 2026-04-24: Fix Giacomo (begin)
#endif
//		  BS = 640;
                  // Grid size
                  GS = h_bin_size[i];
                  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                  #ifdef DEBUG_MXM
                  printf("(%2d) [%27s - case %2d (%10s)] \t GS %4d, BS %4d, SH %d, shmemsize %5ld, a.maxDynamicSharedSizeBytes %d, maxNblk %d, maxThrMP %d, maxBlkMP %d, ShMemMP %d, ShMemBlk %d, a.sharedSizeBytes %d, stream %d, numRegs per thread: %d, maxRegs: %d\n",
                         rank,__FUNCTION__,i,"tb_large",GS,BS,SH, shmemsize,a.maxDynamicSharedSizeBytes,maxNblk, maxThrMP, maxBlkMP, ShMemMP, ShMemBlk, a.sharedSizeBytes, i, a.numRegs, ChronosDevice.get_regsPerBlock());
                  #endif
                  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                  // try to compute all the rows using the standard kernel (hasmap_mod)
                  LaunchCudaKernel(nsp_set_row_nz_bin_each_tb_large<<<GS, BS, shmemsize, bin->stream[i]>>>
                     (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_row_perm,d_row_nz,d_fail_count,d_fail_perm,
                      h_bin_offset[i],h_bin_size[i],SH));

                  // copy fail_count D2H
                  cudaError = cudaMemcpy(&h_fail_count, d_fail_count, sizeof(iReg), cudaMemcpyDeviceToHost);
                  CheckCudaError(ERROR_INFO,"copying fail_count D2H failed", cudaError);

                  // check if the computation failed for some rows
                  if (h_fail_count > 0) {

                     // Memory per block (to use hasmap_bit, roundown to the nearest power of 2)
                     SH = SH ? (1ULL << (63 - __builtin_clzll((unsigned long long)SH))) : 0;
                     shmemsize = SH * sizeof(IDXcol);
                     // Grid size
                     GS = h_fail_count;
                     // compute remaining rows using the chunk kernel (hasmap_bit)
                     LaunchCudaKernel(nsp_set_row_nz_bin_each_tb_chunk<<<GS, BS, shmemsize, bin->stream[i]>>>
                        (d_iat_A,d_ja_A,d_iat_B,d_ja_B,d_fail_perm,d_row_nz,0,h_fail_count,ncols_C,SH));

                  }

                  // remove auxiliary variables for large rows
                  cudaError = cudaFree(d_fail_count);
                  CheckCudaError(ERROR_INFO,"freeing d_fail_count failed", cudaError);

               } // end scope case 9

               break;

            default :
               throw linsol_error(ERROR_INFO,"kernel not implemented yet");
               break;

         } // end select case

      } // end check group size

   } // end loop over groups

   // syncronize device
   cudaError = cudaDeviceSynchronize();
   CheckCudaError(ERROR_INFO,"runtime failure",cudaError);

   // Set row pointer of matrix C
   prefixSumExclusive(d_row_nz, nrows_C + 1, d_iat_C, 0);

   cudaError = cudaMemcpy(nterm_C, d_iat_C + nrows_C, sizeof(iExt), cudaMemcpyDeviceToHost);
   CheckCudaError(ERROR_INFO,"copying nterm_C D2H failed", cudaError);

}

//----------------------------------------------------------------------------------------
