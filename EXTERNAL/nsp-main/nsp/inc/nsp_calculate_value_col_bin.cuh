// compute C
void nsp_calculate_value_col_bin(int *d_iat_A, int *d_ja_A, double *d_coef_A,
                                 int *d_iat_B, int *d_ja_B, double *d_coef_B,
                                 int *d_iat_C, int *d_ja_C, double *d_coef_C,
                                 sfBIN *bin, int nrows_C, int ncols_C,
                                 int nterm_C, int DIRECT) {
   // set handles
   int *h_bin_offset   = bin->h_bin_offset;
   int *h_bin_size     = bin->h_bin_size;
   int *d_row_perm     = (DIRECT) ? nullptr : bin->d_row_perm;
   int *d_row_nz       = bin->d_row_nz;

   // define varibles for GPU resources
   int GS,BS,SH;
   size_t shmemsize;

   #if DEBUG
   printf("d_iat_A = %p,\n"
            "d_ja_A = %p,\n"
            "d_coef_A = %p,\n"
            
            "d_iat_B = %p,\n"
            "d_ja_B = %p,\n"
            "d_coef_B = %p,\n"

            "d_iat_C = %p,\n"
            "d_ja_C = %p,\n"
            "d_coef_C = %p,\n"
            
            "d_row_perm = %p,\n"
            "d_row_nz = %p,\n"
            
            "h_bin_offset = %p,\n"
            "h_bin_size = %p\n",
         
      d_iat_A, d_ja_A, d_coef_A,
      d_iat_B, d_ja_B, d_coef_B,
      d_iat_C, d_ja_C, d_coef_C,
      
      d_row_perm, d_row_nz,
      h_bin_offset, h_bin_size);
   #endif

   // loop over groups
   for (int i = BIN_NUM - 1; i >= 0; i--) {
      // check sizes
      if (h_bin_size[i] > 0) {
         // select group kernel
         switch (i) {
            case 0: // <= 16
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = BS * 16 / 4;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               #if DEBUG
               printf("i = %d, h_bin_offset[i] = %d, h_bin_size[i] = %d\n", i, h_bin_offset[i], h_bin_size[i]);
               #endif
               LOG_KERNEL(nsp_calculate_value_col_bin_mpwarp<4,16><<<h_bin_size[i]/(BS/4)+1, BS, shmemsize, bin->stream[i]>>>
                                                (d_iat_A,d_ja_A,d_coef_A,
                                                 d_iat_B,d_ja_B,d_coef_B,
                                                 d_iat_C,d_ja_C,d_coef_C,
                                                 d_row_perm,//d_row_nz,nrows_C,
                                                 h_bin_offset[i],h_bin_size[i]));
               break;
            case 1: // <= 32
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = BS * 32 / 8;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               LOG_KERNEL(nsp_calculate_value_col_bin_mpwarp<8,32><<<h_bin_size[i]/(BS/8)+1, BS, shmemsize, bin->stream[i]>>>
                                                (d_iat_A,d_ja_A,d_coef_A,
                                                 d_iat_B,d_ja_B,d_coef_B,
                                                 d_iat_C,d_ja_C,d_coef_C,
                                                 d_row_perm,//d_row_nz,nrows_C,
                                                 h_bin_offset[i],h_bin_size[i]));
               break;
            case 2 : // <= 64
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = BS * 64 / 16;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               LOG_KERNEL(nsp_calculate_value_col_bin_mpwarp<16,64><<<h_bin_size[i]/(BS/16)+1, BS, shmemsize, bin->stream[i]>>>
                                                (d_iat_A,d_ja_A,d_coef_A,
                                                 d_iat_B,d_ja_B,d_coef_B,
                                                 d_iat_C,d_ja_C,d_coef_C,
                                                 d_row_perm,//d_row_nz, nrows_C,
                                                 h_bin_offset[i],h_bin_size[i]));
               break;

            case 3 : // <= 128   
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               SH = 128 * BS / 32;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_warp<128><<<h_bin_size[i]/(BS/32)+1, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,
                                                   h_bin_offset[i],h_bin_size[i]);
               break;

            case 4 : // <= 256
               #if CC == 86
                  BS = 96;
               #else
                  BS = 64;
               #endif
               GS = h_bin_size[i];
               SH = 256;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb<256,64,2><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,h_bin_offset[i]);
               break;

            case 5 : // <= 512
               BS = 128;
               GS = h_bin_size[i];
               SH = bin->SHTB_cmp_max / 8;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb<512,128,4><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,h_bin_offset[i]);
               break;

            case 6 : // <= 1024
               BS = 256;
               GS = h_bin_size[i];
               SH = bin->SHTB_cmp_max / 4;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb<1024,256,8><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,h_bin_offset[i]);
               break;

            case 7 : // <= 2048
               BS = 512;
               GS = h_bin_size[i];
               SH = bin->SHTB_cmp_max / 2;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb<2048,512,16><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,h_bin_offset[i]);
               break;

            case 8 : // <= 4096
               BS = 512;
               GS = h_bin_size[i];
               SH = bin->SHTB_cmp_max;
               shmemsize = SH * ( sizeof(int) + sizeof(double) );
               nsp_calculate_value_col_bin_each_tb_outsort<4096><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                  (d_iat_A,d_ja_A,d_coef_A,
                                                   d_iat_B,d_ja_B,d_coef_B,
                                                   d_iat_C,d_ja_C,d_coef_C,
                                                   d_row_perm,d_row_nz,
                                                   h_bin_offset[i],h_bin_size[i]);

               // sort the arrays on global memory using cub library
               {int nSigBits;
               nSigBits = countBITS(ncols_C);         
               nsp_calc_val_sort_rows<<<GS,512,shmemsize,bin->stream[i]>>>(nSigBits,d_row_perm,h_bin_offset[i],
                                                                          d_iat_C,d_ja_C,d_coef_C);  
               }

               break;

            case 9 :
               BS = 512;
               GS = h_bin_size[i];
               SH = 4096-512;         // the hash table size must be a multiple of the blocksize, in addition, new compaction takes extra 33*sizeof(int) entries
               shmemsize = 49152;     // max shared memory size   

               nsp_calculate_value_col_bin_each_tb_chunk_dynamic<3584><<<GS, BS, shmemsize, bin->stream[i]>>>
                                                        (d_iat_A,d_ja_A,d_coef_A,
                                                         d_iat_B,d_ja_B,d_coef_B,
                                                         d_iat_C,d_ja_C,d_coef_C,
                                                         d_row_perm,d_row_nz,
                                                         h_bin_offset[i],h_bin_size[i],
                                                         ncols_C);
               
               break;

            case 10 :
               BS = 512;
               GS = h_bin_size[i];
    
               if (ncols_C <= MAX_SH_DENSE) {
                  // unfortunately, we have to use the size of shared table as the multiple of the blocksize. Otherwise, the shared table initialization cannot succeed
                  SH = ((ncols_C - 1) / BS + 1) * BS;
                  shmemsize = SH * (sizeof(double)+sizeof(unsigned char)) + (WARPSIZE + 1)*sizeof(int);
                  
		           nsp_calc_C_dense<<<GS, BS, shmemsize, bin->stream[i]>>>
                                                   (d_iat_A,d_ja_A,d_coef_A,
                                                    d_iat_B,d_ja_B,d_coef_B,
                                                    d_iat_C,d_ja_C,d_coef_C,
                                                    d_row_perm,d_row_nz,ncols_C,
                                                    h_bin_offset[i],h_bin_size[i],SH);
               }
               else{
                  int *d_A_bin_col, *d_A_col_offsets, A_bin_terms;  
                  //cudaError_t cudaError;

                  //cudaError = cudaMalloc((void **)&(d_A_col_offsets), (h_bin_size[i]+1)*sizeof(int));
                  //CheckCudaError("nsp_calculate_value_col_bin","allocating d_A_col_offsets",cudaError);
				  checkCudaErrors(cudaMalloc((void **)&(d_A_col_offsets), (h_bin_size[i]+1)*sizeof(int)));
                  
                  // set d_A_col_offsets - to hold the offsets between the rows of A that fit the chunk bin
                  set_row_size<<<1,1024, 0, bin->stream[i]>>>(d_iat_A, d_row_perm,h_bin_size[i], h_bin_offset[i], d_A_col_offsets);

                  // find the cumulative sum of the entries of d_A_col_offsets using thrust
                  prefixSumExclusive(d_A_col_offsets, h_bin_size[i],0);
      
                  cudaMemcpy( &A_bin_terms, &(d_A_col_offsets[h_bin_size[i]]), sizeof(int), cudaMemcpyDeviceToHost );
               
                  //cudaError = cudaMalloc((void **)&(d_A_bin_col), A_bin_terms*sizeof(int));
                  //CheckCudaError("nsp_calculate_value_col_bin","allocating d_A_bin_col",cudaError);
				  checkCudaErrors(cudaMalloc((void **)&(d_A_bin_col), A_bin_terms*sizeof(int)));
                  
                  // initialize the array of d_A_bin_col offsets to store the end of the previous chunk
                  cudaMemset(d_A_bin_col, 0, A_bin_terms*sizeof(int)); // for atomic MAX, alternatively we could use atomic min
                  #if CC == 86
                     #define bs 768
                     #define sh 5376
                  #else
                     #define bs 1024
                     #define sh 5120
                  #endif

                  shmemsize = sh * (sizeof(double)+sizeof(unsigned char)) + (WARPSIZE + 1)*sizeof(int);
                  
                  nsp_calculate_value_col_chunk_B_dense<bs,sh><<<GS, bs, shmemsize, bin->stream[i]>>>
                                                                     (d_iat_A,d_ja_A,d_coef_A,
                                                                      d_iat_B,d_ja_B,d_coef_B,
                                                                      d_iat_C,d_ja_C,d_coef_C,
                                                                      d_row_perm,h_bin_offset[i],ncols_C,
                                                                      d_A_col_offsets,d_A_bin_col);

                  //cudaFree(d_A_bin_col);
                  //cudaFree(d_A_col_offsets);
                  cudaFreeAsync(d_A_bin_col,bin->stream[i]);
                  cudaFreeAsync(d_A_col_offsets,bin->stream[i]);
                  }
               break;

            default :
			   printf("nsp_calculate_value_col_bin -- kernel not implemented yet");
               //throw linsol_error ("nsp_calculate_value_col_bin","kernel not implemented yet");          
               break;

         } // end select case
      // }
      } // end check group size
   } // end loop over groups
   // syncronize device
   cudaDeviceSynchronize();
}
