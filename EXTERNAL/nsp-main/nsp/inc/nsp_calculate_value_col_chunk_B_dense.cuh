// Compute C rows using TB/ROW and shared memory. B and C are divided into column chunks.
template <const int BS,const int SH_ROW>
__global__ void nsp_calculate_value_col_chunk_B_dense(const int *iat_A, const int *ja_A, const double *coef_A,
                                                      const int *iat_B, const int *ja_B, const double *coef_B,
                                                      int *iat_C, int *ja_C, double *coef_C,
                                                      const int *row_perm, const int bin_offset,
                                                      const int ncols_C,
                                                      const int *d_A_col_offsets, int *d_A_col_chunks) {

   /*
      since we parse the rows of B in chunks size of the shared table, we can use the bitmask as "check" array
      - at each iteration of loop over B terms we store the last visited index of B added into the hash table.
      - in the next chunk iteraion use the offset to go directly to the non visited elements of B.
      - you can use less memory: ideally, the scratch space needed is the number of blocks per SM times the size of the biggest rows of A
      - be careful to not exceed the GPU global memory

      this kernel is configured to use 30 registers when int == int. If you use const __restrict__ then 32
   */

   // retrieve thread infos
   int rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE; 
   int wnum = BS / WARPSIZE;

   // registers
   extern __shared__ int sh_mem[];
   unsigned char *check = (unsigned char*)sh_mem;
   double *value = (double*) &(check[SH_ROW]);
   int *sh_sums = (int*)&(value[SH_ROW]);

   // initialize shared table and bitmask
   #pragma unroll 
   for (int jr = threadIdx.x; jr < SH_ROW; jr += BS) {
      value[jr] = 0.;
      check[jr] = 0x00;
   }

   // block synchronization to ensure check initialization
   __syncthreads();

   // get the offset between different rows of C
   int offset = iat_C[rid];

   // get the offset between different rows of A
   int A_row_offset = d_A_col_offsets[blockIdx.x]; // you can delete it together with B_col_offset if you have a direct access 

   // start loop over B chunks
   int istrB  = 0;
   int iendB  = SH_ROW;

   while (1) {

      // loop over A-row coefficients
      for (int je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

         // load from global memory without using the cache
         int jcol_A = load_glob(  ja_A + je);
         double    cval_A = load_glob(coef_A + je);

         // get the offset for a term of the row of A that correponds to the row of B
         int B_col_offset = A_row_offset + je - iat_A[rid];
         int iat_B_ind = iat_B[jcol_A];
         
         // loop over row terms of B
         for (int ke = d_A_col_chunks[B_col_offset] + iat_B_ind + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {

            // load from global memory using the cache
            int jcol_B = ja_B[ke];

            // check end of the chunk
            if ( jcol_B >= iendB ) break;
            else{
               int key = jcol_B - istrB;
               double cval_B = coef_B[ke];

               // add the keys to bitmask and update the values in shared table
               if(key<0) {
                  printf("Horror in key %d, jcol_B=%d, iendB=%d,SH_ROW=%d,\n",key,jcol_B,iendB,SH_ROW);
               }

               check[key] = 0x01;
               atomicAdd(value + key, cval_A * coef_B[ke]);
               
               // update the B column offsets
               atomicMax((long long int *)(d_A_col_chunks + B_col_offset), ke - iat_B_ind + 1);
         
            } // end check column index

         } // end loop over row terms of B

      } // end loop over A-row coefficients

      // thread-block synchronization
      __syncthreads(); 
      
      // compact the shared table
      dev_compactVal_chunk_dense(SH_ROW, sh_sums+32,check, value,sh_sums,istrB,coef_C + offset, ja_C + offset, tid, wid, wnum);

      // thread-block synchronization
      __syncthreads();
     
      // update the offset by adding the number of nonzeros of the previous chunk
      offset += sh_sums[32];

      // check end of loop over B chunks
      if ( iendB >= ncols_C ) break;

      // update B indices
      istrB  = iendB;
      iendB += SH_ROW;

      // initialize the shared table and the bitmask
      #pragma unroll 
      for (int jr = threadIdx.x; jr < SH_ROW; jr += BS) {
         value[jr] = 0.;
         check[jr] = 0x00;
      }

      // synchronize before the next chunk cycle
      __syncthreads();

   } // end loop over B chunks

}
