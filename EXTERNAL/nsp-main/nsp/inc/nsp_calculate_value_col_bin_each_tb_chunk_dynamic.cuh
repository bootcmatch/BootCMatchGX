#define mymax(a,b) ((a)>(b)?(a):(b))

// this barrier should be at least blockDim.x less than the hash table size in order to not have an if - break statement whithin the while loop
#define barrier_sh_tb 3072

template <int SH_ROW>
__global__ void nsp_calculate_value_col_bin_each_tb_chunk_dynamic(const int *iat_A,const int *ja_A,const double *coef_A,
                                                         const int * __restrict__ iat_B,const int * __restrict__ ja_B,const double * __restrict__ coef_B,
                                                         const int *iat_C, int *ja_C, double *coef_C,
                                                         const int *row_perm, int *row_nz,
                                                         const int bin_offset,const int nrows_tb,
                                                         const int ncols_C){

   /* 
      - this kernel is aimed at sparse and big size rows. The idea is to use a smaller number of chunks to process a row of C. If the row is dense
      then this kernel is not efficient, instead use nsp_calculate_value_col_chunk_B_dense.cuh. The density to use this kernel is set to 30%, however,
      it should be configured more precisely.
      
      - while hashing we count the number of elements added to the hash table and we initialize the column_length (maximal number of B columns traversed). 
         When the hash table becomes overfull - we decrease column_length
      --------------------------------------------------------------------------------------------------------------------------------------------------
      - make compaction with atomic addition to increase the hash table size
   */

   // retrieve thread infos
   int rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   int tid  = threadIdx.x & (WARPSIZE - 1);
   int wid  = threadIdx.x / WARPSIZE;
   int wnum = blockDim.x / WARPSIZE;

   // registers
   int jr;
   int je,ke;
   int ichunk;
   int jcol_A,jcol_B;
   double cval_A,cval_B;
   int istrB = 0,iendB;
   int key;
   int offset;
   int nz;
   int column_length;

   // block shared memory
   extern __shared__ int sh_mem[];
   int *check = (int*) sh_mem;
   double *value = (double*) (&sh_mem[SH_ROW]);
   int *sh_sums = (int*) &(value[SH_ROW]);

   // initialize hash tablenz
   #pragma unroll 
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
       value[jr] = 0.;
   }

   // initialize number of non zeros
   if (threadIdx.x == 0 ) {
      // row_nz[rid] = 0;
      sh_sums[32] = 0;
   }

   // block synchronization to ensure check initialization
   __syncthreads();

   // get term offset
   offset = iat_C[rid];

   // start loop over B chunks
   ichunk = 0;
   istrB  = 0;

   //                   (                min number of chunks                    )
   column_length  = mymax(ncols_C / ( ((iat_C[rid+1] - offset - 1) / SH_ROW ) + 1 ) / 2 ,SH_ROW) ;

   iendB = column_length;

   while (1) {
      // loop over A-row coefficients
      for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {
         // load from global memory without using the cache
         jcol_A = load_glob(ja_A + je);
         cval_A = load_glob(coef_A + je);

         // get the offset for a term of the row of A that correponds to the row of B

         // loop over row terms of B
         for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {
            // load from global memory using the cache
            jcol_B = ja_B[ke];
            // check end of the chunk
            if (jcol_B >= iendB ) break;
            if (jcol_B >= istrB){
               cval_B = coef_B[ke];
               key  = jcol_B - istrB;
               hashmap_mod_count(sh_sums+32,check, value, key, cval_A * cval_B, SH_ROW);
            } // end check column index
            if (sh_sums[32] > barrier_sh_tb && column_length > SH_ROW) break;
         } // end loop over row terms of B
         if (sh_sums[32] > barrier_sh_tb && column_length > SH_ROW) break;
      } // end loop over A-row coefficients

      // Thread-Block synchronization
      __syncthreads();
      
      // check if the hash table is not overfull, else rerun with a smaller column_length
      if (sh_sums[32] > barrier_sh_tb && column_length > SH_ROW){
         iendB -= column_length;
         column_length /= 2;
         column_length = mymax(SH_ROW,column_length);
         iendB += column_length;

         goto table_init;
      }

      //                NEW COMPACTION
      // compact the hash table and store the number of nonzero into sh_sums
      dev_compactKeyVal_inplace(SH_ROW,sh_sums,check,value,sh_sums,istrB);
      __syncthreads();
      nz = *sh_sums;

      // Sorting for shared data and copy to global memory
      for (jr = threadIdx.x; jr < nz; jr += blockDim.x) {
         key = check[jr];
         int count = 0;
         for (int kr = 0; kr < nz; kr++) {
            count += (unsigned int)(check[kr] - key) >> 31;
         }
           ja_C[offset + count] = key;
         coef_C[offset + count] = value[jr];
      }
      __syncthreads();  
      
      // check end of loop over B chunks
      if ( iendB >= ncols_C ) break;

      ichunk++;

      // update B indeces
      istrB  = iendB;
      iendB += column_length;

      // update term offset
      offset += nz;

      table_init: ;

      // initialize shared scratch
      #pragma unroll 
      for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
         check[jr] = -1;
         value[jr] = 0.;
      }
      if (threadIdx.x == 0) sh_sums[32] = 0;

      // synchronize before next chunk cycle
      __syncthreads();

   } // end loop over B chunks

}

#undef barrier_sh_tb 
