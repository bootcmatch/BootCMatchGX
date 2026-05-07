
//----------------------------------------------------------------------------------------

// Compute the number of non-zero terms in C rows using TB/ROW and shared memory
// Additional check on failed rows
//#define MAX_THREADS_PER_BLOCK 512
//#define MIN_BLOCKS_PER_MP 1
template <typename IDXcol>
__global__ void 
//__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
nsp_set_row_nz_bin_each_tb_large( const iExt   * __restrict__ iat_A,
                                                  const IDXcol * __restrict__ ja_A,
                                                  const iExt   * __restrict__ iat_B,
                                                  const IDXcol * __restrict__ ja_B,
                                                  const iReg   * __restrict__ row_perm,
                                                        iReg   * __restrict__ row_nz,
                                                        iReg   * __restrict__ fail_count,
                                                        iReg   * __restrict__ fail_perm,
                                                  const iReg bin_offset,
                                                  const iReg nrows_tb,
                                                  const iReg SH_ROW ) {

   // retrieve thread infos
   const iReg tid  = threadIdx.x & (WARPSIZE - 1);
   const iReg wid  = threadIdx.x / WARPSIZE;
   const iReg wnum = blockDim.x / WARPSIZE;

   // Local row index (with permutation if needed)
   const iReg rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];

   // registers
   iReg jr;
   iExt je,ke;
   IDXcol jcol_A;
   IDXcol key,hash,old;
   iReg count;
   iReg border; 
   iReg dest;

   // block shared memory
   extern __shared__ IDXcol sh_mem[];
   IDXcol *check = (IDXcol*) sh_mem;
   // __shared__ iReg snz[1];
   iReg *snz = (iReg*)(check + SH_ROW);

   // initialize hash table
   for (jr = threadIdx.x; jr < SH_ROW; jr += blockDim.x) {
       check[jr] = -1;
   }
   if (threadIdx.x == 0) snz[0] = 0;

   // block synchronization to ensure check initialization
   __syncthreads();

   // initialize registers
   count = 0;
   border = SH_ROW >> 1;

   // loop over A-row coefficients
   for (je = iat_A[rid] + wid; je < iat_A[rid + 1]; je += wnum) {

      // load from global memory without using the cache
      jcol_A = load_glob(ja_A + je);

      // loop over B-row coefficients
      for (ke = iat_B[jcol_A] + tid; ke < iat_B[jcol_A + 1]; ke += WARPSIZE) {

         // load from global memory using the cache
         key = ja_B[ke];
         hash = (key * HASH_SCAL) % SH_ROW;

         // put the key inside the hash table
         while (count < border && snz[0] < border) {
            
            if (check[hash] == key) {
               // key already added
               break; 
            } else if (check[hash] == -1) {
               // add the key
               old = myatomicCAS(check + hash, -1, key);
               if (old == -1) {
                  myatomicAdd(snz,1);
                  break;
               }
            } else if (check[hash] != key) {
               // find a free place to add the key
               hash = (hash + 1) % SH_ROW;
               count++;
            }
         }
   
         // check fail: gone outside hash
         if (count >= border || snz[0] >= border) break;

      } // end loop over B-row coefficients

      // check fail: gone outside hash
      if (count >= border || snz[0] >= border) break;

   } // end loop over A-row coefficients

   // block syncronization
   __syncthreads();

   // check compuatation fail
   if (count >= border || snz[0] >= border) {
      // store failed row index
      if (threadIdx.x == 0) {
         dest = myatomicAdd(fail_count, 1);
         fail_perm[dest] = rid;
      }
   } else {
      // store row non-zeros
      if (threadIdx.x == 0) {
         row_nz[rid] = snz[0];
      }
   }

}

//----------------------------------------------------------------------------------------
