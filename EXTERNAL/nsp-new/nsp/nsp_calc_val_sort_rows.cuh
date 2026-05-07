
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
using namespace cub;

//----------------------------------------------------------------------------------------

// This version is assumed to use only one block of threads.
template<int THREADS_PER_BLOCK, int ITEMS_PER_THREAD, typename IDXcol>
__device__ __forceinline__ void _mySort_KeyVal_row( const iReg   nn,
                                                    const IDXcol begin_bit,
                                                    const IDXcol end_bit,
                                                          IDXcol * __restrict__ Key,
                                                          rExt   * __restrict__ Val,
                                                          char   * sh_mem ) {
   IDXcol K_items[ITEMS_PER_THREAD];
   rExt   V_items[ITEMS_PER_THREAD];

   using BlockRadixSortT =
      BlockRadixSort<IDXcol, THREADS_PER_BLOCK, ITEMS_PER_THREAD, rExt>;

   typename BlockRadixSortT::TempStorage* tmp_sort =
      reinterpret_cast<typename BlockRadixSortT::TempStorage*>(sh_mem);

   const IDXcol MAX_KEY = cub::Traits<IDXcol>::Max();
   const rExt   MAX_VAL = cub::Traits<rExt>::Max();

   LoadDirectBlocked(threadIdx.x, Key, K_items, nn, MAX_KEY);
   LoadDirectBlocked(threadIdx.x, Val, V_items, nn, MAX_VAL);

   __syncthreads();

   BlockRadixSortT(*tmp_sort).SortBlockedToStriped(K_items, V_items, begin_bit, end_bit);

   StoreDirectStriped<THREADS_PER_BLOCK>(threadIdx.x, Key, K_items, nn);
   StoreDirectStriped<THREADS_PER_BLOCK>(threadIdx.x, Val, V_items, nn);
}

//----------------------------------------------------------------------------------------

// Entry point for mySort_KeyVal_row
template <int BS, typename IDXcol>
__device__ void mySort_KeyVal_row( const iReg   nn,
                                   const IDXcol end_bit,
                                   IDXcol * __restrict__ Key,
                                   rExt   * __restrict__ Val,
                                   void   * sh_mem ) {

   // you can make no more than 12 items per thread since BlockRadixSortT::TempStorage takes 8 bytes
   // using block size of 512 with 12 items per thread = 512*12*8 = 49152 (max size of shared memory)

   // Select the proper function
   switch ((nn-1)/BS + 1){

      case 1:
         _mySort_KeyVal_row<BS, 1, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 2:
         _mySort_KeyVal_row<BS, 2, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 3:
         _mySort_KeyVal_row<BS, 3, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 4:
         _mySort_KeyVal_row<BS, 4, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 5:
         _mySort_KeyVal_row<BS, 5, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 6:
         _mySort_KeyVal_row<BS, 6, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 7:
         _mySort_KeyVal_row<BS, 7, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 8:
         _mySort_KeyVal_row<BS, 8, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 9:
         _mySort_KeyVal_row<BS, 9, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 10:
         _mySort_KeyVal_row<BS,10, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 11:
         _mySort_KeyVal_row<BS,11, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      case 12:
         _mySort_KeyVal_row<BS,12, IDXcol>(nn,0,end_bit,Key,Val,(char*)sh_mem);
         break;

      default:
         if (nn > blockDim.x && threadIdx.x == 0)
            printf("Vettore troppo grande %d > BLKSZ %d; case = %d, BS = %d\n",nn,blockDim.x,(nn-1)/BS + 1,BS);
         break;

   }

}

//----------------------------------------------------------------------------------------

template <int BS, typename IDXcol>
__global__ void nsp_calc_val_sort_rows( const IDXcol nSigBits,
                                        const iReg   SH_ROW,
                                        const iReg   bin_offset,
                                        const iReg * __restrict__ row_perm,
                                        const iExt * __restrict__ iat,
                                        IDXcol * __restrict__ ja,
                                        rExt   * __restrict__ coef){

   extern __shared__ char csh_mem[];

   // Retrieve thread infos
   iReg rid = (row_perm == nullptr) ? blockIdx.x : row_perm[blockIdx.x + bin_offset];
   iReg offset;
   iReg nz;

   // offset and nonzero of the row
   offset = iat[rid];
   nz = iat[rid + 1] - offset;

   // sort the rows
   mySort_KeyVal_row<BS>(nz,nSigBits,(IDXcol*) &(ja[offset]),&(coef[offset]),csh_mem);

}

//----------------------------------------------------------------------------------------

// Counts how many significant bits are necessary to represent nn
template<typename IDXcol>
inline IDXcol countBITS( const IDXcol nn ) {

   IDXcol k = 1;
   IDXcol imax = 2;
   while (nn >= imax){
      imax *= 2;
      k++;
   }
   return k;

}

//----------------------------------------------------------------------------------------
