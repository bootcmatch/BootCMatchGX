#pragma once

//----------------------------------------------------------------------------------------

template <typename ind_type0, typename ind_type1, typename ind_type2, typename val_type>
__device__ __forceinline__ void hashmap_mod_count(       ind_type0 * sh_sums,
                                                         ind_type1 * check,
                                                         val_type  * value,
                                                   const ind_type2 key,
                                                   const val_type val,
                                                   const int SH_ROW ) {

   int hash = (key * HASH_SCAL) % SH_ROW;
   if (check[hash] != key) {
      ind_type1 tmp = check[hash];
      while(tmp!=key && tmp!=-1){
         hash = (hash + 1) % SH_ROW;
         tmp = check[hash];
      }
      if (tmp != key) {
         while(1){
            ind_type1 old = myatomicCAS(check + hash, -1, key);
            if (old == -1 || old == key) {
               if (old == -1) myatomicAdd_block(sh_sums, 1);
               break;
            } else hash = (hash + 1) % SH_ROW;
         }
      }
   }
   atomicAdd_block(value + hash, val);

}

//----------------------------------------------------------------------------------------

template <typename ind_type1, typename ind_type2, typename val_type>
__device__ __forceinline__ void hashmap_bit( ind_type1 * check,
                                             val_type  * value,
                                             const ind_type2 key,
                                             const val_type val,
                                             const int SH_ROW_1 ) {

   int hash = (key * HASH_SCAL) & SH_ROW_1;
   if (check[hash] != key) {
      ind_type1 tmp = check[hash];
      while(tmp!=key && tmp!=-1){
         hash = (hash + 1) & SH_ROW_1;
         tmp = check[hash];
      }
      if (tmp != key) {
         while(1){
            ind_type1 old = myatomicCAS(check + hash, -1, key);
            if (old == -1 || old == key) break;
            else hash = (hash + 1) & SH_ROW_1;
         }
      }
   }
   atomicAdd_block(value + hash, val);

}

//----------------------------------------------------------------------------------------

template <typename ind_type0, typename ind_type1, typename ind_type2>
__device__ __forceinline__ void hashmap_symbolic_bit( ind_type0 & nz,
                                                      ind_type1 * check,
                                                      const ind_type2 key,
                                                      const int SH_ROW_1 ) {

   int hash = (key * HASH_SCAL) & SH_ROW_1;
   if (check[hash] != key) {
      ind_type1 tmp = check[hash];
      while(tmp!=key && tmp!=-1){
         hash = (hash + 1) & SH_ROW_1;
         tmp = check[hash];
      }
      if (tmp != key) {
         while(1){
            ind_type1 old = myatomicCAS(check + hash, -1, key);
            if (old == -1 || old == key) {
               nz += (unsigned int)old >> 31;
               break;
            } else hash = (hash + 1) & SH_ROW_1;
         }
      }
   }

}

//----------------------------------------------------------------------------------------

template <typename ind_type0, typename ind_type1, typename ind_type2>
__device__ __forceinline__ void hashmap_symbolic_mod( ind_type0 & nz,
                                                      ind_type1 * check,
                                                      const ind_type2 key,
                                                      const int SH_ROW ) {

   int hash = (key * HASH_SCAL) % SH_ROW;
   if (check[hash] != key) {
      ind_type1 tmp = check[hash];
      while(tmp!=key && tmp!=-1){
         hash = (hash + 1) % SH_ROW;
         tmp = check[hash];
      }
      if (tmp != key) {
         while(1){
            ind_type1 old = myatomicCAS(check + hash, -1, key);
            if (old == -1 || old == key) {
               nz += (unsigned int)old >> 31;
               break;
            } else hash = (hash + 1) % SH_ROW;
         }
      }
   }

}

//----------------------------------------------------------------------------------------
