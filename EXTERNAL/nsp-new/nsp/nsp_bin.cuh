
//----------------------------------------------------------------------------------------

// Data structure for nsparse
struct sfBIN {

   // Stream for each group
   cudaStream_t* stream = nullptr;

   // Number of nnz of each row of C
   VEC_GPU<iReg> d_row_nz;

   // Size (number of rows) of each group
   VEC_CPU<iReg> h_bin_size;
   VEC_GPU<iReg> d_bin_size;

   // Offset of each group
   VEC_CPU<iReg> h_bin_offset;
   VEC_GPU<iReg> d_bin_offset;

   // Permutation array (based on d_row_z)
   VEC_GPU<iReg> d_row_perm;

   // Maximum nnz that can be stored in shared memory
   int max_nnz_shared_symb;
   int max_nnz_shared_comp;

};

//----------------------------------------------------------------------------------------

// Initialize the sfBIN data structure
template <typename IDXcol>
void nsp_init_bin( sfBIN *bin, const iReg nrows_C ) {

   // Initilize error flag
   cudaError_t cudaError;

   // Retrieve GPU resources
   long int ShMemMP  = ChronosDevice.get_sharedMemPerMultiprocessor();
   long int ShMemBlk = ChronosDevice.get_sharedMemPerBlock();

   // Set the maximum number of entries that can fit in a ja-hash
   int max_entries_symb = static_cast<int>(min(ShMemMP,ShMemBlk) / sizeof(IDXcol));
   int max_bin_size_symb = 64;
   while (max_bin_size_symb < max_entries_symb) max_bin_size_symb *= 2;
   if (max_bin_size_symb > max_entries_symb) max_bin_size_symb /= 2;
   bin->max_nnz_shared_symb = max_bin_size_symb;

   // Set the maximum number of entries that can fit in a (ja,coef)-hash
   int max_entries_comp = static_cast<int>(min(ShMemMP,ShMemBlk) / (sizeof(IDXcol) + sizeof(rExt)));
   int max_bin_size_comp = 64;
   while (max_bin_size_comp < max_entries_comp) max_bin_size_comp *= 2;
   if (max_bin_size_comp > max_entries_comp) max_bin_size_comp /= 2;
   bin->max_nnz_shared_comp = max_bin_size_comp;

   // Allocating streams
   bin->stream = (cudaStream_t *) malloc(sizeof(cudaStream_t) * BIN_NUM);
   for (iReg i = 0; i < BIN_NUM; i++) {
      cudaError = cudaStreamCreate(&(bin->stream[i]));
      CheckCudaError(ERROR_INFO,"cudaStreamCreate failed",cudaError);
   }
   if ( bin->stream == nullptr ) throw linsol_error(ERROR_INFO,"stream");

   // Allocate host members
   try {
      bin->h_bin_size.resize(BIN_NUM);
      bin->h_bin_offset.resize(BIN_NUM);
   } catch (linsol_error) {
      throw linsol_error(ERROR_INFO,"allocating host members");
   }

   // Allocate device members
   try {

      // Alloc an extra entry to make prefixExclusiveSum pass cuda-memcheck
      bin->d_row_nz.assign(nrows_C+1, 0);

      bin->d_bin_size.resize(BIN_NUM);
      bin->d_bin_offset.resize(BIN_NUM);
      bin->d_row_perm.resize(nrows_C);

   } catch (linsol_error) {
      throw linsol_error(ERROR_INFO,"allocating device members");
   }

}

//----------------------------------------------------------------------------------------

// Release the sfBIN data structure
void nsp_release_bin(sfBIN *bin) {

   cudaError_t cudaError;

   // Destroy streams
   for (iReg i = 0; i < BIN_NUM; i++) {
      cudaError = cudaStreamDestroy(bin->stream[i]);
      CheckCudaError(ERROR_INFO,"cudaStreamDestroy failed",cudaError);
   }
   free(bin->stream);

}

//----------------------------------------------------------------------------------------

// distribution of rows in groups for symbolic computation
__global__ void nsp_set_bin( const iReg * __restrict__ row_nz,
                                   iReg * __restrict__ bin_size,
                             const iReg nrows_C,
                             const int  max_nnz_shared) {

   // retrieve row index
   iReg rid = blockIdx.x * blockDim.x + threadIdx.x;

   if (rid >= nrows_C) return;

   // registers
   // printf("nsp_set_bin: HASH_FAC_SYMB=%d\n",HASH_FAC_SYMB);

   iReg nz_per_row = HASH_FAC_SYMB*row_nz[rid];

   if (nz_per_row <= max_nnz_shared){

      if (nz_per_row <= 64){
         // mpwarp
         myatomicAdd(bin_size+0,1);
      } else if (nz_per_row <= 128){
         // mpwarp
         myatomicAdd(bin_size+1,1);
      } else if (nz_per_row <= 256){
         // mpwarp
         myatomicAdd(bin_size+2,1);
      } else if (nz_per_row <= 512){
         // tb
         myatomicAdd(bin_size+3,1);
      } else if (nz_per_row <= 1024){
         // tb
         myatomicAdd(bin_size+4,1);
      } else if (nz_per_row <= 2048){
         // tb
         myatomicAdd(bin_size+5,1);
      } else if (nz_per_row <= 4096){
         // tb
         myatomicAdd(bin_size+6,1);
      } else if (nz_per_row <= 8192){
         // tb
         myatomicAdd(bin_size+7,1);
      } else if (nz_per_row <= 12288){
         // tb_max
         myatomicAdd(bin_size+8,1);
      } else {
         // tb_chunk
         myatomicAdd(bin_size+9,1);
      }

   } else {
      // tb_chunk
      myatomicAdd(bin_size+9,1);
   }

}

//----------------------------------------------------------------------------------------

// distribution of rows in groups for calculation of values
__global__ void nsp_set_bin_min( const iReg * __restrict__ row_nz,
                                       iReg * __restrict__ bin_size,
                                 const iReg nrows_C,
                                 const iReg ncols_C,
                                 const int  max_nnz_shared) {

   // retrieve row index
   iReg rid = blockIdx.x * blockDim.x + threadIdx.x;

   if (rid >= nrows_C) return;

   // registers
   iReg nz_per_row = HASH_FAC_COMP*row_nz[rid];
   rExt density = (rExt)nz_per_row / ncols_C * 100.;

   if (nz_per_row <= max_nnz_shared){

      if ( (ncols_C <= MAX_SH_DENSE && density > MIN_DENSITY) || (density > MIN_DENSITY_chunk) ){
         // dense bin
         myatomicAdd(bin_size+10,1);
      } else if (nz_per_row <= 16){
         // mpwarp
         myatomicAdd(bin_size+0,1);
      } else if (nz_per_row <= 32){
         // mpwarp
         myatomicAdd(bin_size+1,1);
      } else if (nz_per_row <= 64){
         // mpwarp
         myatomicAdd(bin_size+2,1);
      } else if (nz_per_row <= 128){
         // warp
         myatomicAdd(bin_size+3,1);
      } else if (nz_per_row <= 256){
         // tb
         myatomicAdd(bin_size+4,1);
      } else if (nz_per_row <= 512){
         // tb
         myatomicAdd(bin_size+5,1);
      } else if (nz_per_row <= 1024){
         // tb
         myatomicAdd(bin_size+6,1);
      } else if (nz_per_row <= 2048){
         // tb
         myatomicAdd(bin_size+7,1);
      } else if (nz_per_row <= 4096){
         // tb_outsort
         myatomicAdd(bin_size+8,1);
      } else {
         // tb_chunk
         myatomicAdd(bin_size+9,1);
      }

   } else {
      // tb_chunk
      myatomicAdd(bin_size+9,1);
   }

}

//----------------------------------------------------------------------------------------
