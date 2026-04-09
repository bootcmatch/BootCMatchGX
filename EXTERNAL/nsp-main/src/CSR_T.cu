template <typename Tst, typename Tgs, typename Ti>
  CSRtype<Tst,Tgs,Ti>* init_CSR(Tst n, Tgs m, Tst nnz, bool allocate_mem, bool on_the_device, bool is_symmetric, Tgs full_n, Tgs row_shift=0){

  #if CUDAMALLOCCNTON
      if (allocate_mem && on_the_device) {
          FUNCTIONCALL_CNT( cudamalloc_cnt, "CSRm::Init", "Total CSRm::Init" )
      }
  #endif
  //   assert(n > 0 && m > 0 && nnz >= 0);
    
  // ---------- Pico ----------
    if ( n<=0 || m <=0 || nnz <=0){
        fprintf(stderr, "error in CSRm::init:\n\tint  n: %lu  m: %lu  nnz: %lu\n\tunsigned  n: %lu  m: %lu  nnz: %lu\n",
        (unsigned long)n, (unsigned long)m, (unsigned long)nnz, (unsigned long)n, (unsigned long)m, (unsigned long)nnz);
        //printf("error in CSRm::init:\n\tint  n: %d  m: %d  nnz: %d\n\tunsigned  n: %u  m: %u  nnz: %u\n", n, m, nnz, n, m, nnz);
        //fflush(stdout);
    }
    assert(n > 0);
    assert(m > 0);
    assert(nnz > 0);
  // --------------------------
    
    CSRtype<Tst,Tgs,Ti> *A = NULL;

    // on the host
    A = (CSRtype<Tst,Tgs,Ti>*) malloc(sizeof(CSRtype<Tst,Tgs,Ti>));
    CHECK_HOST(A);

    A->nnz = nnz;
    A->n = n;
    A->m = m;
    A->full_m = m; // NOTE de sostituire cambiando CSRm::Init inserendo full_m

    A->on_the_device = on_the_device;
    A->is_symmetric = false;
    A->custom_alloced = false;

    A->full_n = full_n;
    A->row_shift = row_shift;

    A->rows_to_get = NULL;
    
    //Ti shrinked_firstrow = 0;
    //Ti shrinked_lastrow  = 0;
    A->shrinked_flag = false;
    A->shrinked_col = NULL;
    A->shrinked_m = m;
    A->halo.init = false;
    A->col_shifted=0;
    
    A->post_local = 0;
    A->bitcolsize = 0;
    A->bitcol = NULL;

    if(allocate_mem){
      if(on_the_device){
        // on the device
        cudaError_t err;
        err = cudaMalloc( (void**) &A->val, nnz * sizeof(vtype) );
        CHECK_DEVICE(err);
        err = cudaMalloc( (void**) &A->col, nnz * sizeof(Ti) );
        CHECK_DEVICE(err);
        err = cudaMalloc( (void**) &A->row, (n + 1) * sizeof(Ti) );
        CHECK_DEVICE(err);
      }else{
        // on the host
        A->val = (vtype*) malloc( nnz * sizeof(vtype) );
        CHECK_HOST(A->val);
        A->col = (Ti*) malloc( nnz * sizeof(Ti) );
        CHECK_HOST(A->col);
        A->row = (Ti*) malloc( (n + 1) * sizeof(Ti) );
        CHECK_HOST(A->row);
      }
    }

    cusparseMatDescr_t *descr = NULL;
    descr = (cusparseMatDescr_t*) malloc( sizeof(cusparseMatDescr_t) );
    CHECK_HOST(descr);

    cusparseStatus_t  err = cusparseCreateMatDescr(descr);
    CHECK_CUSPARSE(err);

    cusparseSetMatIndexBase(*descr, CUSPARSE_INDEX_BASE_ZERO);

    if(is_symmetric)
      cusparseSetMatType(*descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    else
      cusparseSetMatType(*descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    A->descr = descr;
    return A;
  }


template <typename Tst, typename Tgs, typename Ti>
void free(CSRtype<Tst,Tgs,Ti>* A){
  if(A->on_the_device){
    if (A->custom_alloced == false) {
        cudaError_t err;
        err = cudaFree(A->val);
        CHECK_DEVICE(err);
        err = cudaFree(A->col);
        CHECK_DEVICE(err);
        err = cudaFree(A->row);
        CHECK_DEVICE(err);
        if (A->shrinked_col != NULL) {
            err = cudaFree(A->shrinked_col);
            CHECK_DEVICE(err);
        }
    }
  }else{
    std::free(A->val);
    std::free(A->col);
    std::free(A->row);
  }
  CHECK_CUSPARSE( cusparseDestroyMatDescr(*A->descr) );
  if (A->rows_to_get != NULL){
    std::free( A->rows_to_get->rcvprow);
    std::free( A->rows_to_get->whichprow);
    std::free( A->rows_to_get->rcvpcolxrow);
    std::free( A->rows_to_get->scounts);
    std::free( A->rows_to_get->displs);
    std::free( A->rows_to_get->displr);
    std::free( A->rows_to_get->rcounts2);
    std::free( A->rows_to_get->scounts2);
    std::free( A->rows_to_get->displs2);
    std::free( A->rows_to_get->displr2);
    std::free( A->rows_to_get->rcvcntp);
    std::free( A->rows_to_get->P_n_per_process);
    if (A->rows_to_get->nnz_per_row_shift != NULL){
        Vector::free(A->rows_to_get->nnz_per_row_shift);
    }
    std::free( A->rows_to_get );
  }
  
  // Free the halo_info halo halo_info halo; 
  std::free(A->descr);
  std::free(A);
}