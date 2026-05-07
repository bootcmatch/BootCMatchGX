// Rename the old sfBIN from nsparse.h so it does not conflict with the new
// sfBIN struct defined by nsp-new's nsp_bin.cuh.
// The #define must appear BEFORE nsp.h (which includes nsparse.h).
#define sfBIN nsparse_sfBIN_old

#include "nsp.h"       // → nsparse.h (#pragma once, first time) → sfCSR + nsparse_sfBIN_old defined
#undef sfBIN           // restore name: nsp_bin.cuh will now define the real new sfBIN
#undef WARP            // nsparse.h defines WARP 32 which breaks CUB template param 'int WARP'

#include "nsp_support.cuh"  // VEC_GPU, VEC_CPU, ChronosDevice, linsol_error, constants
#include "utility/memory.h" // CUDA_MALLOC, CUDA_FREE (brings in DIE via die.h)
#include <cuda_runtime.h>

// utils.h also defines getMaskByWarpID with a different signature, causing an ambiguity
// with nsp/getMaskByWarpID.cuh once the nsp-new headers are included below.
// Reproduce only the CHECK_DEVICE macro we need, without pulling in utils.h.
#define CHECK_DEVICE(X)                                                        \
    {                                                                          \
        cudaError_t _status = (X);                                             \
        if (_status != cudaSuccess)                                            \
            DIE("NSP spgemm CUDA error: %s at %s:%d\n",                       \
                cudaGetErrorString(_status), __FILE__, __LINE__);              \
    }

// ---------------------------------------------------------------------------
// Type and macro aliases expected by nsp-new kernel headers
// ---------------------------------------------------------------------------

#define myatomicAdd        atomicAdd
#define myatomicMax        atomicMax
#define myatomicCAS        atomicCAS
#define myatomicAdd_block  atomicAdd

#define HASH_SCAL   107
#define MASKFULL    0xFFFFFFFF
#define WARPSIZE    32

// iReg: column indices and bin counts (int)
// iExt: row pointers (int; same width here since nnz fits in int for our matrices)
// rExt: value type — must match 'real' from nsparse.h (double when compiled with -DDOUBLE)
#define iReg int
#define iExt int
#define rExt real

// nsp-new headers call CheckCudaError(ERROR_INFO, "msg", err) — 3 macro args,
// because the preprocessor counts ERROR_INFO as one token before expansion.
#define CheckCudaError(info, msg, cudaErr) \
    if ((cudaErr) != cudaSuccess) { throw linsol_error(info, msg); }

// ---------------------------------------------------------------------------
// nsp-new kernel headers (header-only library in EXTERNAL/nsp-new/nsp/)
// Included via -I$(NSP_PATH), so "nsp/foo.cuh" resolves to EXTERNAL/nsp-new/nsp/foo.cuh
// ---------------------------------------------------------------------------

// #define DEBUG_MXM 1

// Stubs required by the DEBUG_MXM guards in nsp-new kernel headers.
// type_MPI_int: integer type used for MPI rank (always plain int).
// Chronos: minimal object with Get_currComm() returning the communicator.
// MPI_Comm and MPI_COMM_WORLD are available via die.h → <mpi.h>.
#ifdef DEBUG_MXM
using type_MPI_int = int;
static struct {
    MPI_Comm Get_currComm() {
        return MPI_COMM_WORLD;
    }
} Chronos;
#endif

#include "nsp/hashmap_mod.cuh"
#include "nsp/nsp_bin.cuh"                              // defines new sfBIN
#include "nsp/nsp_set_intprod_num.cuh"
#include "nsp/nsp_set_row_perm.cuh"
#include "nsp/nsp_init_row_perm.cuh"
#include "nsp/nsp_set_max_bin.cuh"
#include "nsp/nsp_set_min_bin.cuh"
#include "nsp/dev_compact.cuh"
#include "nsp/set_row_size.cuh"
#include "nsp/nsp_set_row_nz_bin_mpwarp.cuh"
#include "nsp/nsp_set_row_nz_bin_each_tb.cuh"
#include "nsp/nsp_set_row_nz_bin_each_tb_large.cuh"
#include "nsp/nsp_set_row_nz_bin_each_tb_chunk.cuh"
#include "nsp/nsp_set_row_nz_bin_each_tb_max.cuh"
#include "nsp/nsp_set_row_nnz.cuh"
#include "nsp/nsp_calculate_value_col_bin_mpwarp.cuh"
#include "nsp/nsp_calculate_value_col_bin_each_tb.cuh"
#include "nsp/nsp_calc_C_dense.cuh"
#include "nsp/nsp_calculate_value_col_bin_each_tb_outsort.cuh"
#include "nsp/nsp_calculate_value_col_chunk_B_dense.cuh"
#include "nsp/nsp_calculate_value_col_bin_each_tb_chunk_dynamic.cuh"
#include "nsp/nsp_calc_val_sort_rows.cuh"
#include "nsp/nsp_calculate_value_col_bin.cuh"

// ---------------------------------------------------------------------------

void nsp_spgemm_kernel_hash(sfCSR *a, sfCSR *b, sfCSR *c) {

    iReg nrows_C = (iReg)a->M;
    iReg ncols_C = (iReg)b->N;

    iExt *d_iat_A  = a->d_rpt;
    iReg *d_ja_A   = a->d_col;
    rExt *d_coef_A = a->d_val;
    iExt *d_iat_B  = b->d_rpt;
    iReg *d_ja_B   = b->d_col;
    rExt *d_coef_B = b->d_val;

    int DIRECT = 0;

    // Initialize bin structure
    sfBIN bin;
    try {
        nsp_init_bin<iReg>(&bin, nrows_C);
    } catch (linsol_error) {
        DIE("NSP spgemm: bin initialization failed\n");
    }

    // Estimate C row sizes and populate bin
    try {
        nsp_set_max_bin(d_iat_A, d_ja_A, d_iat_B, &bin, nrows_C, DIRECT);
    } catch (linsol_error) {
        DIE("NSP spgemm: estimating C row sizes failed\n");
    }

    // Temporary row-pointer array for the symbolic phase output
    VEC_GPU<iExt> d_vec_iat_C;
    try {
        d_vec_iat_C.resize(nrows_C + 1);
    } catch (linsol_error) {
        DIE("NSP spgemm: allocating d_vec_iat_C failed\n");
    }
    iExt *d_tmp_iat_C = d_vec_iat_C.data();

    // Symbolic computation: exact per-row nnz via hash table
    iExt nterm_C = 0;
    try {
        nsp_set_row_nnz(d_iat_A, d_ja_A, d_iat_B, d_ja_B,
                        d_tmp_iat_C, &bin,
                        nrows_C, ncols_C, &nterm_C, DIRECT);
    } catch (linsol_error) {
        DIE("NSP spgemm: symbolic phase (set_row_nnz) failed: %s\n",
            cudaGetErrorString(cudaGetLastError()));
    }

    // Allocate output matrix C on device
    c->d_rpt = CUDA_MALLOC(iExt, nrows_C + 1);
    c->d_col = CUDA_MALLOC(iReg, nterm_C);
    c->d_val = CUDA_MALLOC(rExt, nterm_C);
    c->M   = (int)nrows_C;
    c->N   = (int)ncols_C;
    c->nnz = (int)nterm_C;

    iExt *d_iat_C  = c->d_rpt;
    iReg *d_ja_C   = c->d_col;
    rExt *d_coef_C = c->d_val;

    // Copy row pointers from temp buffer to C
    CHECK_DEVICE(cudaMemcpy(d_iat_C, d_tmp_iat_C, (nrows_C + 1) * sizeof(iExt), cudaMemcpyDeviceToDevice));
    d_vec_iat_C.resize(0);

    // Refine bin assignment using exact row sizes
    iReg ifrac;
    rExt frac;
    try {
        nsp_set_min_bin(&bin, nrows_C, ncols_C, DIRECT, ifrac, frac);
    } catch (linsol_error) {
        DIE("NSP spgemm: bin refinement (set_min_bin) failed\n");
    }

    // Numeric computation: fill column indices and values of C
    try {
        nsp_calculate_value_col_bin(d_iat_A, d_ja_A, d_coef_A,
                                    d_iat_B, d_ja_B, d_coef_B,
                                    d_iat_C, d_ja_C, d_coef_C,
                                    &bin, nrows_C, ncols_C, nterm_C, DIRECT);
    } catch (linsol_error) {
        DIE("NSP spgemm: numeric phase (calculate_value_col_bin) failed\n");
    }

    // Release bin
    try {
        nsp_release_bin(&bin);
    } catch (linsol_error) {
        DIE("NSP spgemm: bin release failed\n");
    }
}
