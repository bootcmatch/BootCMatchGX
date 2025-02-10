#include "CSRVector_product_adaptive_miniwarp_splitted.h"

// #include "halo_communication/extern.h"
#include "halo_communication/extern2.h"
#include "halo_communication/halo_communication.h"
#include "utility/memory.h"
#include "utility/profiling.h"

#define SYNCSOL_TAG_RECV 4321
#define SYNCSOL_TAG_SEND 9876
#define MAXNTASKS 1024
MPI_Request mpkrequests[MAXNTASKS];
int mpkntr = -1;


/**
 * @brief Synchronizes and updates the global vector based on local and received data.
 *
 * This kernel updates the global vector `x` based on the local vector `local_x`
 * and the data received from other processes. It handles the merging of local
 * and received data into the global vector.
 *
 * @param local_x Pointer to the local vector.
 * @param local_n The number of elements in the local vector.
 * @param what_to_receive_d Pointer to the data received from other processes.
 * @param receive_n The number of elements to receive.
 * @param post_local The number of elements in the global vector that are local.
 * @param x Pointer to the global vector that will be updated.
 * @param x_n The total number of elements in the global vector.
 */
__global__ void _vector_sync_splitted(vtype* local_x, itype local_n, vtype* what_to_receive_d, itype receive_n, itype post_local, vtype* x, itype x_n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < x_n) {
        if (id < post_local) {
            x[id] = what_to_receive_d[id];
        } else {
            if (id < post_local + local_n) {
                x[id] = local_x[id - post_local];
            } else {
                x[id] = what_to_receive_d[id - local_n];
            }
        }
    }
}

/**
 * @brief Performs an adaptive matrix-vector product with halo communication.
 *
 * This function computes the product of a sparse matrix `A` and a local vector `local_x`,
 * while handling communication of halo data between processes. It updates the result in
 * the vector `w` based on the specified parameters.
 *
 * @param A Pointer to the sparse matrix in CSR format.
 * @param local_x Pointer to the local input vector.
 * @param w Pointer to the output vector where the result will be stored.
 * @param degree The current degree of communication.
 * @param maxdegree The maximum allowed degree of communication.
 * @param alpha The scalar multiplier for the matrix-vector product.
 * @param beta The scalar multiplier for the output vector.
 * @return vector<vtype>* Pointer to the output vector containing the result.
 */
vector<vtype>* CSRVector_product_adaptive_miniwarp_splitted(CSR* A, vector<vtype>* local_x, vector<vtype>* w, int degree, int maxdegree, vtype alpha, vtype beta)
{
    _MPI_ENV;
    BEGIN_PROF(__FUNCTION__);

    static cudaStream_t mpk_stream;
    static int first = 1;

    if (nprocs == 1) {
        vector<vtype>* w_ = NULL;
        if (w == NULL) {
            w_ = Vector::init<vtype>(A->n, true, true);
            Vector::fillWithValue(w_, 0.);
        } else {
            w_ = w;
        }
        CSRm::CSRVector_product_adaptive_miniwarp(A, local_x, w_, alpha, beta);
        END_PROF(__FUNCTION__);
        return (w_);
    }
    halo_info hi = A->halo;
    if (first) {
        first = 0;
        mpk_stream = *(A->os.streams->local_stream);
    }
    assert(A->shrinked_flag == 1);

    CSR* A_ = CSRm::init(A->n, A->shrinked_m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);
    A_->row = A->row;
    A_->val = A->val;
    A_->col = A->shrinked_col;

    // ----------------------------------------- temp check -----------------------------------------
    if (A->halo.to_receive_n + local_x->n != A_->m) {
        printf("ERROR: Myid %d"
               /*", level %d, File %s, Line %d"*/
               ": A->halo.to_receive_n + local_x->n = %d + %d = %d != %lu = A_->m (A->m = %lu, A->n = %d, A->full_n = %lu)\n",
               myid,
               /*pico_info.level, pico_info.file, pico_info.line,*/
               A->halo.to_receive_n, local_x->n, A->halo.to_receive_n + local_x->n, A_->m, A->m, A->n, A->full_n);
        printf("[%d] A->halo.to_receive_n = %d, A->halo.to_send_n = %d\n", myid, A->halo.to_receive_n, A->halo.to_send_n);
    }
    assert(A->halo.to_receive_n + local_x->n == A_->m);
    // ----------------------------------------------------------------------------------------------

    int post_local = A->post_local;

    vector<vtype>* x_ = NULL;
    if (A->halo.to_receive_n > 0) {
        x_ = Vector::init<vtype>(A_->m, false, true);
        if (A_->m > xsize) {
            if (xsize > 0) {
                CUDA_FREE(xvalstat);
            }
            xsize = A_->m;
            xvalstat = CUDA_MALLOC(vtype, xsize, true);
        }
        x_->val = xvalstat;
        GridBlock gb = gb1d(A_->m, BLOCKSIZE);
        _vector_sync_splitted<<<gb.g, gb.b>>>(local_x->val, A->n, A->halo.what_to_receive_d, A->halo.to_receive_d->n, post_local, x_->val, x_->n);
    } else {
        x_ = local_x;
    }

    vector<vtype>* w_ = NULL;
    if (w == NULL) {
        w_ = Vector::init<vtype>(A->n, true, true);
    } else {
        w_ = w;
    }

    if ((hi.to_send_n || hi.to_receive_n) && degree < maxdegree) {
        int j = 0, ntr = 0;
        if (hi.to_send_n) {
            itype n = A->n;
            int density = A->nnz / A->n;
            int min_w_size;
            if (density < MINI_WARP_THRESHOLD_2) {
                min_w_size = 2;
            } else if (density < MINI_WARP_THRESHOLD_4) {
                min_w_size = 4;
            } else if (density < MINI_WARP_THRESHOLD_8) {
                min_w_size = 8;
            } else {
                min_w_size = 16;
            }

#if SMART_AGGREGATE_GETSET_GPU == 1
            GridBlock gb = gb1d(hi.to_send_n, BLOCKSIZE, true, min_w_size);
            gb = gb1d(n, BLOCKSIZE, true, min_w_size);
            if (alpha == 1. && beta == 0.) {
                CSRm::_CSR_vector_mul_mini_warp<1><<<gb.g, gb.b, 0, mpk_stream>>>(n, min_w_size, alpha, beta, A_->val, A_->row, A_->col, x_->val, w_->val);
            } else if (alpha == -1. && beta == 1.) {
                CSRm::_CSR_vector_mul_mini_warp<2><<<gb.g, gb.b, 0, mpk_stream>>>(n, min_w_size, alpha, beta, A_->val, A_->row, A_->col, x_->val, w_->val);
            } else {
                CSRm::_CSR_vector_mul_mini_warp<0><<<gb.g, gb.b, 0, mpk_stream>>>(n, min_w_size, alpha, beta, A_->val, A_->row, A_->col, x_->val, w_->val);
            }
#else
            fprintf(stderr, "NO SUPPORTED!\n");
            exit(1);
#endif
        }
        if (hi.to_receive_n) {
            for (int t = 0; t < nprocs; t++) {
                if (t == myid) {
                    continue;
                }
                if (hi.to_receive_counts[t] > 0) {
                    CHECK_MPI(
                        MPI_Irecv(hi.what_to_receive + (hi.to_receive_spls[t]), hi.to_receive_counts[t], VTYPE_MPI, t, SYNCSOL_TAG_SEND, MPI_COMM_WORLD, mpkrequests + j));
                    j++;
                    if (j == MAXNTASKS) {
                        fprintf(stderr, "Too many tasks in sync_solution, max is %d\n", MAXNTASKS);
                        exit(1);
                    }
                }
            }
            ntr = j;
            mpkntr = ntr;
        }
        cudaStreamSynchronize(mpk_stream);
    } else {
        CSRm::CSRVector_product_adaptive_miniwarp(A_, x_, w_, alpha, beta);
    }

    // --------------------------------------- print -----------------------------------------
    //   vector<vtype> *what_to_receive_d = Vector::init<vtype>(A->halo.to_receive_n, false, true);
    //   what_to_receive_d->val = A->halo.what_to_receive_d;
    //
    //   PICO_PRINT(  \
    //     fprintf(fp, "A->halo:\n\tto_receive: "); Vector::print(A->halo.to_receive, -1, fp); \
    //     fprintf(fp, "\tto_send: "); Vector::print(A->halo.to_send, -1, fp); \
    //     fprintf(fp, "post_local = %d\n", post_local); \
    //     fprintf(fp, "what_to_receive_d: "); Vector::print(what_to_receive_d, -1, fp); \
    //     fprintf(fp, "local_x: "); Vector::print(local_x, -1, fp); \
    //     fprintf(fp, "x_: "); Vector::print(x_, -1, fp); \
    //   )
    //
    //   FREE(what_to_receive_d);
    // ---------------------------------------------------------------------------------------

    if (A->halo.to_receive_n > 0) {
        FREE(x_);
    }
    A_->col = NULL;
    A_->row = NULL;
    A_->val = NULL;
    FREE(A_);

    END_PROF(__FUNCTION__);
    return (w_);
}
