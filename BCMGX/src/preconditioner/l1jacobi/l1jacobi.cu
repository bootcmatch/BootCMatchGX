#include "halo_communication/halo_communication.h"
#include "op/addAbsoluteRowSumNoDiag.h"
#include "op/basic.h"
#include "op/diagScal.h"
#include "op/mydiag.h"
#include "preconditioner/l1jacobi/l1jacobi.h"
#include "preconditioner/prec_setup.h"

#define VERBOSE 0
#define USE_M 1

#define DEBUG_JACOBI 0
#if DEBUG_JACOBI
int jacobi_counter = 0;
#endif

// =============================================================================

template <int OP_TYPE, int MINI_WARP_SIZE>
__global__ void _jacobi_it_partial_new(itype n, itype* rows, vtype relax_weight, vtype* A_val, itype* A_row, itype* A_col, vtype* D, vtype* u, vtype* f, vtype* u_, gstype shift, itype An, gstype* halo_index, itype halo_index_size, vtype* halo_val, bool balc_flag, int post_local, int halo_val_size, int taskid)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    // only local rows
    warp = rows[warp];

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    vtype T_i = 0.;

    // A * u
    unsigned itype offset = balc_flag ? 0 : post_local; /* must be done only for the "internal part" */
    for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
        T_i += A_val[j] * __ldg(&halo_val[A_col[j] - offset]);
    }

// WARP sum reduction
#pragma unroll MINI_WARP_SIZE
    for (int k = MINI_WARP_SIZE >> 1; k > 0; k = k >> 1) {
        T_i += __shfl_down_sync(warp_mask, T_i, k);
    }

    if (lane == 0) {
        u_[warp] = ((-T_i + f[warp]) / D[warp]) + u[warp];
    }
}

static int relax_xsize = 0;
static vtype* relax_xvalstat = NULL;

#define MAXNTASKS 4096
#define JACOBI_TAG 1234
#define SYNCSOL_TAG_SEND 9876

extern int cntrelax;
extern MPI_Request mpkrequests[];
extern int mpkntr;

__global__ void _relax_sync(vtype* local_x, itype local_n, vtype* what_to_receive_d, itype receive_n, itype post_local, vtype* x, itype x_n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < x_n) {
        if (id < post_local) {
            if (id > receive_n) {
                printf("Horror 1 in thread %d\n", id);
            }
            x[id] = what_to_receive_d[id];
        } else {
            if (id < post_local + local_n) {
                if ((id - post_local) < 0 || (id - post_local) > local_n) {
                    printf("Horror 2 in thread %d %d\n", id, post_local);
                }
                x[id] = local_x[id - post_local];
            } else {
                if ((id - local_n) < 0 || (id - local_n) > receive_n) {
                    printf("Horror 3 in thread %d local_n=%d receive_n=%d x_n=%d post_local=%d\n", id, local_n, receive_n, x_n, post_local);
                }
                x[id] = what_to_receive_d[id - local_n];
            }
        }
    }
}

template <int MINI_WARP_SIZE>
vector<vtype>* internal_jacobi_overlapped(cublasHandle_t cublas_h, int k, CSR* A, vector<vtype>* u, vector<vtype>** u_, vector<vtype>* f, vector<vtype>* D, vtype relax_weight)
{
    _MPI_ENV;

    GridBlock gb;
    static MPI_Request requests[MAXNTASKS];
    static int ntr = 0;
    vector<vtype>* ret = NULL;

    if (A->os.loc_n == 0 && A->os.needy_n == 0) {
        setupOverlapped(A);
    }

    overlapped os = A->os;
    halo_info hi = A->halo;

    if (A->shrinked_flag == false) {
        fprintf(stderr, "A must be shrinked before relaxation!\n");
        exit(1);
    }

    int post_local = A->post_local;
    vector<vtype>* x_ = NULL;

    if (VERBOSE > 1) {
        vtype tnrm = Vector::norm(cublas_h, u);
        std::cout << "Before jacobi iter XTent " << tnrm << "\n";
        std::cout << "niter " << k << "\n";
    }

    vector<vtype>* orig_u = u;
    vector<vtype>* orig_u_ = *u_;
    for (int i = 0; i < k; i++) {
        cudaStreamSynchronize(*(os.streams->comm_stream));

        // start get to send
        {
            if (hi.to_send_n) {
                assert(hi.what_to_send != NULL);
                assert(hi.what_to_send_d != NULL);
                GridBlock gb = gb1d(hi.to_send_n, BLOCKSIZE);
                _getToSend_new<<<gb.g, gb.b, 0, *(os.streams->comm_stream)>>>(hi.to_send_d->n, u->val, hi.what_to_send_d, hi.to_send_d->val, A->row_shift);
                CHECK_DEVICE(cudaMemcpyAsync(hi.what_to_send, hi.what_to_send_d, hi.to_send_n * sizeof(vtype), cudaMemcpyDeviceToHost, *(os.streams->comm_stream)));
            }
        }

        x_ = u;

        if (os.loc_n) {
            // start compute local
            gb = gb1d(os.loc_n, BLOCKSIZE, true, MINI_WARP_SIZE);
            _jacobi_it_partial_new<0, MINI_WARP_SIZE><<<gb.g, gb.b, 0, *(os.streams->local_stream)>>>(
                os.loc_n,
                os.loc_rows->val,
                relax_weight,
                A->val, A->row, A->shrinked_col,
                D->val,
                u->val,
                f->val,
                (*u_)->val,
                A->row_shift,
                A->n,
                hi.to_receive_d->val,
                hi.to_receive_d->n,
                x_->val,
                false,
                post_local, /*, for debugging March 13 2024 */
                x_->n,
                myid);
        }

        int j = 0;
        if (mpkntr < 0) {
            for (int t = 0; t < nprocs; t++) {
                if (t == myid) {
                    continue;
                }
                if (hi.to_receive_counts[t] > 0) {
                    CHECK_MPI(
                        MPI_Irecv(hi.what_to_receive + (hi.to_receive_spls[t]), hi.to_receive_counts[t], VTYPE_MPI, t, JACOBI_TAG, MPI_COMM_WORLD, requests + j));
                    j++;
                    if (j == MAXNTASKS) {
                        fprintf(stderr, "Too many tasks in jacobi, max is %d\n", MAXNTASKS);
                        exit(1);
                    }
                }
            }
            ntr = j;
        }

        if (hi.to_send_n) {
            cudaStreamSynchronize(*(os.streams->comm_stream));
        }

        for (int t = 0; t < nprocs; t++) {
            if (t == myid) {
                continue;
            }
            if (hi.to_send_counts[t] > 0) {
                if (mpkntr < 0) {
                    CHECK_MPI(MPI_Send(hi.what_to_send + (hi.to_send_spls[t]), hi.to_send_counts[t], VTYPE_MPI, t, JACOBI_TAG, MPI_COMM_WORLD));
                } else {
                    CHECK_MPI(MPI_Send(hi.what_to_send + (hi.to_send_spls[t]), hi.to_send_counts[t], VTYPE_MPI, t, SYNCSOL_TAG_SEND, MPI_COMM_WORLD));
                }
            }
        }

        // copy received data
        if (hi.to_receive_n) {
            if (mpkntr < 0) {
                if (ntr > 0) {
                    CHECK_MPI(MPI_Waitall(ntr, requests, MPI_STATUSES_IGNORE));
                }
            } else {
                CHECK_MPI(MPI_Waitall(mpkntr, mpkrequests, MPI_STATUSES_IGNORE));
                mpkntr = -1;
            }
            assert(hi.what_to_receive != NULL);
            assert(hi.what_to_receive_d != NULL);
            CHECK_DEVICE(cudaMemcpyAsync(hi.what_to_receive_d, hi.what_to_receive, hi.to_receive_n * sizeof(vtype), cudaMemcpyHostToDevice, *(os.streams->comm_stream)));

            x_ = Vector::init<vtype>(A->shrinked_m, false, true);
            if (A->shrinked_m > relax_xsize) {
                if (relax_xsize > 0) {
                    CHECK_DEVICE(cudaFree(relax_xvalstat));
                }
                relax_xsize = A->shrinked_m;
                cudaError_t err = cudaMalloc(&relax_xvalstat, sizeof(vtype) * relax_xsize);
                CHECK_DEVICE(err);
            }

            x_->val = relax_xvalstat;
            GridBlock gb = gb1d(A->shrinked_m, BLOCKSIZE);
            _relax_sync<<<gb.g, gb.b, 0, *(os.streams->comm_stream)>>>(u->val, A->n, hi.what_to_receive_d, hi.to_receive_d->n, post_local, x_->val, x_->n);

            // complete computation for halo
            gb = gb1d(os.needy_n, BLOCKSIZE, true, MINI_WARP_SIZE);
            if (os.needy_n) {
                // start compute local
                _jacobi_it_partial_new<0, MINI_WARP_SIZE><<<gb.g, gb.b, 0, *(os.streams->comm_stream)>>>(
                    os.needy_n,
                    os.needy_rows->val,
                    relax_weight,
                    A->val, A->row, A->shrinked_col,
                    D->val,
                    u->val,
                    f->val,
                    (*u_)->val,
                    A->row_shift,
                    A->n,
                    hi.to_receive_d->val,
                    hi.to_receive_d->n,
                    x_->val,
                    true,
                    post_local, /* ,for debugging March 13 2024 */
                    x_->n,
                    myid);
            }
        }

        cudaStreamSynchronize(*(os.streams->local_stream));
        cudaStreamSynchronize(*(os.streams->comm_stream));

        ret = *u_;
        vector<vtype>* swap_temp;
        swap_temp = u;
        u = *u_;
        *u_ = swap_temp;

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(cublas_h, u);
            std::cout << "After jacobi iter " << i << " XTent " << tnrm << "\n";
        }
    }

    cudaStreamSynchronize(*(os.streams->local_stream));
    cudaStreamSynchronize(*(os.streams->comm_stream));

    if (hi.to_receive_n > 0) {
        std::free(x_);
    }

    // #if DEBUG_JACOBI
    //     dump(u, "%s/%s_%04d_%04d_%s_u_p%d%s.txt",
    //         output_dir.c_str(), output_prefix.c_str(),
    //         jacobi_counter, __LINE__, __func__,
    //         myid, output_suffix.c_str()
    //     );
    //     dump(*u_, "%s/%s_%04d_%04d_%s_uu_p%d%s.txt",
    //         output_dir.c_str(), output_prefix.c_str(),
    //         jacobi_counter, __LINE__, __func__,
    //         myid, output_suffix.c_str()
    //     );
    //     jacobi_counter++;
    // #endif

    u = orig_u;
    *u_ = orig_u_;
    return ret;
}

vector<vtype>* jacobi_adaptive_miniwarp_overlapped(
    handles* h,
    int k, CSR* A, vector<vtype>* u,
    vector<vtype>** u_,
    vector<vtype>* f,
    vector<vtype>* D,
    vtype relax_weight)
{
    _MPI_ENV;

    cublasHandle_t cublas_h = h->cublas_h;

    int density = A->nnz / A->n;

    vector<vtype>* ret = NULL;
    if (density < MINI_WARP_THRESHOLD_2) {
        ret = internal_jacobi_overlapped<2>(cublas_h, k, A, u, u_, f, D, relax_weight);
    } else if (density < MINI_WARP_THRESHOLD_4) {
        ret = internal_jacobi_overlapped<4>(cublas_h, k, A, u, u_, f, D, relax_weight);
    } else if (density < MINI_WARP_THRESHOLD_8) {
        ret = internal_jacobi_overlapped<4>(cublas_h, k, A, u, u_, f, D, relax_weight);
    } else if (density < MINI_WARP_THRESHOLD_16) {
        ret = internal_jacobi_overlapped<16>(cublas_h, k, A, u, u_, f, D, relax_weight);
    } else {
        ret = internal_jacobi_overlapped<32>(cublas_h, k, A, u, u_, f, D, relax_weight);
    }

    // #if DEBUG_JACOBI
    //     dump(u, "%s/%s_%04d_%04d_%s_u_p%d%s.txt",
    //         output_dir.c_str(), output_prefix.c_str(),
    //         jacobi_counter, __LINE__, __func__,
    //         myid, output_suffix.c_str()
    //     );
    //     dump(*u_, "%s/%s_%04d_%04d_%s_uu_p%d%s.txt",
    //         output_dir.c_str(), output_prefix.c_str(),
    //         jacobi_counter, __LINE__, __func__,
    //         myid, output_suffix.c_str()
    //     );
    //     dump(ret, "%s/%s_%04d_%04d_%s_ret_p%d%s.txt",
    //         output_dir.c_str(), output_prefix.c_str(),
    //         jacobi_counter, __LINE__, __func__,
    //         myid, output_suffix.c_str()
    //     );
    //     jacobi_counter++;
    // #endif

    return ret;
}

template <int OP_TYPE, int MINI_WARP_SIZE>
__global__ void _jacobi_it_full(itype n, vtype relax_weight, vtype* A_val, itype* A_row, itype* A_col, vtype* D, vtype* u, vtype* f, vtype* u_)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;

    if (warp >= n) {
        return;
    }

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    vtype T_i = 0.;

    // A * u
    for (int j = A_row[warp] + lane; j < A_row[warp + 1]; j += MINI_WARP_SIZE) {
        T_i += A_val[j] * __ldg(&u[A_col[j]]);
    }

// WARP sum reduction
#pragma unroll MINI_WARP_SIZE
    for (int k = MINI_WARP_SIZE >> 1; k > 0; k = k >> 1) {
        T_i += __shfl_down_sync(warp_mask, T_i, k);
    }

    if (lane == 0) {
        if (OP_TYPE == 0) {
            u_[warp] = ((-T_i + f[warp]) / D[warp]) + u[warp];
        } else if (OP_TYPE == 1) {
            u_[warp] = (-T_i / D[warp]) + u[warp];
        }
    }
}

template <int MINI_WARP_SIZE>
vector<vtype>* internal_jacobi_coarsest(int k, CSR* A, vector<vtype>* u, vector<vtype>** u_, vector<vtype>* f, vector<vtype>* D, vtype relax_weight)
{
    GridBlock gb = gb1d(A->n, BLOCKSIZE, true, MINI_WARP_SIZE);

    assert(f != NULL);

    vector<vtype>* ret = NULL;
    vector<vtype>* orig_u = u;
    vector<vtype>* orig_u_ = *u_;
    for (int i = 0; i < k; i++) {
        _jacobi_it_full<0, MINI_WARP_SIZE><<<gb.g, gb.b>>>(A->n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
        ret = *u_;
        vector<vtype>* swap_temp = u;
        u = *u_;
        *u_ = swap_temp;
    }
    u = orig_u;
    *u_ = orig_u_;

    return ret;
}

vector<vtype>* jacobi_adaptive_miniwarp_coarsest(handles* h, int k, CSR* A, vector<vtype>* u, vector<vtype>** u_, vector<vtype>* f, vector<vtype>* D, vtype relax_weight)
{
    _MPI_ENV;

    assert(f != NULL);

    int density = A->nnz / A->n;

    vector<vtype>* ret = NULL;
    if (density < MINI_WARP_THRESHOLD_2) {
        ret = internal_jacobi_coarsest<2>(k, A, u, u_, f, D, relax_weight);
    } else if (density < MINI_WARP_THRESHOLD_4) {
        ret = internal_jacobi_coarsest<4>(k, A, u, u_, f, D, relax_weight);
    } else if (density < MINI_WARP_THRESHOLD_8) {
        ret = internal_jacobi_coarsest<8>(k, A, u, u_, f, D, relax_weight);
    } else if (density < MINI_WARP_THRESHOLD_16) {
        ret = internal_jacobi_coarsest<16>(k, A, u, u_, f, D, relax_weight);
    } else {
        ret = internal_jacobi_coarsest<32>(k, A, u, u_, f, D, relax_weight);
    }

    return ret;
}

void l1jacobi_iter(handles* h,
    int k,
    CSR* A,
    vector<vtype>* D,
    vector<vtype>* f, // rhs
    vector<vtype>* u, // Xtent
    vector<vtype>** u_ // Xtent2
)
{
    _MPI_ENV;

    if (VERBOSE > 1) {
        vtype unrm = Vector::norm(h->cublas_h, u);
        vtype fnrm = Vector::norm(h->cublas_h, f);
        vtype Dnrm = Vector::norm(h->cublas_h, D);
#if USE_M
        fprintf(stdout, "P%d: in my_relax k %d u %lf f %lf M %lf rw %lf\n", myid, k, unrm, fnrm, Dnrm, 1.);
#else
        fprintf(stdout, "P%d: in my_relax k %d u %lf f %lf D %lf rw %lf\n", myid, k, unrm, fnrm, Dnrm, 1.);
#endif
    }

    vector<vtype>* ret = NULL;
    if (nprocs == 1) {
        // printf("Calling jacobi_adaptive_miniwarp_coarsest\n");
        ret = jacobi_adaptive_miniwarp_coarsest(h, k, A, u, u_, f, D, 1.);
    } else {
        // printf("Calling jacobi_adaptive_miniwarp_overlapped\n");
        ret = jacobi_adaptive_miniwarp_overlapped(h, k, A, u, u_, f, D, 1.);
    }

#if DEBUG_JACOBI
    dump(u, "%s/%s_%04d_%04d_%s_u_p%d%s.txt",
        output_dir.c_str(), output_prefix.c_str(),
        jacobi_counter, __LINE__, __func__,
        myid, output_suffix.c_str());
    dump(*u_, "%s/%s_%04d_%04d_%s_uu_p%d%s.txt",
        output_dir.c_str(), output_prefix.c_str(),
        jacobi_counter, __LINE__, __func__,
        myid, output_suffix.c_str());
    dump(ret, "%s/%s_%04d_%04d_%s_ret_p%d%s.txt",
        output_dir.c_str(), output_prefix.c_str(),
        jacobi_counter, __LINE__, __func__,
        myid, output_suffix.c_str());
    jacobi_counter++;
#endif

    if (u != ret) {
        Vector::copyTo(u, ret);
        // Vector::copyTo(u, ret, (nprocs>1)?*(A->os.streams->comm_stream):0);
    }
}

/*----------------------------------------------------------------------------------
 * Given the local A returns the local vector of the diagonal of l1-jacobi smoother
 *---------------------------------------------------------------------------------*/
void set_l1j(CSR* Alocal, vector<vtype>* pl1j_loc)
{
    _MPI_ENV;

    // D^-1: pl1j_loc per ogni riga i di A contiene la somma dell'elemento
    // diagonale della riga i + i valori assoluti di tutti gli elementi extra-diagonali

    mydiag(Alocal, pl1j_loc);
    addAbsoluteRowSumNoDiag(Alocal, pl1j_loc);
}

void l1jacobi_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p)
{
    pr->l1jacobi.pl1j = Vector::init<vtype>(Alocal->n, true, true);
    pr->l1jacobi.w_loc = Vector::init<vtype>(Alocal->n, true, true);
    pr->l1jacobi.rcopy_loc = Vector::init<vtype>(Alocal->n, true, true);

    set_l1j(Alocal, pr->l1jacobi.pl1j);
}

void l1jacobi_finalize(CSR* Alocal, cgsprec* pr, const params& p)
{
    Vector::free(pr->l1jacobi.pl1j);
    if (p.l1jacsweeps % 2 == 0) {
        Vector::free(pr->l1jacobi.w_loc);
    }
    Vector::free(pr->l1jacobi.rcopy_loc);
}

void l1jacobi_apply(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, cgsprec* pr, const params& p, PrecOut* out)
{
    CHECK_DEVICE(cudaMemcpy(pr->l1jacobi.rcopy_loc->val, r_loc->val, r_loc->n * sizeof(vtype), cudaMemcpyDeviceToDevice));
    l1jacobi_iter(h,
        p.l1jacsweeps,
        Alocal,
        pr->l1jacobi.pl1j,
        pr->l1jacobi.rcopy_loc, // rhs
        u_loc, // Xtent
        &pr->l1jacobi.w_loc); // Xtent2
}
