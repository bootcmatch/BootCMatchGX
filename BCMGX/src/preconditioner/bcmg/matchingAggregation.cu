#include "halo_communication/halo_communication.h"
#include "op/spspmpi.h"
#include "preconditioner/bcmg/matchingAggregation.h"
#include "preconditioner/bcmg/matchingPairAggregation.h"
#include "utility/memory.h"
#include "utility/profiling.h"

#define FTCOARSE_INC 100
#define COARSERATIO_THRSLD 1.2

int MUL_NUM = 0;
int I = 0;

itype* iPtemp1;
vtype* vPtemp1;
itype* iAtemp1;
vtype* vAtemp1;
itype* idevtemp1;
vtype* vdevtemp1;
itype* idevtemp2;
// --------- TEST ----------
itype* dev_rcvprow_stat;
vtype* completedP_stat_val;
itype* completedP_stat_col;
itype* completedP_stat_row;
// -------- AH glob --------
itype* AH_glob_row;
itype* AH_glob_col;
vtype* AH_glob_val;
// -------------------------
int* buffer_4_getmct;
int sizeof_buffer_4_getmct = 0;
unsigned int* idx_4shrink;
bool alloced_idx = false;
// ------ cuCompactor ------
int* glob_d_BlocksCount;
int* glob_d_BlocksOffset;
// -------------------------

void relaxPrepare(handles* h, int level, CSR* A, hierarchy* hrrch, buildData* amg_data)
{
    BEGIN_PROF(__FUNCTION__);

    RelaxType relax_type = amg_data->CRrelax_type;

    if (relax_type == RelaxType::L1_JACOBI) {
        // L1 smoother
        if (hrrch->D_array[level] != NULL) {
            Vector::free(hrrch->D_array[level]);
        }
        hrrch->D_array[level] = CSRm::diag(A);

        if (hrrch->M_array[level] != NULL) {
            Vector::free(hrrch->M_array[level]);
        }
        hrrch->M_array[level] = CSRm::absoluteRowSum(A, NULL);
    }

    END_PROF(__FUNCTION__);
}

vector<itype>* makePCol_CPU(vector<itype>* mask, itype* ncolc)
{
    BEGIN_PROF(__FUNCTION__);
    vector<itype>* col = Vector::init<itype>(mask->n, true, false);

    for (itype v = 0; v < mask->n; v++) {
        itype u = mask->val[v];
        if ((u >= 0) && (v != u) && (v < u)) {
            col->val[v] = ncolc[0];
            col->val[u] = ncolc[0];
            ncolc[0]++;
        }
    }

    for (itype v = 0; v < mask->n; v++) {
        if (mask->val[v] == -2) {
            col->val[v] = ncolc[0] - 1;
        } else if (mask->val[v] == -1) {
            col->val[v] = ncolc[0];
            ncolc[0]++;
        }
    }

    END_PROF(__FUNCTION__);
    return col;
}

__global__ void __setPsRow4prod(itype n, itype* row, itype nnz, itype start, itype stop)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    if (i < start) {
        row[i] = 0;
    }

    if (i > stop) {
        row[i] = nnz;
    }
}

CSR* matchingAggregation(handles* h, buildData* amg_data, CSR* A, vector<vtype>** w, CSR** P, CSR** R, int level)
{
    BEGIN_PROF(__FUNCTION__);

    _MPI_ENV;

    CSR *Ai_ = A, *Ai = NULL;

    CSR* Ri_ = NULL;
    vector<vtype>*wi_ = *w, *wi = NULL;

    double size_coarse, size_precoarse;
    double coarse_ratio;

    for (int i = 0; i < amg_data->sweepnumber; i++) {
        CSR* Pi_;

        matchingPairAggregation(h, amg_data, Ai_, wi_, &Pi_, &Ri_, (i == 0)); /* routine with the real work. It calls the suitor procedure */

        // BEGIN_PROF("MUL");

        // --------------- PICO ------------------
        CSR* AP = nsparseMGPU_commu_new(h, Ai_, Pi_, false);
        CSRm::shift_cols(Ri_, -AP->row_shift);
        Ri_->col_shifted = -AP->row_shift;
        Ai = nsparseMGPU_noCommu_new(h, Ri_, AP);
        if (myid != 0 && Ai->col_shifted == 0) {
            CSRm::shift_cols(Ai, -(Ai->row_shift));
            Ai->col_shifted = -(Ai->row_shift);
        }
#if DEBUG
        if (myid != 0 && AP->col_shifted == 0) {
            CSRm::shift_cols(AP, -(AP->row_shift));
            AP->col_shifted = -(AP->row_shift);
        }
        CSRm::printMM(Ri_, MName);
        CSRm::printMM(Ai, AiName);
        CSRm::printMM(AP, APName);
#endif

        // END_PROF("MUL");
        MUL_NUM += 2;

        // ---------------------------------------

        // BEGIN_PROF("OTHER");
        CSRm::free(AP);
        // END_PROF("OTHER");

        wi = Vector::init<vtype>(Ai->n, true, true);

        // BEGIN_PROF("SHIFTED_CSRVEC");
        CSRm::shifted_CSRVector_product_adaptive_miniwarp2(Ri_, wi_, wi, 0);
        // END_PROF("SHIFTED_CSRVEC");

        size_precoarse = Ai_->full_n;
        size_coarse = Ai->full_n;
        coarse_ratio = size_precoarse / size_coarse;

        if (coarse_ratio <= COARSERATIO_THRSLD) {
            amg_data->ftcoarse = FTCOARSE_INC;
        }

        bool brk_flag = (i + 1 >= amg_data->sweepnumber) || (size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize);

        if (i == 0) {
            *P = Pi_;
            if (amg_data->sweepnumber > 1 && !(size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize)) {
                CSRm::free(Ri_);
                Ri_ = NULL;
            }
        } else {

            // BEGIN_PROF("MUL");

            CSRm::shift_cols(*P, -(Pi_->row_shift));
            (*P)->m = (unsigned long)Pi_->n;
            csrlocinfo Pinfo1p;
            Pinfo1p.fr = 0;
            Pinfo1p.lr = Pi_->n;
            Pinfo1p.row = Pi_->row;
            Pinfo1p.col = NULL;
            Pinfo1p.val = Pi_->val;

            CSR* tmpP = *P;
            *P = nsparseMGPU(*P, Pi_, &Pinfo1p, brk_flag);
            CSRm::free(tmpP);

            // END_PROF("MUL");
            MUL_NUM += 1;

            // BEGIN_PROF("OTHER");
            CSRm::free(Ri_);
            Ri_ = NULL;
            CSRm::free(Pi_);
            CSRm::free(Ai_);
            // END_PROF("OTHER");
        }
        Vector::free(wi_);

        if (size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize) {
            break;
        }

        Ai_ = Ai;
        wi_ = wi;
        if (myid != 0 && Ai_->col_shifted == 0) {
            CSRm::shift_cols(Ai_, -(Ai_->row_shift));
            Ai_->col_shifted = -(Ai_->row_shift);
        }
    }

    *w = wi;

    if (Ri_ == NULL) {

        // BEGIN_PROF("TRA_P");
        if (nprocs > 1) {
            gstype m_shifts[nprocs];
            // send columns numbers to each process
            m_shifts[myid] = Ai->row_shift;
            CSRm::shift_cols(*P, -m_shifts[myid]);

            gstype swp_m = (*P)->m;
            if (myid == nprocs - 1) {
                (*P)->m = Ai->n;
            } else {
                (*P)->m = Ai->n;
            }

#if defined(GENERAL_TRANSPOSE)
            CSRm::shift_cols(*P, (*P)->row_shift);
            *R = CSRm::transpose(*P, log_file, "R");
            CSRm::shift_cols(*P, -(*P)->row_shift);
#else
            *R = CSRm::Transpose_local(*P, log_file);
#endif

            (*P)->m = swp_m;
            CSRm::shift_cols(*P, m_shifts[myid]);

            (*R)->m = (*P)->full_n;
            (*R)->full_n = (*P)->m;

#if !defined(GENERAL_TRANSPOSE)
            CSRm::shift_cols(*R, (*P)->row_shift);
#endif
            (*R)->row_shift = m_shifts[myid];
        } else {
            *R = CSRm::Transpose_local(*P, log_file);
        }
        // END_PROF("TRA_P");
    } else {
        *R = Ri_;
    }

    if (myid != 0 && Ai->col_shifted == 0) {
        CSRm::shift_cols(Ai, -(Ai->row_shift));
        Ai->col_shifted = -(Ai->row_shift);
    }

    END_PROF(__FUNCTION__);
    return Ai;
}

/**
 * @brief Function for adaptive coarsening in a multilevel solver hierarchy.
 * 
 * This function builds the multilevel hierarchy by performing adaptive coarsening based on the AMG (Algebraic Multigrid) method.
 * It allocates memory for various buffers, sets up the communication patterns for the solver, and computes the prolongation
 * and restriction operators for the AMG hierarchy.
 * 
 * @param h A pointer to the `handles` structure which contains solver-related data.
 * @param amg_data A pointer to the `buildData` structure which contains the matrix and related data.
 * @param p A reference to the `params` structure which holds solver parameters such as memory allocation size and preconditioner type.
 * 
 * @return A pointer to the `hierarchy` structure which holds the multilevel hierarchy and its components.
 * 
 * @note This function also involves device memory management for CUDA-based operations, and it manages communication patterns
 *       when multiple processes are involved.
 */
hierarchy* adaptiveCoarsening(handles* h, buildData* amg_data, const params& p)
{
    BEGIN_PROF(__FUNCTION__);

    _MPI_ENV;

    // BEGIN_PROF("MEM");
    CSR* A = amg_data->A;

    // init memory pool
    amg_data->ws_buffer = Vector::init<vtype>(A->n, true, true);
    amg_data->mutex_buffer = Vector::init<itype>(A->n, true, true);
    amg_data->_M = Vector::init<itype>(A->n, true, true);

    iPtemp1 = NULL;
    vPtemp1 = NULL;

    iAtemp1 = CUDA_MALLOC_HOST(itype, p.mem_alloc_size, true);
    vAtemp1 = CUDA_MALLOC_HOST(vtype, p.mem_alloc_size, true);

    idevtemp1 = CUDA_MALLOC(itype, p.mem_alloc_size, true);
    vdevtemp1 = CUDA_MALLOC(vtype, p.mem_alloc_size, true);
    idevtemp2 = CUDA_MALLOC(itype, p.mem_alloc_size, true);

    // -------- AH glob --------
    AH_glob_row = CUDA_MALLOC(itype, A->n + 1, true);
    AH_glob_col = CUDA_MALLOC(itype, A->nnz, true);
    AH_glob_val = CUDA_MALLOC(vtype, A->nnz, true);
    // -------------------------

    vector<vtype>* w = amg_data->w;
    vector<vtype>* w_temp = NULL;

    if (w->on_the_device) {
        w_temp = Vector::init<vtype>(w->n, true, true);
        cudaError_t err = cudaMemcpy(w_temp->val, w->val, w_temp->n * sizeof(vtype), cudaMemcpyDeviceToDevice);
        CHECK_DEVICE(err);
    } else {
        w_temp = Vector::clone(w);
    }
    // ----------------------------//

    CSR *P = NULL, *R = NULL;
    hierarchy* hrrch = AMG::Hierarchy::init(amg_data->maxlevels + 1);
    hrrch->A_array[0] = A;

    // END_PROF("MEM");

    // compute comunication patterns for solver
    if (nprocs > 1) {
        // BEGIN_PROF("HALOSETUP");
        halo_info hi = haloSetup(hrrch->A_array[0], NULL);
        // END_PROF("HALOSETUP");
        hrrch->A_array[0]->halo = hi;
    }

    vtype avcoarseratio = 0.;
    int level = 0;
    if (p.sprec != PreconditionerType::NONE) {
        relaxPrepare(h, level, hrrch->A_array[level], hrrch, amg_data);
    }

    amg_data->ftcoarse = 1;

    if (p.sprec != PreconditionerType::NONE) {
        for (level = 1; level < amg_data->maxlevels;) {

            hrrch->A_array[level] = matchingAggregation(h, amg_data, hrrch->A_array[level - 1], &w_temp, &P, &R, level - 1);

            if (nprocs > 1) {
                // BEGIN_DETAILED_TIMING(PREC_SETUP, HALOSETUP);
                if (myid != 0 && hrrch->A_array[level]->col_shifted == 0) {
                    CSRm::shift_cols(hrrch->A_array[level], -(hrrch->A_array[level]->row_shift));
                    hrrch->A_array[level]->col_shifted = -(hrrch->A_array[level]->row_shift);
                }
                halo_info hi = haloSetup(hrrch->A_array[level], NULL);
                // END_DETAILED_TIMING(PREC_SETUP, HALOSETUP);

                hrrch->A_array[level]->halo = hi;
            }

            if (!amg_data->agg_interp_type) {
                relaxPrepare(h, level, hrrch->A_array[level], hrrch, amg_data);
            }

            hrrch->P_array[level - 1] = P;
            hrrch->R_array[level - 1] = R;

            // --------------- PICO ------------------
            bool shrink_col(CSR*, CSR*);
            shrink_col(hrrch->A_array[level - 1], NULL);
            if (myid != 0 && hrrch->P_array[level - 1]->col_shifted == 0) {
                CSRm::shift_cols(hrrch->P_array[level - 1], -(hrrch->A_array[level]->row_shift));
                hrrch->P_array[level - 1]->col_shifted = -(hrrch->A_array[level]->row_shift);
            }
            shrink_col(hrrch->P_array[level - 1], hrrch->A_array[level]);

            if (level != hrrch->num_levels - 1) {
                if (myid != 0 && hrrch->R_array[level - 1]->col_shifted == 0) {
                    hrrch->R_array[level - 1]->bitcol = NULL;
                    hrrch->R_array[level - 1]->bitcolsize = 0;
                    CSRm::shift_cols(hrrch->R_array[level - 1], -(hrrch->A_array[level - 1]->row_shift));
                    hrrch->R_array[level - 1]->col_shifted = -(hrrch->A_array[level - 1]->row_shift);
                }

                shrink_col(hrrch->R_array[level - 1], hrrch->A_array[level - 1]);
            }
            // ---------------------------------------

            if (nprocs > 1) {
                // BEGIN_DETAILED_TIMING(PREC_SETUP, HALOSETUP);
                halo_info hi = haloSetup(hrrch->P_array[level - 1], NULL);
                // END_DETAILED_TIMING(PREC_SETUP, HALOSETUP);
                hrrch->P_array[level - 1]->halo = hi;
            }

            if (nprocs > 1 && (level != hrrch->num_levels - 1)) {
                // BEGIN_DETAILED_TIMING(PREC_SETUP, HALOSETUP);
                halo_info hi = haloSetup(hrrch->R_array[level - 1], NULL);
                // END_DETAILED_TIMING(PREC_SETUP, HALOSETUP);
                hrrch->R_array[level - 1]->halo = hi;
            }

            vtype size_coarse = hrrch->A_array[level]->full_n;

            vtype coarse_ratio = hrrch->A_array[level - 1]->full_n / size_coarse;
            avcoarseratio = avcoarseratio + coarse_ratio;
            level++;

            if (size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize) {
                break;
            }
        }
        shrink_col(hrrch->A_array[level - 1], NULL);
    } else {
        bool shrink_col(CSR*, CSR*);
        shrink_col(hrrch->A_array[level], NULL);
    }

    // BEGIN_PROF("MEM");

    if (p.sprec != PreconditionerType::NONE) {
        AMG::Hierarchy::finalize_level(hrrch, level);
        AMG::Hierarchy::finalize_cmplx(hrrch);
        AMG::Hierarchy::finalize_wcmplx(hrrch);
        hrrch->avg_cratio = level > 1
            ? (avcoarseratio / (level - 1))
            : 1;
    }
    CUDA_FREE_HOST(iPtemp1);
    FREE(vPtemp1);
    CUDA_FREE_HOST(iAtemp1);
    CUDA_FREE_HOST(vAtemp1);
    CUDA_FREE(idevtemp1);
    CUDA_FREE(idevtemp2);
    CUDA_FREE(vdevtemp1);
    // ----------------- TEST --------------------
    CUDA_FREE(dev_rcvprow_stat);
    CUDA_FREE(completedP_stat_val);
    CUDA_FREE(completedP_stat_col);
    CUDA_FREE(completedP_stat_row);
    // --------------- AH glob -------------------
    CUDA_FREE(AH_glob_row);
    CUDA_FREE(AH_glob_col);
    CUDA_FREE(AH_glob_val);
    // -------------------------------------------
    if (alloced_idx == true) {
        CUDA_FREE(idx_4shrink);
    }

    Vector::free(w_temp);
    Vector::free(amg_data->ws_buffer);
    Vector::free(amg_data->mutex_buffer);
    Vector::free(amg_data->_M);

    // END_PROF("MEM");

    END_PROF(__FUNCTION__);
    return hrrch;
}
