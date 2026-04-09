#include "matchingAggregation.h"
#include "op/spspmpi.h"
#include "preconditioner/bcmg/matchingPairAggregation.h"
#include "utility/col8.h"
#include "utility/memory.h"
#include "utility/profiling.h"

#define FTCOARSE_INC 100
#define COARSERATIO_THRSLD 1.2

int I = 0;

// -------------------------
long* buffer_4_getmct;
int sizeof_buffer_4_getmct = 0;
unsigned int* idx_4shrink;
bool alloced_idx = false;
// ------ cuCompactor ------
int* glob_d_BlocksCount;
int* glob_d_BlocksOffset;
// -------------------------

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

    assert(amg_data->A->row);
    assert(amg_data->A->col);
    assert(amg_data->A->val);
    assert(amg_data->A->col8);

    CSR *Ai_ = A, *Ai = NULL;

    CSR* Ri_ = NULL;
    vector<vtype>*wi_ = *w, *wi = NULL;

    double size_coarse, size_precoarse;
    double coarse_ratio;
    for (int i = 0; i < amg_data->sweepnumber; i++) {
        // {
        //     char fname[1024] = {0};
        //     snprintf(fname, 1024, "%s/%s%s%d_level%d_id%d_nprocs%d.mtx", output_dir.c_str(), "", "A", i, level, myid, nprocs);
        //     CSRm::printMM(Ai_, fname, false);
        // }

        CSR* Pi_;

        TRACE("Before matchingPairAggregation");
        matchingPairAggregation(h, amg_data, Ai_, wi_, &Pi_, &Ri_); /* routine with the real work. It calls the suitor procedure */
        TRACE("After matchingPairAggregation");

        // {
        //     char fname[1024] = {0};
        //     snprintf(fname, 1024, "%s/%s%s%d_level%d_id%d_nprocs%d.mtx", output_dir.c_str(), "", "P", i, level, myid, nprocs);
        //     CSRm::printMM(Pi_, fname, false);
        // }

        TRACE("Before nsparseMGPU_commu_new");
        CSR* AP = nsparseMGPU_commu_new(h, Ai_, Pi_);
        col2col8(AP->col, AP->col8, AP->nnz);
        TRACE("After nsparseMGPU_commu_new");

        // {
        //     char fname[1024] = {0};
        //     snprintf(fname, 1024, "%s/%s%s%d_level%d_id%d_nprocs%d.mtx", output_dir.c_str(), "", "AP", i, level, myid, nprocs);
        //     CSRm::printMM(AP, fname, false);
        // }

        CSRm::shift_cols(Ri_, -AP->row_shift);
        Ri_->col_shifted = -AP->row_shift;

        // {
        //     char fname[1024] = {0};
        //     snprintf(fname, 1024, "%s/%s%s%d_level%d_id%d_nprocs%d.mtx", output_dir.c_str(), "", "R", i, level, myid, nprocs);
        //     CSRm::printMM(Ri_, fname, false);
        // }

        TRACE("Before nsparseMGPU_noCommu_new");
        Ai = nsparseMGPU_noCommu_new(h, Ri_, AP);
        col2col8(Ai->col, Ai->col8, Ai->nnz);
        TRACE("After nsparseMGPU_noCommu_new");

        if (myid != 0 && Ai->col_shifted == 0) {
            CSRm::shift_cols(Ai, -(Ai->row_shift));
            Ai->col_shifted = -(Ai->row_shift);
        }

        // {
        //     char fname[1024] = {0};
        //     snprintf(fname, 1024, "%s/%s%s%d_level%dnext_id%d_nprocs%d.mtx", output_dir.c_str(), "", "A", i, level, myid, nprocs);
        //     CSRm::printMM(Ai, fname, false);
        // }

        // ---------------------------------------

        // BEGIN_PROF("OTHER");
        CSRm::free(AP);
        // END_PROF("OTHER");

        wi = Vector::init<vtype>(Ai->n, true, true);

        // BEGIN_PROF("SHIFTED_CSRVEC");
        TRACE("Before shifted_CSRVector_product_adaptive_miniwarp2");
        CSRm::shifted_CSRVector_product_adaptive_miniwarp2(Ri_, wi_, wi, 0);
        TRACE("After shifted_CSRVector_product_adaptive_miniwarp2");
        // END_PROF("SHIFTED_CSRVEC");

        size_precoarse = Ai_->full_n;
        size_coarse = Ai->full_n;
        coarse_ratio = size_precoarse / size_coarse;

        if (coarse_ratio <= COARSERATIO_THRSLD) {
            amg_data->ftcoarse = FTCOARSE_INC;
        }

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
            CSR* tmpP = *P;

            csrlocinfo Pinfo1p;
            Pinfo1p.fr = 0;
            Pinfo1p.lr = Pi_->n;
            Pinfo1p.row = Pi_->row;
            Pinfo1p.col = NULL;
            Pinfo1p.val = Pi_->val;
            *P = nsparseMGPU(*P, Pi_, &Pinfo1p);
            col2col8((*P)->col, (*P)->col8, (*P)->nnz);

            CSRm::free(tmpP);

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

    // assert(Ai->row);
    // assert(Ai->col);
    // assert(Ai->val);
    // assert(Ai->col8);

    *w = wi;

    if (Ri_ == NULL) {

        // BEGIN_PROF("TRA_P");
        if (nprocs > 1) {
            CSRm::shift_cols(*P, -Ai->row_shift);

            gstype swp_m = (*P)->m;
            (*P)->m = Ai->n;
            *R = CSRm::Transpose_local(*P, log_file);
            (*P)->m = swp_m;
            CSRm::shift_cols(*P, Ai->row_shift);

            (*R)->m = (*P)->full_n;
            (*R)->full_n = (*P)->m;
            CSRm::shift_cols(*R, (*P)->row_shift);
            (*R)->row_shift = Ai->row_shift;

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

    if (myid != 0 && (*P)->col_shifted == 0) {
        CSRm::shift_cols(*P, -Ai->row_shift);
        (*P)->col_shifted = -Ai->row_shift;
    }

    if (myid != 0 && (*R)->col_shifted == 0) {
        (*R)->bitcol = NULL;
        (*R)->bitcolsize = 0;
        CSRm::shift_cols(*R, -A->row_shift);
        (*R)->col_shifted = -A->row_shift;
    }

    // static int counter = 0;
    // {
    //     char fname[1024] = {0};
    //     snprintf(fname, 1024, "%s/%sma%d_%s_id%d_nprocs%d.mtx", output_dir.c_str(), "", counter, "R", myid, nprocs);
    //     CSRm::printMM(*R, fname, false);
    // }
    // CSR *Raux = CSRm::transpose(*P, log_file);
    // {
    //     char fname[1024] = {0};
    //     snprintf(fname, 1024, "%s/%sma%d_%s_id%d_nprocs%d.mtx", output_dir.c_str(), "", counter, "Raux", myid, nprocs);
    //     CSRm::printMM(Raux, fname, false);
    // }
    // CSRm::free(Raux);
    // counter++;

    END_PROF(__FUNCTION__);

    return Ai;
}


