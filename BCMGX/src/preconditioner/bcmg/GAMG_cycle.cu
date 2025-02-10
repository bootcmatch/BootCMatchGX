#include "datastruct/vector.h"
#include "halo_communication/halo_communication.h"
#include "op/addAbsoluteRowSumNoDiag.h"
#include "op/mydiag.h"
#include "preconditioner/bcmg/GAMG_cycle.h"
#include "preconditioner/l1jacobi/l1jacobi.h"
#include "utility/distribuite.h"
#include "utility/globals.h"
#include "utility/profiling.h"
#include <stdlib.h>

#define VERBOSE 0
#define USE_M 1

#define DEBUG_GAMG_CYCLE 0
#if DEBUG_GAMG_CYCLE
int gamg_counter = 0;
#endif

// =============================================================================

vector<vtype>* GAMGcycle::Res_buffer;

void GAMGcycle::initContext(int n)
{
    GAMGcycle::Res_buffer = Vector::init<vtype>(n, true, true);
}

__inline__ void GAMGcycle::setBufferSize(itype n)
{
    GAMGcycle::Res_buffer->n = n;
}

void GAMGcycle::freeContext()
{
    Vector::free(GAMGcycle::Res_buffer);
}

int cntrelax = 0;
extern char idstring[];

// =============================================================================

// HIGHLY INEFFICIENT
bool check_same_file_content(const std::string& f1, const std::string& f2)
{
    std::ifstream ifs1(f1);
    auto ifs1_start = std::istreambuf_iterator<char>(ifs1);
    auto ifs1_end = std::istreambuf_iterator<char>();

    std::ifstream ifs2(f2);
    auto ifs2_start = std::istreambuf_iterator<char>(ifs2);
    auto ifs2_end = std::istreambuf_iterator<char>();

    return std::string(ifs1_start, ifs1_end) == std::string(ifs2_start, ifs2_end);
}

void GAMG_cycle(handles* h, int k, bootBuildData* bootamg_data, boot* boot_amg, applyData* amg_cycle, vectorCollection<vtype>* Rhs, vectorCollection<vtype>* Xtent, vectorCollection<vtype>* Xtent_2, int l /*, int coarsesolver_type*/)
{
    _MPI_ENV;

    hierarchy* hrrc = boot_amg->H_array[k];

    if (VERBOSE > 0) {
        std::cout << "P" << myid << ": " << "GAMGCycle: start of level " << l << " Max level " << hrrc->num_levels << "\n";
    }

    // -------------------------------------------------------------------------
    // l ==  hrrc->num_levels means we are at the coarsest level, hence
    // we have to solve the coarsest system.
    // Currently, L1-Jacobi is used.
    // -------------------------------------------------------------------------
    // TODO: support multiple solvers
    if (l == hrrc->num_levels) {
        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            std::cout << "P" << myid << ": " << "Before coarsest level " << l << " XTent " << tnrm << "\n";
        }

        // BEGIN_DETAILED_TIMING(PREC_APPLY, SOLRELAX);
#if USE_M
        l1jacobi_iter(h, amg_cycle->relaxnumber_coarse, hrrc->A_array[l - 1], hrrc->M_array[l - 1], Rhs->val[l - 1], Xtent->val[l - 1], &Xtent_2->val[l - 1]);
#else
        l1jacobi_iter(h, amg_cycle->relaxnumber_coarse, hrrc->A_array[l - 1], hrrc->D_array[l - 1], Rhs->val[l - 1], Xtent->val[l - 1], &Xtent_2->val[l - 1]);
#endif
        // END_DETAILED_TIMING(PREC_APPLY, SOLRELAX);

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            std::cout << "P" << myid << ": " << "After coarsest level " << l << " XTent " << tnrm << "\n";
        }
    }
    // -------------------------------------------------------------------------
    // l != hrrc->num_levels, means we are at an intermediate level, hence
    // we have to solve the system recursively.
    // -------------------------------------------------------------------------
    else {
        // fprintf(stdout, "l != hrrc->num_levels\n");

        // ---------------------------------------------------------------------
        // PRE-SMOOTHING (begin)
        // ---------------------------------------------------------------------

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Rhs->val[l - 1]);
            fprintf(stdout, "P%d: Before pre smoothing at level %d Rhs %lf\n", myid, l, tnrm);

            tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            fprintf(stdout, "P%d: Before pre smoothing at level %d Xtent %lf\n", myid, l, tnrm);

            tnrm = Vector::norm(h->cublas_h, hrrc->M_array[l - 1]);
            fprintf(stdout, "P%d: Before pre smoothing at level %d M %lf\n", myid, l, tnrm);
        }

        int prerelax_coeff = amg_cycle->cycle_type == CycleType::VARIABLE_V_CYCLE
            ? (1 << (l - 1))
            : 1;

        // pre_smoothing
        // BEGIN_DETAILED_TIMING(PREC_APPLY, SOLRELAX);
#if USE_M
        l1jacobi_iter(h, amg_cycle->prerelax_number * prerelax_coeff, hrrc->A_array[l - 1], hrrc->M_array[l - 1], Rhs->val[l - 1], Xtent->val[l - 1], &Xtent_2->val[l - 1]);
#else
        l1jacobi_iter(h, amg_cycle->prerelax_number * prerelax_coeff, hrrc->A_array[l - 1], hrrc->D_array[l - 1], Rhs->val[l - 1], Xtent->val[l - 1], &Xtent_2->val[l - 1]);
#endif
        // END_DETAILED_TIMING(PREC_APPLY, SOLRELAX);

#if DEBUG_GAMG_CYCLE
        dump(Xtent->val[l - 1], "%s/%s_%04d_%04d_%s_Xtent_p%d%s.txt",
            output_dir.c_str(), output_prefix.c_str(),
            gamg_counter, __LINE__, __func__,
            myid, output_suffix.c_str());
        gamg_counter++;
#endif

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            fprintf(stdout, "P%d: After pre smoothing at level %d Xtent %lf\n", myid, l, tnrm);
            tnrm = Vector::norm(h->cublas_h, Rhs->val[l - 1]);
            fprintf(stdout, "P%d: After pre smoothing at level %d Rhs %lf\n", myid, l, tnrm);
        }

        // ---------------------------------------------------------------------
        // PRE-SMOOTHING (end)
        // ---------------------------------------------------------------------

        // compute residual
        // BEGIN_DETAILED_TIMING(PREC_APPLY, RESTGAMG);

        GAMGcycle::setBufferSize(Rhs->val[l - 1]->n);
        vector<vtype>* Res = GAMGcycle::Res_buffer;

        // BEGIN_DETAILED_TIMING(PREC_APPLY, VECTOR_COPY);
        Vector::copyTo(Res, Rhs->val[l - 1], (nprocs > 1) ? *(hrrc->A_array[l - 1]->os.streams->comm_stream) : 0);
        // END_DETAILED_TIMING(PREC_APPLY, VECTOR_COPY);

        CSRm::CSRVector_product_adaptive_miniwarp_witho(hrrc->A_array[l - 1], Xtent->val[l - 1], Res, -1., 1.);

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm_MPI(h->cublas_h, Res);
            // std::cout << "Residual at level " << l << " " << tnrm << "\n";
            fprintf(stdout, "P%d: Residual at level %d: %lf\n", myid, l, tnrm);
        }

        if (nprocs == 1) {
            CSRm::CSRVector_product_adaptive_miniwarp_witho(hrrc->R_array[l - 1], Res, Rhs->val[l], 1., 0.);
        } else {

            CSR* R_local = hrrc->R_local_array[l - 1];
            assert(hrrc->A_array[l - 1]->full_n == R_local->m);
            vector<vtype>* Res_full = Xtent_2->val[l - 1];

            // BEGIN_DETAILED_TIMING(PREC_APPLY, CUDAMEMCOPY);
            cudaMemcpy(Res_full->val, Res->val, hrrc->A_array[l - 1]->n * sizeof(vtype), cudaMemcpyDeviceToDevice);
            // END_DETAILED_TIMING(PREC_APPLY, CUDAMEMCOPY);

            CSRm::CSRVector_product_adaptive_miniwarp_witho(R_local, Res_full, Rhs->val[l], 1., 0.);
        }

        Vector::fillWithValue(Xtent->val[l], 0.);

        // END_DETAILED_TIMING(PREC_APPLY, RESTGAMG);

        if (hrrc->num_levels > 2 || amg_cycle->relaxnumber_coarse > 0) {
            for (int i = 1; i <= amg_cycle->num_grid_sweeps[l - 1]; i++) {
                GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, Rhs, Xtent, Xtent_2, l + 1);
                if (l == hrrc->num_levels - 1) {
                    break;
                }
            }
        }

        // BEGIN_DETAILED_TIMING_NODEC(PREC_APPLY, RESTGAMG);
        if (nprocs == 1) {
            CSRm::CSRVector_product_adaptive_miniwarp(hrrc->P_array[l - 1], Xtent->val[l], Xtent->val[l - 1], 1., 1.);
        } else {
            CSRm::CSRVector_product_adaptive_miniwarp_witho(hrrc->P_local_array[l - 1], Xtent->val[l], Xtent->val[l - 1], 1., 1.);
        }
        // END_DETAILED_TIMING(PREC_APPLY, RESTGAMG);

        // ---------------------------------------------------------------------
        // POST-SMOOTHING (begin)
        // ---------------------------------------------------------------------

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            std::cout << "P" << myid << ": " << "Before post smoothing at level " << l << " XTent " << tnrm << "\n";
        }

        int postrelax_coeff = amg_cycle->cycle_type == CycleType::VARIABLE_V_CYCLE
            ? (1 << (l - 1))
            : 1;

        // BEGIN_DETAILED_TIMING_NODEC(PREC_APPLY, SOLRELAX);
#if USE_M
        l1jacobi_iter(h, amg_cycle->postrelax_number * postrelax_coeff, hrrc->A_array[l - 1], hrrc->M_array[l - 1], Rhs->val[l - 1], Xtent->val[l - 1], &Xtent_2->val[l - 1]);
#else
        l1jacobi_iter(h, amg_cycle->postrelax_number * postrelax_coeff, hrrc->A_array[l - 1], hrrc->D_array[l - 1], Rhs->val[l - 1], Xtent->val[l - 1], &Xtent_2->val[l - 1]);
#endif
        // END_DETAILED_TIMING(PREC_APPLY, SOLRELAX);

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            std::cout << "P" << myid << ": " << "After post smoothing at level " << l << " XTent " << tnrm << "\n";
        }

        // ---------------------------------------------------------------------
        // POST-SMOOTHING (end)
        // ---------------------------------------------------------------------
    }

    if (VERBOSE > 0) {
        std::cout << "P" << myid << ": " << "GAMGCycle: end of level " << l << "\n";
    }
    cntrelax = 0;
}
