#include "basic_kernel/halo_communication/halo_communication.h"
#include "datastruct/vector.h"
#include "preconditioner/bcmg/GAMG_cycle.h"
#include "preconditioner/l1jacobi/l1jacobi.h"
#include "utility/distribuite.h"
#include "utility/function_cnt.h"
#include "utility/metrics.h"
#include "utility/timing.h"
#include <stdlib.h>

vector<vtype>* GAMGcycle::Res_buffer;

void GAMGcycle::initContext(int n)
{
    Vectorinit_CNT
        GAMGcycle::Res_buffer
        = Vector::init<vtype>(n, true, true);
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

void my_relax(handles* h, int niter, CSR* A, vector<vtype>* rhs, vector<vtype>* Xtent, vector<vtype>* D)
{
    vector<vtype>* rcopy_loc = Vector::init<vtype>(A->n, true, true);
    vector<vtype>* w_loc = Vector::init<vtype>(A->n, true, true);
    for (int i = 0; i < niter; i++) {
        // ui+1 = ui +D^-1(r-A*ui)
        l1jacobi_iter(h, A, rhs, Xtent, D, rcopy_loc, w_loc);
    }
    Vector::free(rcopy_loc);
    Vector::free(w_loc);
}

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
    static int flow = 1;
    static int first = 0;

    hierarchy* hrrc = boot_amg->H_array[k];

    if (VERBOSE > 0) {
        std::cout << "GAMGCycle: start of level " << l << " Max level " << hrrc->num_levels << "\n";
    }

    char filename[256];
    FILE* fp;

    // -------------------------------------------------------------------------
    // l ==  hrrc->num_levels means we are at the coarsest level, hence
    // we have to solve the coarsest system.
    // Currently, L1-Jacobi is used.
    // -------------------------------------------------------------------------
    // TODO: support multiple solvers
    if (l == hrrc->num_levels) {

#if DETAILED_TIMING
        if (ISMASTER) {
            TIME::start();
        }
#endif

        my_relax(h, amg_cycle->relaxnumber_coarse, hrrc->A_array[l - 1], Rhs->val[l - 1], Xtent->val[l - 1], hrrc->D_array[l - 1]);

#if DETAILED_TIMING
        if (ISMASTER) {
            TOTAL_SOLRELAX_TIME += TIME::stop();
        }
#endif

    }
    // -------------------------------------------------------------------------
    // l != hrrc->num_levels, means we are at an intermediate level, hence
    // we have to solve the system recursively.
    // -------------------------------------------------------------------------
    else {

        // ---------------------------------------------------------------------
        // PRE-SMOOTHING (begin)
        // ---------------------------------------------------------------------

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            std::cout << "Before pre smoothing at level " << l << " XTent " << tnrm << "\n";
        }

#if DETAILED_TIMING
        if (ISMASTER) {
            TIME::start();
        }
#endif

        // pre_smoothing
        my_relax(h, amg_cycle->prerelax_number, hrrc->A_array[l - 1], Rhs->val[l - 1], Xtent->val[l - 1], hrrc->D_array[l - 1]);

#if DETAILED_TIMING
        if (ISMASTER) {
            TOTAL_SOLRELAX_TIME += TIME::stop();
        }
#endif

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            std::cout << "After pre smoothing at level " << l << " XTent " << tnrm << "\n";
        }

        // ---------------------------------------------------------------------
        // PRE-SMOOTHING (end)
        // ---------------------------------------------------------------------

// compute residual
#if DETAILED_TIMING
        if (ISMASTER) {
            TIME::start();
        }
#endif

        GAMGcycle::setBufferSize(Rhs->val[l - 1]->n);
        vector<vtype>* Res = GAMGcycle::Res_buffer;
        Vector::copyTo(Res, Rhs->val[l - 1]);

        CSRm::CSRVector_product_adaptive_miniwarp_witho(hrrc->A_array[l - 1], Xtent->val[l - 1], Res, -1., 1.);
        cudaDeviceSynchronize();

        if (nprocs == 1) {
            if (first == 1 || flow == 0) {
                snprintf(filename, sizeof(filename), "Rlocal_%d_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                CSRm::printMM(hrrc->R_array[l - 1], filename);
                snprintf(filename, sizeof(filename), "Res_%d_full_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                FILE* fp = fopen(filename, "w");
                if (fp == NULL) {
                    printf("Could not open %s\n", filename);
                    exit(1);
                }
                Vector::print(Res, -1, fp);
                fclose(fp);
            }

            CSRm::CSRVector_product_adaptive_miniwarp_witho(hrrc->R_array[l - 1], Res, Rhs->val[l], 1., 0.);
            cudaDeviceSynchronize();

            if (first == 1 || flow == 0) {
                snprintf(filename, sizeof(filename), "Rhs2_%d_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                fp = fopen(filename, "w");
                if (fp == NULL) {
                    printf("Could not open %s\n", filename);
                    exit(1);
                }
                Vector::print(Rhs->val[l], -1, fp);
                fclose(fp);
                flow++;
            }
        } else {

            CSR* R_local = hrrc->R_local_array[l - 1];
            assert(hrrc->A_array[l - 1]->full_n == R_local->m);
            vector<vtype>* Res_full = Xtent_2->val[l - 1];
            cudaMemcpy(Res_full->val, Res->val, hrrc->A_array[l - 1]->n * sizeof(vtype), cudaMemcpyDeviceToDevice);
            if (first == 1 || flow == 0) {
                snprintf(filename, sizeof(filename), "Rlocal_%d_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                CSRm::printMM(R_local, filename);
                snprintf(filename, sizeof(filename), "Res_%d_full_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                FILE* fp = fopen(filename, "w");
                if (fp == NULL) {
                    printf("Could not open %s\n", filename);
                    exit(1);
                }
                Vector::print(Res_full, -1, fp);
                fclose(fp);
            }

            CSRm::CSRVector_product_adaptive_miniwarp_witho(R_local, Res_full, Rhs->val[l], 1., 0.);
            cudaDeviceSynchronize();

            if (first == 1 || flow == 0) {
                snprintf(filename, sizeof(filename), "Rhs2_%d_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                fp = fopen(filename, "w");
                if (fp == NULL) {
                    printf("Could not open %s\n", filename);
                    exit(1);
                }
                Vector::print(Rhs->val[l], -1, fp);
                fclose(fp);
                flow++;
            }
        }

        Vector::fillWithValue(Xtent->val[l], 0.);

#if DETAILED_TIMING
        if (ISMASTER) {
            TOTAL_RESTGAMG_TIME += TIME::stop();
        }
#endif

        if (hrrc->num_levels > 2 || amg_cycle->relaxnumber_coarse > 0) {
            for (int i = 1; i <= amg_cycle->num_grid_sweeps[l - 1]; i++) {
                GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, Rhs, Xtent, Xtent_2, l + 1);
                if (l == hrrc->num_levels - 1) {
                    break;
                }
            }
        }

#if DETAILED_TIMING
        if (ISMASTER) {
            TIME::start();
        }
#endif

        if (nprocs == 1) {
            if (first == 1 || flow == 0) {
                snprintf(filename, sizeof(filename), "Plocal_%d_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                CSRm::printMM(hrrc->P_array[l - 1], filename);
                snprintf(filename, sizeof(filename), "Xtent_%d_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                FILE* fp = fopen(filename, "w");
                if (fp == NULL) {
                    printf("Could not open %s\n", filename);
                    exit(1);
                }
                Vector::print(Xtent->val[l], -1, fp);
                fclose(fp);
            }
            CSRm::CSRVector_product_adaptive_miniwarp(hrrc->P_array[l - 1], Xtent->val[l], Xtent->val[l - 1], 1., 1.);
            if (first == 1 || flow == 0) {
                snprintf(filename, sizeof(filename), "Xtent_%d_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                fp = fopen(filename, "w");
                if (fp == NULL) {
                    printf("Could not open %s\n", filename);
                    exit(1);
                }
                Vector::print(Xtent->val[l - 1], -1, fp);
                fclose(fp);
                flow++;
            }

        } else {
            CSRm::CSRVector_product_adaptive_miniwarp_witho(hrrc->P_local_array[l - 1], Xtent->val[l], Xtent->val[l - 1], 1., 1.);
            cudaDeviceSynchronize();

            if (first == 1 || flow == 0) {
                snprintf(filename, sizeof(filename), "Xtent_%d_%s_%d_%d_%d", __LINE__, idstring, amg_cycle->relaxnumber_coarse, flow, myid);
                fp = fopen(filename, "w");
                if (fp == NULL) {
                    printf("Could not open %s\n", filename);
                    exit(1);
                }
                Vector::print(Xtent->val[l - 1], -1, fp);
                fclose(fp);
                flow++;
            }
        }

#if DETAILED_TIMING
        if (ISMASTER) {
            TOTAL_RESTGAMG_TIME += TIME::stop();
        }
#endif

        // ---------------------------------------------------------------------
        // POST-SMOOTHING (begin)
        // ---------------------------------------------------------------------

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            std::cout << "Before post smoothing at level " << l << " XTent " << tnrm << "\n";
        }

#if DETAILED_TIMING
        if (ISMASTER) {
            TIME::start();
        }
#endif

        my_relax(h, amg_cycle->postrelax_number, hrrc->A_array[l - 1], Rhs->val[l - 1], Xtent->val[l - 1], hrrc->D_array[l - 1]);

#if DETAILED_TIMING
        if (ISMASTER) {
            TOTAL_SOLRELAX_TIME += TIME::stop();
        }
#endif

        if (VERBOSE > 1) {
            vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l - 1]);
            std::cout << "After post smoothing at level " << l << " XTent " << tnrm << "\n";
        }

        // ---------------------------------------------------------------------
        // POST-SMOOTHING (end)
        // ---------------------------------------------------------------------
    }
    first = 0;

    if (VERBOSE > 0) {
        std::cout << "GAMGCycle: end of level " << l << "\n";
    }
    cntrelax = 0;
}
