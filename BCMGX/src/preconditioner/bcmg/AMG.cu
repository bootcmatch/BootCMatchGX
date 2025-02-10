
#include "AMG.h"

#include "utility/cudamacro.h"
#include "utility/logf.h"
#include "utility/memory.h"
#include "utility/profiling.h"

namespace AMG {

namespace Hierarchy {

    hierarchy* init(itype num_levels, bool allocate_mem)
    {
        // on the host
        hierarchy* H = MALLOC(hierarchy, 1, true);

        H->num_levels = num_levels;
        H->op_cmplx = 0;

        H->A_array = NULL;
        H->P_array = NULL;
        H->R_array = NULL;
        H->R_local_array = NULL;
        H->P_local_array = NULL;

        if (allocate_mem) {
            H->A_array = MALLOC(CSR*, num_levels, true);
            H->P_array = MALLOC(CSR*, num_levels - 1, true);
            H->R_array = MALLOC(CSR*, num_levels - 1, true);
            H->R_local_array = MALLOC(CSR*, num_levels - 1, true);
            H->P_local_array = MALLOC(CSR*, num_levels - 1, true);
            H->D_array = MALLOC(vector<vtype>*, num_levels, true);
            H->M_array = MALLOC(vector<vtype>*, num_levels, true);

            for (int i = 0; i < H->num_levels; i++) {
                H->D_array[i] = NULL;
                if (i != H->num_levels - 1) {
                    H->R_array[i] = NULL;
                    H->R_local_array[i] = NULL;
                    H->P_local_array[i] = NULL;
                }
                H->M_array[i] = NULL;
            }
        }

        return H;
    }

    void free(hierarchy* H)
    {
        if (H) {
            for (int i = 0; i < H->num_levels; i++) {

                // skip the original matrix
                if (i > 0) {
                    CSRm::free(H->A_array[i]);
                }

                if (H->D_array[i] != NULL) {
                    Vector::free(H->D_array[i]);
                }

                if (H->M_array[i] != NULL) {
                    Vector::free(H->M_array[i]);
                }

                if (i != H->num_levels - 1) {
                    CSRm::free(H->P_array[i]);
                    if (H->R_array[i] != NULL) {
                        CSRm::free(H->R_array[i]);
                    }
                    if (H->R_local_array[i] != NULL) {
                        CSRm::free(H->R_local_array[i]);
                    }
                    if (H->P_local_array[i] != NULL) {
                        CSRm::free(H->P_local_array[i]);
                    }
                }
            }

            FREE(H->A_array);
            FREE(H->D_array);
            FREE(H->M_array);
            FREE(H->P_array);
            H->A_array = NULL;
            H->D_array = NULL;
            H->M_array = NULL;
            H->P_array = NULL;

            FREE(H);
        }
    }

    void finalize_level(hierarchy* H, int levels_used)
    {
        assert(levels_used > 0);

        H->num_levels = levels_used;

        H->A_array = (CSR**)realloc(H->A_array, levels_used * sizeof(CSR*));
        CHECK_HOST(H->A_array);

        H->D_array = (vector<vtype>**)realloc(H->D_array, levels_used * sizeof(vector<vtype>*));
        CHECK_HOST(H->D_array);

        H->M_array = (vector<vtype>**)realloc(H->M_array, levels_used * sizeof(vector<vtype>*));
        CHECK_HOST(H->M_array);

        if (levels_used == 1) {
            return;
        }

        H->P_array = (CSR**)realloc(H->P_array, (levels_used - 1) * sizeof(CSR*));
        CHECK_HOST(H->P_array);

        H->R_array = (CSR**)realloc(H->R_array, (levels_used - 1) * sizeof(CSR*));
        CHECK_HOST(H->R_array);

        H->R_local_array = (CSR**)realloc(H->R_local_array, (levels_used - 1) * sizeof(CSR*));
        CHECK_HOST(H->R_local_array);

        H->P_local_array = (CSR**)realloc(H->P_local_array, (levels_used - 1) * sizeof(CSR*));
        CHECK_HOST(H->P_local_array);
    }

    long getNNZglobal(CSR* A)
    {
        // unsigned long nnzp = 0;
        unsigned long nnzp = 0;
        unsigned long lnnz = A->nnz;
        CHECK_MPI(
            MPI_Allreduce(
                &lnnz,
                &nnzp,
                1,
                MPI_LONG,
                MPI_SUM,
                MPI_COMM_WORLD));

        return nnzp;
    }

    vtype finalize_cmplx(hierarchy* h)
    {
        vtype cmplxfinal = 0;

        for (int i = 0; i < h->num_levels; i++) {
            cmplxfinal += getNNZglobal(h->A_array[i]);
        }
        cmplxfinal /= getNNZglobal(h->A_array[0]);
        h->op_cmplx = cmplxfinal;
        return cmplxfinal;
    }

    vtype finalize_wcmplx(hierarchy* h)
    {
        vtype wcmplxfinal = 0;
        for (int i = 0; i < h->num_levels; i++) {
            wcmplxfinal += pow(2, i) * getNNZglobal(h->A_array[i]);
        }
        wcmplxfinal /= getNNZglobal(h->A_array[0]);
        h->op_wcmplx = wcmplxfinal;
        return wcmplxfinal;
    }

    void printInfo(FILE* fp, hierarchy* h)
    {
        logf(fp, "Number of levels                             : %d\n", h->num_levels);
        for (int i = 0; i < h->num_levels; i++) {
            CSR* Ai = h->A_array[i];
            float avg_nnz = (float)Ai->nnz / (float)Ai->n;

            logf(fp, "A%-44d: n: %ld nnz: %d avg_nnz: %f\n",
                i, Ai->full_n, Ai->nnz, avg_nnz);
        }
        logf(fp, "Current cmplx for V-cycle                    : %lf\n", h->op_cmplx);
        logf(fp, "Current cmplx for W-cycle                    : %lf\n", h->op_wcmplx);
        logf(fp, "Average Coarsening Ratio                     : %lf\n", h->avg_cratio);
    }
}

namespace BuildData {
    buildData* init(itype maxlevels, itype maxcoarsesize, itype sweepnumber, itype agg_interp_type, CoarseSolverType coarse_solver, RelaxType CRrelax_type, vtype CRrelax_weight, itype CRit, vtype CRratio)
    {
        buildData* bd = MALLOC(buildData, 1, true);

        bd->maxlevels = maxlevels;
        bd->maxcoarsesize = maxcoarsesize;
        bd->sweepnumber = sweepnumber;
        bd->agg_interp_type = agg_interp_type;
        bd->coarse_solver = coarse_solver;
        bd->CRrelax_type = CRrelax_type;
        bd->CRrelax_weight = CRrelax_weight;
        bd->CRit = CRit;
        bd->CRratio = CRratio;

        bd->A = NULL;
        bd->w = NULL;

        bd->ftcoarse = 1;

        return bd;
    }

    void free(buildData* bd)
    {
        if (bd) {
            Vector::free(bd->w);
            bd->w = NULL;
            FREE(bd);
        }
    }

    buildData* initDefault()
    {
        buildData* bd = MALLOC(buildData, 1, true);

        bd->maxlevels = 100;
        bd->maxcoarsesize = 100;
        bd->sweepnumber = 1;
        bd->agg_interp_type = 0;
        bd->coarse_solver = CoarseSolverType::L1_JACOBI;

        bd->CRrelax_type = RelaxType::L1_JACOBI;
        bd->CRrelax_weight = 1. / 3.;
        bd->CRit = 0;
        bd->CRratio = .3;

        bd->A = NULL;
        bd->w = NULL;

        bd->ftcoarse = 1;

        return bd;
    }

    void setMaxCoarseSize(buildData* bd)
    {
        bd->maxcoarsesize = (itype)(40 * pow((double)bd->A->full_n, (double)1. / 3.));
    }

    void print(buildData* bd)
    {
        std::cout << "\nmaxlevels: " << bd->maxlevels << "\n";
        std::cout << "maxcoarsesize: " << bd->maxcoarsesize << "\n";
        std::cout << "sweepnumber: " << bd->sweepnumber << "\n";
        std::cout << "agg_interp_type: " << bd->agg_interp_type << "\n";
        std::cout << "coarse_solver: " << coarse_solver_type_to_string(bd->coarse_solver) << "\n";
        std::cout << "CRrelax_type: " << relax_type_to_string(bd->CRrelax_type) << "\n";
        std::cout << "CRrelax_weight: " << bd->CRrelax_weight << "\n";
        std::cout << "CRit: " << bd->CRit << "\n";
        std::cout << "CRratio: " << bd->CRratio << "\n";
        std::cout << "ftcoarse: " << bd->ftcoarse << "\n\n";
    }
}

namespace ApplyData {
    applyData* initDefault()
    {
        applyData* ad = MALLOC(applyData, 1, true);

        ad->cycle_type = CycleType::V_CYCLE;
        ad->relax_type = RelaxType::L1_JACOBI;
        ad->relaxnumber_coarse = 1;
        ad->prerelax_number = 1;
        ad->postrelax_number = 1;
        ad->relax_weight = 1.0;
        ad->num_grid_sweeps = NULL;

        return ad;
    }

    void free(applyData* ad)
    {
        if (ad) {
            FREE(ad->num_grid_sweeps);
            ad->num_grid_sweeps = NULL;
            FREE(ad);
        }
    }

    void print(FILE* fp, applyData* ad)
    {
        logf(fp, "ncycle_type                                  : %s\n", cycle_type_to_string(ad->cycle_type).c_str());
        logf(fp, "relax_type                                   : %s\n", relax_type_to_string(ad->relax_type).c_str());
        logf(fp, "relaxnumber_coarse                           : %d\n", ad->relaxnumber_coarse);
        logf(fp, "prerelax_number                              : %d\n", ad->prerelax_number);
        logf(fp, "postrelax_number                             : %d\n", ad->postrelax_number);
        logf(fp, "relax_weight                                 : %f\n", ad->relax_weight);
    }

    applyData* initByParams(const params& p)
    {
        applyData* amg_cycle = AMG::ApplyData::initDefault();

        amg_cycle->cycle_type = p.cycle_type;
        amg_cycle->relax_type = p.relax_type;
        amg_cycle->relaxnumber_coarse = p.relaxnumber_coarse;
        amg_cycle->prerelax_number = p.prerelax_sweeps;
        amg_cycle->postrelax_number = p.postrelax_sweeps;

        return amg_cycle;
    }

    void setGridSweeps(applyData* ad, int max_level)
    {
        max_level--;
        if (!max_level) {
            return;
        }

        ad->num_grid_sweeps = MALLOC(int, max_level, false);

        for (int i = 0; i < max_level; i++) {
            ad->num_grid_sweeps[i] = 1;
        }

        if (ad->cycle_type == CycleType::H_CYCLE) {
            for (int i = 0; i < max_level; i++) {
                int j = i % 2; /*step is fixed to 2; it can be also different */
                if (j == 0) {
                    ad->num_grid_sweeps[i] = 2;
                }
            }
        } else if (ad->cycle_type == CycleType::W_CYCLE) {
            for (int i = 0; i < max_level - 1; i++) {
                ad->num_grid_sweeps[i] = 2;
            }
        }
    }
}

namespace BootBuildData {
    bootBuildData* initDefault()
    {
        bootBuildData* bd = MALLOC(bootBuildData, 1, true);

        bd->max_hrc = 10;
        bd->conv_ratio = 0.80;
        bd->bootstrap_composition_type = BootstrapCompositionType::SYMMETRIZED_MULTIPLICATIVE;
        bd->solver_it = 15;

        bd->amg_data = AMG::BuildData::initDefault();

        return bd;
    }

    void free(bootBuildData* ad)
    {
        if (ad) {
            AMG::BuildData::free(ad->amg_data);
            ad->amg_data = NULL;
            FREE(ad);
        }
    }

    void print(FILE* fp, bootBuildData* ad)
    {
        logf(fp, "max_hrc                                      : %d\n", ad->max_hrc);
        logf(fp, "conv_ratio                                   : %f\n", ad->conv_ratio);
        logf(fp, "bootstrap_composition_type                   : %s\n", bootstrap_composition_type_to_string(ad->bootstrap_composition_type).c_str());
        logf(fp, "solver_it                                    : %d\n", ad->solver_it);
    }

    bootBuildData* initByParams(CSR* A, params p)
    {
        bootBuildData* bootamg_data = AMG::BootBuildData::initDefault();
        buildData* amg_data = bootamg_data->amg_data;

        bootamg_data->bootstrap_composition_type = p.bootstrap_composition_type;
        bootamg_data->max_hrc = p.max_hrc;
        bootamg_data->conv_ratio = p.conv_ratio;

        amg_data->sweepnumber = p.aggrsweeps + 1;
        amg_data->agg_interp_type = p.aggrtype;
        amg_data->maxlevels = p.max_levels;
        amg_data->coarse_solver = p.coarse_solver;
        amg_data->CRrelax_type = p.relax_type;

        amg_data->A = A;
        amg_data->w = Vector::init<vtype>(A->n, true, true);
        Vector::fillWithValue(amg_data->w, 1.0);

        AMG::BuildData::setMaxCoarseSize(amg_data);

        return bootamg_data;
    }
}

namespace Boot {

    boot* init(int n_hrc, double estimated_ratio)
    {
        BEGIN_PROF(__FUNCTION__);
        boot* b = MALLOC(boot, 1, true);

        b->n_hrc = n_hrc;
        b->estimated_ratio = estimated_ratio;
        b->H_array = MALLOC(hierarchy*, n_hrc, true);

        END_PROF(__FUNCTION__);
        return b;
    }

    void free(boot* b)
    {
        if (b) {
            for (int i = 0; i < b->n_hrc; i++) {
                AMG::Hierarchy::free(b->H_array[i]);
                b->H_array[i] = NULL;
            }
            FREE(b);
        }
    }

    void finalize(boot* b, int num_hrc)
    {
        assert(num_hrc > 0);
        BEGIN_PROF(__FUNCTION__);
        b->n_hrc = num_hrc;
        b->H_array = (hierarchy**)realloc(b->H_array, num_hrc * sizeof(hierarchy*));
        CHECK_HOST(b->H_array);
        END_PROF(__FUNCTION__);
    }
}
}
