
#include "AMG.h"

#include "utility/cudamacro.h"
#include "utility/function_cnt.h"

namespace AMG {

namespace Hierarchy {

    hierarchy* init(itype num_levels, bool allocate_mem)
    {
        hierarchy* H = NULL;
        // on the host
        H = (hierarchy*)Malloc(sizeof(hierarchy));
        CHECK_HOST(H);

        H->num_levels = num_levels;
        H->op_cmplx = 0;

        H->A_array = NULL;
        H->P_array = NULL;
        H->R_array = NULL;
        H->R_local_array = NULL;
        H->P_local_array = NULL;

        if (allocate_mem) {
            H->A_array = (CSR**)Malloc(num_levels * sizeof(CSR*));
            CHECK_HOST(H->A_array);

            H->P_array = (CSR**)Malloc((num_levels - 1) * sizeof(CSR*));
            CHECK_HOST(H->P_array);

            H->R_array = (CSR**)Malloc((num_levels - 1) * sizeof(CSR*));
            CHECK_HOST(H->R_array);

            H->R_local_array = (CSR**)Malloc((num_levels - 1) * sizeof(CSR*));
            CHECK_HOST(H->R_local_array);

            H->P_local_array = (CSR**)Malloc((num_levels - 1) * sizeof(CSR*));
            CHECK_HOST(H->P_local_array);

            H->D_array = (vector<vtype>**)Malloc(num_levels * sizeof(vector<vtype>*));
            CHECK_HOST(H->D_array);

            H->M_array = (vector<vtype>**)Malloc(num_levels * sizeof(vector<vtype>*));
            CHECK_HOST(H->M_array);

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

        std::free(H->A_array);
        std::free(H->D_array);
        std::free(H->M_array);
        std::free(H->P_array);

        std::free(H);
    }

    void finalize_level(hierarchy* H, int levels_used)
    {

        assert(levels_used > 0);

        H->num_levels = levels_used;

        H->A_array = (CSR**)realloc(H->A_array, levels_used * sizeof(CSR*));
        CHECK_HOST(H->A_array);

        H->P_array = (CSR**)realloc(H->P_array, (levels_used - 1) * sizeof(CSR*));
        CHECK_HOST(H->P_array);

        H->R_array = (CSR**)realloc(H->R_array, (levels_used - 1) * sizeof(CSR*));
        CHECK_HOST(H->R_array);

        H->R_local_array = (CSR**)realloc(H->R_local_array, (levels_used - 1) * sizeof(CSR*));
        CHECK_HOST(H->R_local_array);

        H->P_local_array = (CSR**)realloc(H->P_local_array, (levels_used - 1) * sizeof(CSR*));
        CHECK_HOST(H->P_local_array);

        H->D_array = (vector<vtype>**)realloc(H->D_array, levels_used * sizeof(vector<vtype>*));
        CHECK_HOST(H->D_array);

        H->M_array = (vector<vtype>**)realloc(H->M_array, levels_used * sizeof(vector<vtype>*));
        CHECK_HOST(H->M_array);
    }

    long getNNZglobal(CSR* A)
    {
        PUSH_RANGE(__func__, 7)

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

        POP_RANGE
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

    void printInfo(hierarchy* h)
    {

        for (int i = 0; i < h->num_levels; i++) {
            CSR* Ai = h->A_array[i];
            float avg_nnz = (float)Ai->nnz / (float)Ai->n;
            std::cout << "A" << i << " n: " << Ai->full_n << " nnz: " << Ai->nnz << " avg_nnz: " << avg_nnz << "\n";
        }
        std::cout << "\nCurrent cmplx for V-cycle: " << h->op_cmplx;
        std::cout << "\nCurrent cmplx for W-cycle: " << h->op_wcmplx;
        std::cout << "\nAverage Coarsening Ratio: " << h->avg_cratio << "\n";
    }
}

namespace BuildData {
    buildData* init(itype maxlevels, itype maxcoarsesize, itype sweepnumber, itype agg_interp_type, CoarseSolverType coarse_solver, RelaxType CRrelax_type, vtype CRrelax_weight, itype CRit, vtype CRratio)
    {
        buildData* bd = NULL;

        bd = (buildData*)Malloc(sizeof(buildData));
        CHECK_HOST(bd);

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
        std::free(bd);
    }

    buildData* initDefault()
    {
        buildData* bd = NULL;

        bd = (buildData*)Malloc(sizeof(buildData));
        CHECK_HOST(bd);

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

        applyData* ad = NULL;
        ad = (applyData*)Malloc(sizeof(applyData));
        CHECK_HOST(ad);

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
        std::free(ad->num_grid_sweeps);
        std::free(ad);
    }

    void print(applyData* ad)
    {
        std::cout << "\ncycle_type: " << cycle_type_to_string(ad->cycle_type) << "\n";
        std::cout << "relax_type: " << relax_type_to_string(ad->relax_type) << "\n";
        std::cout << "relaxnumber_coarse: " << ad->relaxnumber_coarse << "\n";
        std::cout << "prerelax_number: " << ad->prerelax_number << "\n";
        std::cout << "postrelax_number: " << ad->postrelax_number << "\n";
        std::cout << "relax_weight: " << ad->relax_weight << "\n\n";
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
        ad->num_grid_sweeps = (int*)Malloc(max_level * sizeof(int));
        CHECK_HOST(ad->num_grid_sweeps);

        int i, j;

        for (i = 0; i < max_level; i++) {
            ad->num_grid_sweeps[i] = 1;
        }
    }
}

namespace BootBuildData {
    bootBuildData* initDefault()
    {

        bootBuildData* bd = NULL;
        bd = (bootBuildData*)Malloc(sizeof(bootBuildData));
        CHECK_HOST(bd);

        bd->max_hrc = 10;
        bd->conv_ratio = 0.80;
        bd->bootstrap_composition_type = BootstrapCompositionType::MULTIPLICATIVE;
        bd->solver_it = 15;

        bd->amg_data = AMG::BuildData::initDefault();

        return bd;
    }

    void free(bootBuildData* ad)
    {
        AMG::BuildData::free(ad->amg_data);
        std::free(ad);
    }

    void print(bootBuildData* ad)
    {
        std::cout << "\nmax_hrc: " << ad->max_hrc << "\n";
        std::cout << "conv_ratio: " << ad->conv_ratio << "\n";
        std::cout << "bootstrap_composition_type: " << bootstrap_composition_type_to_string(ad->bootstrap_composition_type) << "\n";
        std::cout << "solver_it: " << ad->solver_it << "\n\n";
    }

    bootBuildData* initByParams(CSR* A, params p)
    {
        bootBuildData* bootamg_data = AMG::BootBuildData::initDefault();
        buildData* amg_data = bootamg_data->amg_data;

        bootamg_data->bootstrap_composition_type = p.bootstrap_composition_type;
        bootamg_data->max_hrc = p.max_hrc;
        bootamg_data->conv_ratio = p.conv_ratio;

        amg_data->sweepnumber = p.aggrsweeps;
        amg_data->agg_interp_type = p.aggrtype;
        amg_data->maxlevels = p.max_levels;
        amg_data->coarse_solver = p.coarse_solver;
        amg_data->CRrelax_type = p.relax_type;

        amg_data->A = A;
        Vectorinit_CNT
            amg_data->w
            = Vector::init<vtype>(A->n, true, true);
        Vector::fillWithValue(amg_data->w, 1.0);

        AMG::BuildData::setMaxCoarseSize(amg_data);

        return bootamg_data;
    }
}

namespace Boot {

    boot* init(int n_hrc, double estimated_ratio)
    {
        boot* b = (boot*)Malloc(sizeof(boot));
        CHECK_HOST(b);

        b->n_hrc = n_hrc;
        b->estimated_ratio = estimated_ratio;
        b->H_array = (hierarchy**)Malloc(n_hrc * sizeof(hierarchy*));
        CHECK_HOST(b->H_array);

        return b;
    }

    void free(boot* b)
    {
        for (int i = 0; i < b->n_hrc; i++) {
            AMG::Hierarchy::free(b->H_array[i]);
        }
        std::free(b);
    }

    void finalize(boot* b, int num_hrc)
    {
        assert(num_hrc > 0);
        b->n_hrc = num_hrc;
        b->H_array = (hierarchy**)realloc(b->H_array, num_hrc * sizeof(hierarchy*));
        CHECK_HOST(b->H_array);
    }
}
}
