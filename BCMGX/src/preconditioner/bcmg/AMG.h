#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"

struct hierarchy {
    CSR** A_array; /*array of coarse matrices including the fine ones*/
    CSR** P_array; /*array of prolongators */
    CSR** R_array; /*array of prolongators */
    CSR** R_local_array; /*array of prolongators */
    CSR** P_local_array; /*array of prolongators */

    vector<vtype>** D_array; /* diagonal of coarse matrices */
    vector<vtype>** M_array; /* new diagonal matrix */

    itype num_levels; /* number of levels of the hierachy */
    vtype op_cmplx; /* operator complexity of the hierarchy for V-cycle*/
    vtype op_wcmplx; /* operator complexity of the hierarchy for W-cycle*/
    vtype avg_cratio; /* average of coarsening ratio of the hierarchy */
};

struct buildData {
    /* setup params */
    itype maxlevels; /* max number of levels per hierarchy */
    itype maxcoarsesize; /* maximum size of the coarsest matrix */
    itype sweepnumber; /* number of pairwise aggregation steps. Currently 0 for pairwise and 1 for double pairwise */
    itype agg_interp_type; /* 1 for smoothed aggregation, 0 for pure aggregation */
    CoarseSolverType coarse_solver; /* solver to be used on the coarsest level */

    /* CR params */
    RelaxType CRrelax_type; /* to choose relaxation scheme for Compatible Relaxation */
    vtype CRrelax_weight; /* weight for weighted Jacobi in CR */
    itype CRit; /* number of iterations for Compatible Relaxation */
    vtype CRratio; /* optimal convergence ratio in Compatible Relaxation to stop coarsening*/

    CSR* A;
    vector<vtype>* w;

    itype ftcoarse;
};

struct applyData {
    /* cycle type params */
    CycleType cycle_type; /* 0 for V-cycle, 2 for W-cycle, 1 for G-cycle */
    int* num_grid_sweeps; /* number of sweeps on a fixed level in case of G-cycle */

    RelaxType relax_type; /* type of pre/post smoother/smoothing */
    int relaxnumber_coarse;
    int prerelax_number; /* number of pre smoothing steps */
    int postrelax_number; /* number of post smoothing steps */

    vtype relax_weight; /* weight for Jacobi relaxation */
};

struct bootBuildData {
    /* setup params for bootstrap process */
    int max_hrc; /* max number of hierarchies to be built*/
    double conv_ratio; /* desired convergence ratio */
    BootstrapCompositionType bootstrap_composition_type; /* type of composition for applying composite solver */
    /* solver_type =0 -> multiplicative composition
     * solver_type =1 -> symmetrized multiplicative composition
     * solver_type =2 -> additive composition */
    int solver_it; /* number of iterations to be applied for conv. ratio estimating */

    /* setup params per each AMG component */
    buildData* amg_data;
};

struct boot {
    /* data generated in the setup phase */
    hierarchy** H_array; /*array of AMG hierarchies */
    int n_hrc; /* number of hierarchies  */
    double estimated_ratio; /* estimated convergence ratio obtained after built*/
};

namespace AMG {

namespace Hierarchy {
    hierarchy* init(itype num_levels, bool allocate_mem = true);
    void finalize_level(hierarchy* H, int levels_used);
    // itype getNNZglobal(CSR *A);
    long getNNZglobal(CSR* A);
    vtype finalize_cmplx(hierarchy* h);
    vtype finalize_wcmplx(hierarchy* h);
    void printInfo(hierarchy* h);
}

namespace BuildData {
    buildData* init(itype maxlevels, itype maxcoarsesize, itype sweepnumber, itype agg_interp_type, itype coarse_solver, itype CRrelax_type, vtype CRrelax_weight, itype CRit, vtype CRratio);
    void free(buildData* bd);
    buildData* initDefault();
    void setMaxCoarseSize(buildData* bd);
    void print(buildData* bd);
}

namespace ApplyData {
    applyData* initDefault();
    void free(applyData* ad);
    void print(applyData* ad);
    applyData* initByParams(const params& p);
    void setGridSweeps(applyData* ad, int max_level);
}

namespace BootBuildData {
    bootBuildData* initDefault();
    void free(bootBuildData* ad);
    void print(bootBuildData* ad);
    bootBuildData* initByParams(CSR* A, params p);
}

namespace Boot {
    boot* init(int n_hrc, double estimated_ratio);
    void free(boot* b);
    void finalize(boot* b, int num_hrc);
}
}
