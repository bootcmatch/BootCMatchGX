/**
 * @file hierarchy.h
 * @brief Defines data structures and functions for AMG hierarchy construction, destruction and debugging.
 */
#pragma once

#include "config/Params.h"
#include "datastruct/CSR.h"
#include <stdio.h>

/**
 * @struct hierarchy
 * @brief Represents a multilevel hierarchy for Algebraic Multigrid (AMG).
 */
struct hierarchy {
    CSR** A_array; /**< Array of coarse matrices including the fine ones. */
    CSR** P_array; /**< Array of prolongators. */
    CSR** R_array; /**< Array of restrictors. */
    CSR** R_local_array; /**< Array of local restrictors. */
    CSR** P_local_array; /**< Array of local prolongators. */

    vector<vtype>** D_array; /**< Diagonal of coarse matrices. */
    vector<vtype>** M_array; /**< New diagonal matrix. */

    itype num_levels; /**< Number of levels in the hierarchy. */
    vtype op_cmplx; /**< Operator complexity for V-cycle. */
    vtype op_wcmplx; /**< Operator complexity for W-cycle. */
    vtype avg_cratio; /**< Average coarsening ratio of the hierarchy. */
};

/**
 * @struct buildData
 * @brief Stores parameters and data for AMG setup phase.
 */
struct buildData {
    itype maxlevels; /**< Maximum number of levels per hierarchy. */
    itype maxcoarsesize; /**< Maximum size of the coarsest matrix. */
    itype sweepnumber; /**< Number of pairwise aggregation steps. */
    itype agg_interp_type; /**< 1 for smoothed aggregation, 0 for pure aggregation. */
    CoarseSolverType coarse_solver; /**< Solver used on the coarsest level. */

    RelaxType CRrelax_type; /**< Relaxation scheme for Compatible Relaxation. */
    vtype CRrelax_weight; /**< Weight for weighted Jacobi in CR. */
    itype CRit; /**< Number of iterations for Compatible Relaxation. */
    vtype CRratio; /**< Optimal convergence ratio to stop coarsening. */

    CSR* A; /**< Coefficient matrix. */
    vector<vtype>* w; /**< Weight vector. */
    vector<vtype>* ws_buffer; /**< Buffer for weight storage. */
    vector<itype>* mutex_buffer; /**< Buffer for mutex handling. */
    vector<itype>* _M; /**< Additional storage. */

    itype ftcoarse; /**< Coarse threshold flag. */
};

/**
 * @struct applyData
 * @brief Stores parameters for applying AMG cycles.
 */
struct applyData {
    CycleType cycle_type; /**< Cycle type. */
    int* num_grid_sweeps; /**< Number of sweeps for fixed level in G-cycle. */

    RelaxType relax_type; /**< Type of pre/post smoother. */
    int relaxnumber_coarse; /**< Number of relaxations on the coarsest level. */
    int prerelax_number; /**< Number of pre-smoothing steps. */
    int postrelax_number; /**< Number of post-smoothing steps. */

    vtype relax_weight; /**< Relaxation weight for Jacobi smoothing. */
};

/**
 * @struct bootBuildData
 * @brief Stores setup parameters for the bootstrap AMG process.
 */
struct bootBuildData {
    int max_hrc; /**< Maximum number of hierarchies to be built. */
    double conv_ratio; /**< Desired convergence ratio. */
    BootstrapCompositionType bootstrap_composition_type; /**< Type of composition for solver application. */
    int solver_it; /**< Number of iterations for convergence ratio estimation. */

    buildData* amg_data; /**< AMG setup parameters. */
};

/**
 * @struct boot
 * @brief Stores bootstrap AMG data generated in the setup phase.
 */
struct boot {
    hierarchy** H_array; /**< Array of AMG hierarchies. */
    int n_hrc; /**< Number of hierarchies. */
    double estimated_ratio; /**< Estimated convergence ratio after building. */
};

/**
 * @namespace AMG
 * @brief Contains functions for AMG hierarchy construction and application.
 */
namespace AMG {

/**
 * @namespace Hierarchy
 * @brief Functions for managing AMG hierarchies.
 */
namespace Hierarchy {
    hierarchy* init(itype num_levels, bool allocate_mem = true);
    void finalize_level(hierarchy* H, int levels_used);
    long getNNZglobal(CSR* A);
    vtype finalize_cmplx(hierarchy* h);
    vtype finalize_wcmplx(hierarchy* h);
    void printInfo(FILE* fp, hierarchy* h);
}

/**
 * @namespace BuildData
 * @brief Functions for managing AMG build data.
 */
namespace BuildData {
    buildData* init(itype maxlevels, itype maxcoarsesize, itype sweepnumber, itype agg_interp_type, itype coarse_solver, itype CRrelax_type, vtype CRrelax_weight, itype CRit, vtype CRratio);
    void free(buildData* bd);
    buildData* initDefault();
    void setMaxCoarseSize(buildData* bd);
    void print(buildData* bd);
}

/**
 * @namespace ApplyData
 * @brief Functions for managing AMG application data.
 */
namespace ApplyData {
    applyData* initDefault();
    void free(applyData* ad);
    void print(FILE* fp, applyData* ad);
    applyData* initByParams(const params& p);
    void setGridSweeps(applyData* ad, int max_level);
}

/**
 * @namespace BootBuildData
 * @brief Functions for managing bootstrap AMG build data.
 */
namespace BootBuildData {
    bootBuildData* initDefault();
    void free(bootBuildData* ad);
    void print(FILE* fp, bootBuildData* ad);
    bootBuildData* initByParams(CSR* A, params p);
}

/**
 * @namespace Boot
 * @brief Functions for managing bootstrap AMG structures.
 */
namespace Boot {
    boot* init(int n_hrc, double estimated_ratio);
    void free(boot* b);
    void finalize(boot* b, int num_hrc);
}
}
