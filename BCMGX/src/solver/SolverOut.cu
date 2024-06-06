#include "SolverOut.h"
#include "stdarg.h"
#include "utility/metrics.h"
#include "utility/mpi.h"

void logf(FILE* fp, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    vprintf(fmt, args);
    if (fp) {
        vfprintf(fp, fmt, args);
    }

    va_end(args);
}

void dump(const char* filename, const params& p, SolverOut* out)
{
    _MPI_ENV;

    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("File %s opening failed.\n", filename);
    }

    // ---------------------------------------------------------------------
    // Input
    // ---------------------------------------------------------------------

    logf(fp, "nprocs                               : %d\n", nprocs);
    logf(fp, "Tolerance                            : %g\n", p.rtol);
    logf(fp, "Solver                               : %s\n", solver_type_to_string(p.solver_type).c_str());
    logf(fp, "Preconditioner                       : %s\n", preconditioner_type_to_string(p.sprec).c_str());
    logf(fp, "l1-jacobi sweeps                     : %d\n", p.l1jacsweeps);
    logf(fp, "Stop criterion                       : %s\n", p.stop_criterion ? "relative" : "absolute");
    
    // ---------------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------------

    if (out->retv < 0) {
        logf(fp, "*** Exceeded max iter ***\n");
    }
    if (out->retv > 0) {
        logf(fp, "*** Non positive definite matrix found ***\n");
    }

    logf(fp, "Total iterations                     : %d\n", out->niter);
    logf(fp, "Residual                             : %.10lf\n", out->exitRes);
    logf(fp, "Total solving time [s]               : %f\n", out->solTime / 1000);
    logf(fp, "Average time per iteration [s]       : %f\n", out->solTime / out->niter / 1000);

    // logf(fp, "agg;hierarchy_levels_num             : %d\n", pr.bcmg.boot_amg->n_hrc);
    // logf(fp, "agg;final_estimated_ratio            : %f\n", pr.bcmg.boot_amg->estimated_ratio);

    logf(fp, "Global dimension (number of rows)    : %ld\n", out->full_n);
    logf(fp, "Local dimension (number of rows)     : %d\n", out->local_n);
    logf(fp, "Initial residual                     : %.10lf\n", out->del0);

    // ---------------------------------------------------------------------
    // Metrics
    // ---------------------------------------------------------------------

#if DETAILED_TIMING
    logf(fp, "Total PRECAPPLY time [s]             : %f\n", TOTAL_PRECAPPLY_TIME / 1000);
    logf(fp, "Total PRECAPPLY time per iter [s]    : %f\n", TOTAL_PRECAPPLY_TIME / out->niter / 1000);

    logf(fp, "Total SPMV time [s]                  : %f\n", TOTAL_SPMV_TIME / 1000);
    logf(fp, "Total SPMV time per iter [s]         : %f\n", TOTAL_SPMV_TIME / out->niter / 1000);

    logf(fp, "Total CSRVECTOR time [s]             : %f\n", TOTAL_CSRVECTOR_TIME / 1000);
    logf(fp, "Total CSRVECTOR time per iter [s]    : %f\n", TOTAL_CSRVECTOR_TIME / out->niter / 1000);

    logf(fp, "Total AXPY time [s]                  : %f\n", TOTAL_AXPY_TIME / 1000);
    logf(fp, "Total AXPY time per iter [s]         : %f\n", TOTAL_AXPY_TIME / out->niter / 1000);

    logf(fp, "Total DOTP time [s]                  : %f\n", TOTAL_DOTP_TIME / 1000);
    logf(fp, "Total DOTP time per iter [s]         : %f\n", TOTAL_DOTP_TIME / out->niter / 1000);

    logf(fp, "Total ALLREDUCE time [s]             : %f\n", TOTAL_ALLREDUCE_TIME / 1000);
    logf(fp, "Total ALLREDUCE time per iter [s]    : %f\n", TOTAL_ALLREDUCE_TIME / out->niter / 1000);

    logf(fp, "Total SWORK time [s]                 : %f\n", TOTAL_SWORK_TIME / 1000);
    logf(fp, "Total SWORK time per iter [s]        : %f\n", TOTAL_SWORK_TIME / out->niter / 1000);

    logf(fp, "Total CUDAMEMCOPY time [s]           : %f\n", TOTAL_CUDAMEMCOPY_TIME / 1000);
    logf(fp, "Total CUDAMEMCOPY time per iter [s]  : %f\n", TOTAL_CUDAMEMCOPY_TIME / out->niter / 1000);

    logf(fp, "Total NORM time [s]                  : %f\n", TOTAL_NORM_TIME / 1000);
    logf(fp, "Total NORM time per iter [s]         : %f\n", TOTAL_NORM_TIME / out->niter / 1000);

    logf(fp, "Total TRIPLEPROD time [s]            : %f\n", TOTAL_TRIPLEPROD_TIME / 1000);
    logf(fp, "Total TRIPLEPROD time per iter [s]   : %f\n", TOTAL_TRIPLEPROD_TIME / out->niter / 1000);

    logf(fp, "Total DOUBLEMERGED time [s]          : %f\n", TOTAL_DOUBLEMERGED_TIME / 1000);
    logf(fp, "Total DOUBLEMERGED time per iter [s] : %f\n", TOTAL_DOUBLEMERGED_TIME / out->niter / 1000);

    logf(fp, "Total SOLRELAX time [s]              : %f\n", TOTAL_SOLRELAX_TIME / 1000);
    logf(fp, "Total SOLRELAX time per iter [s]     : %f\n", TOTAL_SOLRELAX_TIME / out->niter / 1000);

    logf(fp, "Total RESTPRE time [s]               : %f\n", TOTAL_RESTPRE_TIME / 1000);
    logf(fp, "Total RESTPRE time per iter [s]      : %f\n", TOTAL_RESTPRE_TIME / out->niter / 1000);

    logf(fp, "Total RESTGAMG time [s]              : %f\n", TOTAL_RESTGAMG_TIME / 1000);
    logf(fp, "Total RESTGAMG time per iter [s]     : %f\n", TOTAL_RESTGAMG_TIME / out->niter / 1000);
#endif

    // ---------------------------------------------------------------------

    if (fp) {
        fclose(fp);
    }
}
