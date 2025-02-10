#include "SolverOut.h"

#include "preconditioner/prec_setup.h"
#include "utility/logf.h"
#include "utility/mpi.h"

#include <stdarg.h>

void dump(const char* filename, const params& p, const cgsprec& pr, SolverOut* out)
{
    _MPI_ENV;

    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("File %s opening failed.\n", filename);
    }

    // ---------------------------------------------------------------------
    // Input
    // ---------------------------------------------------------------------

    std::string solver_type = solver_type_to_string(p.solver_type);

    logf(fp, "nprocs                                       : %d\n", nprocs);
    logf(fp, "Solver                                       : %s\n", solver_type.c_str());

    switch (p.solver_type) {
    case SolverType::CGS:
    case SolverType::CGS_CUBLAS:
    case SolverType::PIPELINED_CGS: {
        logf(fp, "CG Steps                                     : %d\n", p.sstep);
        logf(fp, "Stop criterion                               : %s\n", p.stop_criterion ? "relative" : "absolute");
        logf(fp, "Residual                                     : %s\n", p.ru_res ? "updated" : "recomputed");
        break;
    }
    }

    std::string preconditioner_type = preconditioner_type_to_string(p.sprec);

    logf(fp, "Tolerance                                    : %g\n", p.rtol);
    logf(fp, "Preconditioner                               : %s\n", preconditioner_type.c_str());

    switch (p.sprec) {
    case PreconditionerType::L1_JACOBI: {
        logf(fp, "l1-jacobi sweeps                             : %d\n", p.l1jacsweeps);
        break;
    }
    }

    // ---------------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------------

    if (out->retv < 0) {
        logf(fp, "*** Exceeded max iter ***\n");
    }
    if (out->retv > 0) {
        logf(fp, "*** Non positive definite matrix found ***\n");
    }

    logf(fp, "Total iterations                             : %d\n", out->niter);
    logf(fp, "Global dimension (number of rows)            : %ld\n", out->full_n);
    logf(fp, "Local dimension (number of rows)             : %d\n", out->local_n);
    logf(fp, "Initial residual                             : %.10lf\n", out->del0);
    logf(fp, "Final residual                               : %.10lf\n", out->exitRes);

    switch (p.sprec) {
    case PreconditionerType::BCMG: {
        logf(fp, "Number of hierarchies                        : %d\n", pr.bcmg.boot_amg->n_hrc);
        logf(fp, "Estimated ratio                              : %f\n", pr.bcmg.boot_amg->estimated_ratio);

        AMG::BootBuildData::print(fp, pr.bcmg.bootamg_data);
        AMG::ApplyData::print(fp, pr.bcmg.amg_cycle);
        AMG::Hierarchy::printInfo(fp, pr.bcmg.H);

        // AMG::Hierarchy::printInfo(stderr, hrrch);
        // Eval::printMetaData("agg;level_number", level, 0);
        // Eval::printMetaData("agg;avg_coarse_ratio", hrrch->avg_cratio, 1);
        // Eval::printMetaData("agg;OpCmplx", hrrch->op_cmplx, 1);
        // Eval::printMetaData("agg;OpCmplxW", hrrch->op_wcmplx, 1);
        // Eval::printMetaData("agg;coarsest_size", hrrch->A_array[level - 1]->full_n, 0);
        // Eval::printMetaData("agg;total_mul_num", MUL_NUM, 0);

        break;
    }
    }

    // ---------------------------------------------------------------------

    // dumpMetrics(fp, p, out);

    // ---------------------------------------------------------------------

    if (fp) {
        fclose(fp);
    }
}
