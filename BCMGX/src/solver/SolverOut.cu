#include "SolverOut.h"

#include "preconditioner/prec_setup.h"
#include "utility/logf.h"
#include "utility/mpi.h"

#include <stdarg.h>

void dump(const char* filename, const InputParameters& ip, const CurrentParameters& cp, const Preconditioner& pr, SolverOut* out, bool append)
{
    _MPI_ENV;

    FILE* fp = fopen(filename, append ? "a" : "w");
    if (fp == NULL) {
        printf("File %s opening failed.\n", filename);
    }

    // ---------------------------------------------------------------------
    // Input
    // ---------------------------------------------------------------------

    std::string solver_type = solver_type_to_string(cp.solver_type);

    logf(fp, "nprocs                                       : %d\n", nprocs);
    logf(fp, "Solver                                       : %s\n", solver_type.c_str());

    switch (cp.solver_type) {
    case SolverType::CGS:
    case SolverType::CGS_CUBLAS:
    case SolverType::PIPELINED_CGS: {
        logf(fp, "CG Steps                                     : %d\n", cp.sstep);
        logf(fp, "Stop criterion                               : %s\n", ip.stop_criterion ? "relative" : "absolute");
        logf(fp, "Residual                                     : %s\n", ip.ru_res ? "updated" : "recomputed");
        break;
    }
    case SolverType::LSGS: {
        logf(fp, "CG Steps                                     : %d\n", cp.sstep);
        logf(fp, "Stop criterion                               : %s\n", ip.stop_criterion ? "relative" : "absolute");
        logf(fp, "GS iterations                                : %d\n", ip.gs_iterations);
        logf(fp, "GS tolerance                                 : %g\n", ip.gs_tol);
        logf(fp, "Chebyshev                                    : %s\n", ip.use_chebyshev ? "yes" : "no");
        if (ip.use_chebyshev) {
            logf(fp, "PM iterations limit                          : %d\n", ip.power_method_itnlim);
            logf(fp, "PM tolerance                                 : %g\n", ip.power_method_tol);
        }
        break;
    }
    case SolverType::CGSN: {
        logf(fp, "CG Steps                                     : %d\n", cp.sstep);
        logf(fp, "Stop criterion                               : %s\n", ip.stop_criterion ? "relative" : "absolute");
        logf(fp, "FGS                                          : %s\n", ip.use_fgs ? "yes" : "no");
        if (ip.use_fgs) {
        	logf(fp, "GS iterations                                : %d\n", ip.gs_iterations);
        	logf(fp, "GS tolerance                                 : %g\n", ip.gs_tol);
        }
        logf(fp, "Chebyshev                                    : %s\n", ip.use_chebyshev ? "yes" : "no");
        if (ip.use_chebyshev) {
            logf(fp, "PM iterations limit                          : %d\n", ip.power_method_itnlim);
            logf(fp, "PM tolerance                                 : %g\n", ip.power_method_tol);
        }
        break;
    }
    }

    std::string preconditioner_type = preconditioner_type_to_string(ip.sprec);

    logf(fp, "Tolerance                                    : %g\n", ip.rtol);
    logf(fp, "Preconditioner                               : %s\n", preconditioner_type.c_str());

    switch (ip.sprec) {
    case PreconditionerType::L1_JACOBI: {
        logf(fp, "l1-jacobi sweeps                             : %d\n", ip.l1jacsweeps);
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

    switch (ip.sprec) {
    case PreconditionerType::BCMG: {
        AMG::BootBuildData::print(fp, pr.bcmg.bootamg_data);
        AMG::ApplyData::print(fp, pr.bcmg.amg_cycle);

        logf(fp, "\nNumber of hierarchies                        : %d\n", pr.bcmg.boot_amg->n_hrc);
        logf(fp, "Estimated conv. ratio                        : %f\n", pr.bcmg.boot_amg->estimated_ratio);

        boot* boot_amg = pr.bcmg.boot_amg;
        for (int k = 0; k < boot_amg->n_hrc; k++) {
            fprintf(fp, "\nHierarchy %d\n", k);
            AMG::Hierarchy::printInfo(fp, boot_amg->H_array[k]);
        }

        fprintf(fp, "\n");
        AMG::Hierarchy::printAvgInfo(fp, boot_amg->H_array, boot_amg->n_hrc);

        // AMG::Hierarchy::printInfo(fp, pr.bcmg.H);

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

    logf(fp, "\n");
    if (fp) {
        fclose(fp);
    }
}
