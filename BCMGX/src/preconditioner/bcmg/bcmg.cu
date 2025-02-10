#include "bcmg.h"

#include "halo_communication/newoverlap.h"
#include "preconditioner/bcmg/BcmgPreconditionContext.h"
#include "preconditioner/bcmg/GAMG_cycle.h"
#include "preconditioner/bcmg/bootstrap.h"
#include "preconditioner/bcmg/matchingAggregation.h"
#include "preconditioner/prec_setup.h"
#include "utility/distribuite.h"
#include "utility/setting.h"

void bcmg_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p)
{
    buildData* amg_data = NULL;

    _MPI_ENV;

    pr->bcmg.bootamg_data = AMG::BootBuildData::initByParams(Alocal, p);
    amg_data = pr->bcmg.bootamg_data->amg_data;
    pr->bcmg.amg_cycle = AMG::ApplyData::initByParams(p);

    AMG::ApplyData::setGridSweeps(pr->bcmg.amg_cycle, amg_data->maxlevels);
    GAMGcycle::initContext(Alocal->n);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    pr->bcmg.boot_amg = Bootstrap::bootstrap(h, pr->bcmg.bootamg_data, pr->bcmg.amg_cycle, p /*, 1*/); // 1: se stai qui precondition_flag=1
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    pr->bcmg.H = pr->bcmg.boot_amg->H_array[0];

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    if (nprocs > 1) {
        for (int i = 0; i < pr->bcmg.H->num_levels - 1; i++) {
            pr->bcmg.H->R_local_array[i] = pr->bcmg.H->R_array[i];
            pr->bcmg.H->P_local_array[i] = pr->bcmg.H->P_array[i];

            // char fname[256] = {0};
            // snprintf(fname, 256, "%s/%sR%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), output_suffix.c_str(), myid);
            // CSRMatrixPrintMM(pr->bcmg.H->R_local_array[i], fname);
            // snprintf(fname, 256, "%s/%sP%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), output_suffix.c_str(), myid);
            // CSRMatrixPrintMM(pr->bcmg.H->P_local_array[i], fname);
        }
    }

    if (ISMASTER) {
        printf("CGs_prec_set: done.\n");
    }
}

void bcmg_apply(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x, cgsprec* pr, const params& p, PrecOut* out)
{
    bootBuildData* bootamg_data = pr->bcmg.bootamg_data;
    boot* boot_amg = pr->bcmg.boot_amg;
    applyData* amg_cycle = pr->bcmg.amg_cycle;

    _MPI_ENV;

    vectorCollection<vtype>* RHS = Bcmg::context.RHS_buffer;
    vectorCollection<vtype>* Xtent_local = Bcmg::context.Xtent_buffer_local;
    vectorCollection<vtype>* Xtent_2_local = Bcmg::context.Xtent_buffer_2_local;

    // static int counter = 0;

    if (bootamg_data->bootstrap_composition_type == BootstrapCompositionType::MULTIPLICATIVE) {
        for (int k = 0; k < boot_amg->n_hrc; k++) {

            Bcmg::setHrrchBufferSize(boot_amg->H_array[k]);

            Vector::copyTo(RHS->val[0], rhs);
            Vector::copyTo(Xtent_local->val[0], x);

            // dump(rhs, "%s/%srhs_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);
            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // -------------------------------------------------------------------------------------------------
            GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, RHS, Xtent_local, Xtent_2_local, 1);
            // -------------------------------------------------------------------------------------------------

            Vector::copyTo(x, Xtent_local->val[0]);

            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // counter++;
        }
    } else {
        printf("Unsupported BootstrapCompositionType\n");
        exit(1);
    }
}

void bcmg_finalize(CSR* Alocal, cgsprec* pr, const params& p)
{
    // TODO
    // Bcmg::freePreconditionContext();
    AMG::Boot::free(pr->bcmg.boot_amg);
    AMG::BootBuildData::free(pr->bcmg.bootamg_data);
    AMG::ApplyData::free(pr->bcmg.amg_cycle);
    Vector::free(GAMGcycle::Res_buffer);
}
