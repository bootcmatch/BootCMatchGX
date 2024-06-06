#include "bcmg.h"

#include "basic_kernel/halo_communication/newoverlap.h"
#include "custom_cudamalloc/custom_cudamalloc.h"
#include "preconditioner/bcmg/BcmgPreconditionContext.h"
#include "preconditioner/bcmg/GAMG_cycle.h"
#include "preconditioner/bcmg/bootstrap.h"
#include "preconditioner/bcmg/matchingAggregation.h"
#include "preconditioner/prec_setup.h"
#include "utility/distribuite.h"
#include "utility/function_cnt.h"
#include "utility/setting.h"

void bcmg_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p)
{

    buildData* amg_data = NULL;
    itype levelc = 0;

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
    Vectorinit_CNT
        CustomCudaMalloc::free(1);
    CustomCudaMalloc::free(2);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    levelc = pr->bcmg.H->num_levels - 1;

    if (nprocs > 1) {
        for (int i = 0; i < pr->bcmg.H->num_levels - 1; i++) {
            pr->bcmg.H->R_local_array[i] = pr->bcmg.H->R_array[i];
            pr->bcmg.H->P_local_array[i] = pr->bcmg.H->P_array[i];
        }
        itype hn = pr->bcmg.H->num_levels;
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

    PUSH_RANGE(__func__, 4)

    _MPI_ENV;

    vectorCollection<vtype>* RHS = Bcmg::context.RHS_buffer;
    vectorCollection<vtype>* Xtent_local = Bcmg::context.Xtent_buffer_local;
    vectorCollection<vtype>* Xtent_2_local = Bcmg::context.Xtent_buffer_2_local;

    if (bootamg_data->bootstrap_composition_type == BootstrapCompositionType::MULTIPLICATIVE) {
        for (int k = 0; k < boot_amg->n_hrc; k++) {

            Bcmg::setHrrchBufferSize(boot_amg->H_array[k]);

            Vector::copyTo(RHS->val[0], rhs);
            Vector::copyTo(Xtent_local->val[0], x);

            // -------------------------------------------------------------------------------------------------
            GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, RHS, Xtent_local, Xtent_2_local, 1);
            // -------------------------------------------------------------------------------------------------

            Vector::copyTo(x, Xtent_local->val[0]);
        }
    } else {
        printf("Unsupported BootstrapCompositionType\n");
        exit(1);
    }

    POP_RANGE
}

void bcmg_finalize(handles* h, CSR* Alocal, cgsprec* pr, const params& p)
{
    // TODO
    Bcmg::freePreconditionContext();
}
