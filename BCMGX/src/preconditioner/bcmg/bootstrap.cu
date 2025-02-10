#include "bootstrap.h"

#include "preconditioner/bcmg/BcmgPreconditionContext.h"
#include "preconditioner/bcmg/matchingAggregation.h"
#include "utility/profiling.h"

#define VERBOSE 0

namespace Bootstrap {

boot* bootstrap(handles* h, bootBuildData* bootamg_data, applyData* apply_data, const params& p)
{
    BEGIN_PROF(__FUNCTION__);

    _MPI_ENV;

    boot* boot_amg = AMG::Boot::init(bootamg_data->max_hrc, 1.0);
    buildData* amg_data = bootamg_data->amg_data;

    int num_hrc = 0;
    while (boot_amg->estimated_ratio > bootamg_data->conv_ratio && num_hrc < bootamg_data->max_hrc) {

        boot_amg->H_array[num_hrc] = adaptiveCoarsening(h, amg_data, p);

        num_hrc++;
        boot_amg->n_hrc = num_hrc;

        if (VERBOSE > 0) {
            printf("Built new hierarchy. Current number of hierarchies:%d\n", num_hrc);
        }

        if (num_hrc == 1) {
            if (p.sprec != PreconditionerType::NONE) {
                Bcmg::initPreconditionContext(boot_amg->H_array[0]); /* this is always done */
                // TODO innerIteration deve sostituire FCG::initPreconditionContext
            }
        } else {
            assert(false);
        }
    }

    AMG::Boot::finalize(boot_amg, num_hrc);

    END_PROF(__FUNCTION__);
    return boot_amg;
}

}
