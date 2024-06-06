#include "bootstrap.h"

#include "preconditioner/bcmg/BcmgPreconditionContext.h"
#include "preconditioner/bcmg/matchingAggregation.h"
#include "utility/function_cnt.h"
#include "utility/timing.h"

namespace Bootstrap {

boot* bootstrap(handles* h, bootBuildData* bootamg_data, applyData* apply_data, const params& p)
{

    PUSH_RANGE(__func__, 3)

    _MPI_ENV;
    TIMER_DEF;
    double initfinalize_time = 0;
    TIMER_START;
    boot* boot_amg = AMG::Boot::init(bootamg_data->max_hrc, 1.0);
    TIMER_STOP;
    initfinalize_time += TIMER_ELAPSED;
    buildData* amg_data;
    amg_data = bootamg_data->amg_data;

    int num_hrc = 0;
    while (boot_amg->estimated_ratio > bootamg_data->conv_ratio && num_hrc < bootamg_data->max_hrc) {

        TIMER_START;
        boot_amg->H_array[num_hrc] = adaptiveCoarsening(h, amg_data, p); /* this is always done (look at AMG folder) */
        TIMER_STOP;
        if (ISMASTER) {
            printf("adaptiveCoarsening time: %g\n", TIMER_ELAPSED);
        }
        num_hrc++;
        boot_amg->n_hrc = num_hrc;

        if (VERBOSE > 0) {
            printf("Built new hierarchy. Current number of hierarchies:%d\n", num_hrc);
        }

        if (num_hrc == 1) {
            if (p.sprec != PreconditionerType::NONE) {
                TIMER_START;
                // init FGC buffers
                Bcmg::initPreconditionContext(boot_amg->H_array[0]); /* this is always done */

                TIMER_STOP;
                if (ISMASTER) {
                    printf("initPreconditionerContext time: %g\n", TIMER_ELAPSED);
                }
            }
        } else {
            assert(false);
        }
    }

    TIMER_START;
    AMG::Boot::finalize(boot_amg, num_hrc);
    TIMER_STOP;
    if (ISMASTER) {
        printf("initfinalize time: %g\n", TIMER_ELAPSED);
    }

    POP_RANGE
    return boot_amg;
}

}
