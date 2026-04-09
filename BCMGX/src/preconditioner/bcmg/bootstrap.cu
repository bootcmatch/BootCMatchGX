#include "bootstrap.h"

#include "op/Anorm.h"
// #include "preconditioner/bcmg/BcmgPreconditionContext.h"
#include "preconditioner/bcmg/adaptiveCoarsening.h"
#include "preconditioner/prec_apply.h"
#include "utility/profiling.h"

void bcmg_apply(handles* h, CSR* Alocal, Preconditioner* pr, vector<vtype>* rhs, vector<vtype>* x, bootBuildData* bootamg_data, boot* boot_amg, applyData* amg_cycle);

namespace Bootstrap {

void innerIteration(handles* h, bootBuildData* bootamg_data, boot* boot_amg, applyData* apply_data, vector<vtype>* x, vector<vtype>* rhs, Preconditioner* pr, const InputParameters& ip)
{
    _MPI_ENV;

    /* data related to single AMG */
    buildData* amg_data = bootamg_data->amg_data;

    /* data problem (original matrix and current smooth vector) related to single AMG */
    CSR* A = amg_data->A;
    vector<vtype>* w = amg_data->w;

    /* current solution: at the end it will point to the new smooth vector */
    TRACE("Before fillWithValue(x)");
    Vector::fillWithValue(x, 1.0); // TODO bcm_VectorSetRandomValues(x,1.0);
    TRACE("After fillWithValue(x)");

    /* rhs */
    TRACE("Before fillWithValue(rhs)");
    Vector::fillWithValue(rhs, 0.0);
    TRACE("After fillWithValue(rhs)");

    /* start iterative cycle */
    TRACE("Before Anorm(rhs)");
    vtype normold = Anorm(h->cublas_h, A, x);
    TRACE("After Anorm(rhs)");
    
    vtype normnew;
    vtype conv_ratio;
    int iter = 1;
    while (iter <= bootamg_data->solver_it && normold > DBL_MIN) {
        // if (ISMASTER) {
        //     printf("iter: %d\n", iter);
        // }
        assert(A->row);
        assert(A->col);
        assert(A->val);
        assert(A->col8);

        TRACE("Before prec_apply");
        // prec_apply(h, A, rhs, x, pr, ip);
        bcmg_apply(h, A, pr, rhs, x, bootamg_data, boot_amg, apply_data);
        TRACE("After prec_apply");
        normnew = Anorm(h->cublas_h, A, x);
        conv_ratio = normnew / normold;
        normold = normnew;
        iter++;
    }

    /* update smooth vector */
    vtype alpha = (normnew > DBL_MIN)
        ? 1.0 / normnew
        : 1.0;

    if (ISMASTER) {
        printf("current smooth vector A-norm=%e\n", normnew);
    }

    /* update the smooth vector for the possible new bootstrap step */
    Vector::scale(h->cublas_h, x, alpha);
    // Vector::debug(x, "\nx (before returning from innerIteration)");
    Vector::copyTo(w, x);

    boot_amg->estimated_ratio = conv_ratio;
}

boot* bootstrap(handles* h, Preconditioner* pr, const InputParameters& ip)
{
    BEGIN_PROF(__FUNCTION__);

    _MPI_ENV;

    bootBuildData* bootamg_data = pr->bcmg.bootamg_data;
    applyData* apply_data = pr->bcmg.amg_cycle;

    boot* boot_amg = AMG::Boot::init(bootamg_data->max_hrc, 1.0);
    buildData* amg_data = bootamg_data->amg_data;

    vector<vtype>* w = amg_data->w;

    /* current solution: at the end it will point to the new smooth vector */
    vector<vtype>* x = Vector::init<vtype>(w->n, true, true);

    /* rhs */
    vector<vtype>* rhs = Vector::init<vtype>(w->n, true, true);

    int num_hrc = 0;
    while (boot_amg->estimated_ratio > bootamg_data->conv_ratio && num_hrc < bootamg_data->max_hrc) {

        CSR* A = amg_data->A;
        assert(A->row);
        assert(A->col);
        assert(A->val);
        assert(A->col8);

        // bcm_AMGHierarchyDestroy(Harray[num_hrc]);
        TRACE("Before adaptiveCoarsening");
        // Vector::debug(amg_data->w, "\nw (before adaptiveCoarsening)");
        boot_amg->H_array[num_hrc] = adaptiveCoarsening(h, amg_data, ip);
        TRACE("After adaptiveCoarsening");

        A = amg_data->A;
        assert(A->row);
        assert(A->col);
        assert(A->val);
        assert(A->col8);

        num_hrc++;
        boot_amg->n_hrc = num_hrc;

        if (ISMASTER) {
            printf("Built new hierarchy. Current number of hierarchies:%d\n", num_hrc);
        }

        // TRACE("Before initPreconditionContext");
        // Bcmg::initPreconditionContext(boot_amg->H_array[0]);
        // TRACE("After initPreconditionContext");

        A = amg_data->A;
        assert(A->row);
        assert(A->col);
        assert(A->val);
        assert(A->col8);

        // if (num_hrc == 1) {
        //     if (ip.sprec != PreconditionerType::NONE) {
        //         Bcmg::initPreconditionContext(boot_amg->H_array[0]); /* this is always done */
        //         // TODO innerIteration deve sostituire FCG::initPreconditionContext
        //     }
        // }

        if (bootamg_data->max_hrc > 1) {


            TRACE("Before innerIteration");
            innerIteration(h, bootamg_data, boot_amg, apply_data, x, rhs, pr, ip);
            TRACE("After innerIteration");
            if (ISMASTER) {
                printf("Convergence ratio of the current composite solver =%e\n", boot_amg->estimated_ratio);
            }
        }

        A = amg_data->A;
        assert(A->row);
        assert(A->col);
        assert(A->val);
        assert(A->col8);

        // // TODO remove this
        // if (num_hrc != 1) {
        //     assert(false);
        // }
    }

    Vector::free(x);
    Vector::free(rhs);

    AMG::Boot::finalize(boot_amg, num_hrc);

    END_PROF(__FUNCTION__);
    return boot_amg;
}

}
