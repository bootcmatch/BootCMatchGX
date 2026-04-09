#include "bcmg.h"

#include "halo_communication/newoverlap.h"
#include "op/basic.h"
#include "op/spspmpi.h"
// #include "preconditioner/bcmg/BcmgPreconditionContext.h"
#include "preconditioner/bcmg/GAMG_cycle.h"
#include "preconditioner/bcmg/bootstrap.h"
#include "preconditioner/bcmg/matchingAggregation.h"
#include "preconditioner/prec_setup.h"
#include "utility/distribute.h"
#include "utility/setting.h"

#define XTENTFACT 1

void bcmg_setup(handles* h, CSR* Alocal, Preconditioner* pr, const InputParameters& ip)
{
    buildData* amg_data = NULL;

    _MPI_ENV;

    pr->bcmg.bootamg_data = AMG::BootBuildData::initByParams(Alocal, ip);
    amg_data = pr->bcmg.bootamg_data->amg_data;
    pr->bcmg.amg_cycle = AMG::ApplyData::initByParams(ip);

    AMG::ApplyData::setGridSweeps(pr->bcmg.amg_cycle, amg_data->maxlevels);
    GAMGcycle::initContext(Alocal->n);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    pr->bcmg.boot_amg = Bootstrap::bootstrap(h, pr, ip);
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    pr->bcmg.H = pr->bcmg.boot_amg->H_array[0];

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    if (SPSP_LIB == SpSpLib::CUSPARSE) {
        void spgemmcusparseFree();
        spgemmcusparseFree();
    }

    if (ISMASTER) {
        printf("CGs_prec_set: done.\n");
    }
}

void bcmg_memset(Preconditioner *pr, hierarchy* h) {
    _MPI_ENV;

    if (h->num_levels > pr->bcmg.max_num_levels) {
        Vector::Collection::free(pr->bcmg.RHS);
        Vector::Collection::free(pr->bcmg.Xtent_local);
        Vector::Collection::free(pr->bcmg.Xtent_2_local);
        pr->bcmg.RHS            = Vector::Collection::init<vtype>(h->num_levels);
        pr->bcmg.Xtent_local    = Vector::Collection::init<vtype>(h->num_levels);
        pr->bcmg.Xtent_2_local  = Vector::Collection::init<vtype>(h->num_levels);
        pr->bcmg.max_num_levels = h->num_levels;
    }

    for (int i = 0; i < h->num_levels; i++) {
        itype n_i      = h->A_array[i]->n;
        itype n_i_full = h->A_array[i]->full_n;

        if (nprocs > 1) {
            n_i = (int)(n_i * XTENTFACT);
        }

        if (pr->bcmg.RHS->val[i] == NULL) {
            pr->bcmg.RHS->val[i] = Vector::init<vtype>(n_i, true, true);
        } else if (n_i > pr->bcmg.RHS->val[i]->n) {
            Vector::free(pr->bcmg.RHS->val[i]);
            pr->bcmg.RHS->val[i] = Vector::init<vtype>(n_i, true, true);
        } else {
            pr->bcmg.RHS->val[i]->n = n_i;
        }

        if (i == h->num_levels - 1) {

            if (pr->bcmg.Xtent_local->val[i] == NULL) {
                pr->bcmg.Xtent_local->val[i] = Vector::init<vtype>(n_i_full, true, true);
            } else if (n_i_full > pr->bcmg.Xtent_local->val[i]->n) {
                Vector::free(pr->bcmg.Xtent_local->val[i]);
                pr->bcmg.Xtent_local->val[i] = Vector::init<vtype>(n_i_full, true, true);
            } else {
                pr->bcmg.Xtent_local->val[i]->n = n_i_full;
            }

            if (pr->bcmg.Xtent_2_local->val[i] == NULL) {
                pr->bcmg.Xtent_2_local->val[i] = Vector::init<vtype>(n_i_full, true, true);
            } else if (n_i_full > pr->bcmg.Xtent_2_local->val[i]->n) {
                Vector::free(pr->bcmg.Xtent_2_local->val[i]);
                pr->bcmg.Xtent_2_local->val[i] = Vector::init<vtype>(n_i_full, true, true);
            } else {
                pr->bcmg.Xtent_2_local->val[i]->n = n_i_full;
            }

        } else {

            if (pr->bcmg.Xtent_local->val[i] == NULL) {
                pr->bcmg.Xtent_local->val[i] = Vector::init<vtype>(n_i, true, true);
            } else if (n_i > pr->bcmg.Xtent_local->val[i]->n) {
                Vector::free(pr->bcmg.Xtent_local->val[i]);
                pr->bcmg.Xtent_local->val[i] = Vector::init<vtype>(n_i, true, true);
            } else {
                pr->bcmg.Xtent_local->val[i]->n = n_i;
            }

            if (pr->bcmg.Xtent_2_local->val[i] == NULL) {
                pr->bcmg.Xtent_2_local->val[i] = Vector::init<vtype>(n_i, true, true);
            } else if (n_i > pr->bcmg.Xtent_2_local->val[i]->n) {
                Vector::free(pr->bcmg.Xtent_2_local->val[i]);
                pr->bcmg.Xtent_2_local->val[i] = Vector::init<vtype>(n_i, true, true);
            } else {
                pr->bcmg.Xtent_2_local->val[i]->n = n_i;
            }

        }


    }
}

void bcmg_apply(handles* h, CSR* Alocal, Preconditioner* pr, vector<vtype>* rhs, vector<vtype>* x, bootBuildData* bootamg_data, boot* boot_amg, applyData* amg_cycle /*Preconditioner* pr, const InputParameters& ip*/)
{
    _MPI_ENV;

    // static int counter = 0;

    switch (bootamg_data->bootstrap_composition_type) {
    case BootstrapCompositionType::MULTIPLICATIVE: {
        TRACE("BootstrapCompositionType::MULTIPLICATIVE - Before for");
        for (int k = 0; k < boot_amg->n_hrc; k++) {

            TRACE("Before bcmg_memset(%d)", k);
            bcmg_memset(pr, boot_amg->H_array[k]);
            TRACE("After bcmg_memset(%d)", k);

            TRACE("Before copyTo(%d)", k);
            Vector::copyTo(pr->bcmg.RHS->val[0], rhs);
            Vector::copyTo(pr->bcmg.Xtent_local->val[0], x);
            TRACE("After copyTo(%d)", k);

            // dump(rhs, "%s/%srhs_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);
            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // -------------------------------------------------------------------------------------------------
            TRACE("Before GAMG_cycle(%d)", k);
            GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, pr->bcmg.RHS, pr->bcmg.Xtent_local, pr->bcmg.Xtent_2_local, 1);
            TRACE("After GAMG_cycle(%d)", k);
            // -------------------------------------------------------------------------------------------------

            TRACE("Before copyTo(%d)", k);
            Vector::copyTo(x, pr->bcmg.Xtent_local->val[0]);
            TRACE("After copyTo(%d)", k);

            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // counter++;
        }
        TRACE("BootstrapCompositionType::MULTIPLICATIVE - After for");
        break;
    }

    case BootstrapCompositionType::SYMMETRIZED_MULTIPLICATIVE: {
        TRACE("BootstrapCompositionType::SYMMETRIZED_MULTIPLICATIVE");
        for (int k = 0; k < boot_amg->n_hrc; k++) {

            TRACE("Before bcmg_memset(%d)", k);
            bcmg_memset(pr, boot_amg->H_array[k]);
            TRACE("After bcmg_memset(%d)", k);

            Vector::copyTo(pr->bcmg.RHS->val[0], rhs);
            Vector::copyTo(pr->bcmg.Xtent_local->val[0], x);

            // dump(rhs, "%s/%srhs_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);
            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // -------------------------------------------------------------------------------------------------
            GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, pr->bcmg.RHS, pr->bcmg.Xtent_local, pr->bcmg.Xtent_2_local, 1);
            // -------------------------------------------------------------------------------------------------

            Vector::copyTo(x, pr->bcmg.Xtent_local->val[0]);

            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // counter++;
        }

        for (int k = boot_amg->n_hrc - 1; k >= 0; k--) {

            TRACE("Before bcmg_memset(%d)", k);
            bcmg_memset(pr, boot_amg->H_array[k]);
            TRACE("After bcmg_memset(%d)", k);

            Vector::copyTo(pr->bcmg.RHS->val[0], rhs);
            Vector::copyTo(pr->bcmg.Xtent_local->val[0], x);

            // dump(rhs, "%s/%srhs_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);
            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // -------------------------------------------------------------------------------------------------
            GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, pr->bcmg.RHS, pr->bcmg.Xtent_local, pr->bcmg.Xtent_2_local, 1);
            // -------------------------------------------------------------------------------------------------

            Vector::copyTo(x, pr->bcmg.Xtent_local->val[0]);

            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // counter++;
        }

        break;
    }

    case BootstrapCompositionType::ADDITIVE: {
        TRACE("BootstrapCompositionType::ADDITIVE");
        vtype alpha = 0.;
        vector<vtype>* xadd = Vector::init<vtype>(Alocal->n, true, true); // TODO this is inefficient: move init/free outside apply
        Vector::fillWithValue(xadd, 0.);

        for (int k = 0; k < boot_amg->n_hrc; k++) {

            TRACE("Before bcmg_memset(%d)", k);
            bcmg_memset(pr, boot_amg->H_array[k]);
            TRACE("After bcmg_memset(%d)", k);

            Vector::copyTo(pr->bcmg.RHS->val[0], rhs);
            Vector::copyTo(pr->bcmg.Xtent_local->val[0], x);

            // dump(rhs, "%s/%srhs_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);
            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // -------------------------------------------------------------------------------------------------
            GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, pr->bcmg.RHS, pr->bcmg.Xtent_local, pr->bcmg.Xtent_2_local, 1);
            // -------------------------------------------------------------------------------------------------

            my_axpby(pr->bcmg.Xtent_local->val[0]->val, pr->bcmg.Xtent_local->val[0]->n, xadd->val, 1.0, 1.0);

            // dump(x, "%s/%sx_%d%s_%d.mtx", output_dir.c_str(), output_prefix.c_str(), counter, output_suffix.c_str(), myid);

            // counter++;
        }

        alpha = 1.0 / boot_amg->n_hrc;
        Vector::scale(h->cublas_h, xadd, alpha);
        Vector::copyTo(x, xadd);
        Vector::free(xadd); // TODO this is inefficient: move init/free outside apply

        break;
    }

    default: {
        DIE("Unsupported BootstrapCompositionType\n");
    }
    }

    TRACE("BootstrapCompositionType [done]");
}

void bcmg_apply(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x, Preconditioner* pr, const InputParameters& ip)
{
    bootBuildData* bootamg_data = pr->bcmg.bootamg_data;
    boot* boot_amg = pr->bcmg.boot_amg;
    applyData* amg_cycle = pr->bcmg.amg_cycle;

    bcmg_apply(h, Alocal, pr, rhs, x, bootamg_data, boot_amg, amg_cycle);
}

void bcmg_finalize(CSR* Alocal, Preconditioner* pr, const InputParameters& p)
{
    // TODO
    // Bcmg::freePreconditionContext();

    Vector::Collection::free(pr->bcmg.RHS);
    Vector::Collection::free(pr->bcmg.Xtent_local);
    Vector::Collection::free(pr->bcmg.Xtent_2_local);

    AMG::Boot::free(pr->bcmg.boot_amg);
    AMG::BootBuildData::free(pr->bcmg.bootamg_data);
    AMG::ApplyData::free(pr->bcmg.amg_cycle);
    Vector::free(GAMGcycle::Res_buffer);
}
