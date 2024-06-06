#include "preconditioner/bcmg/BcmgPreconditionContext.h"
#include "utility/function_cnt.h"
#include "utility/utils.h"

#define XTENTFACT 1

BcmgPreconditionContext Bcmg::context;

void Bcmg::initPreconditionContext(hierarchy* hrrch)
{
    PUSH_RANGE(__func__, 4)
    _MPI_ENV;
    Bcmg::context.hrrch = hrrch;
    unsigned int num_levels = hrrch->num_levels;

    Bcmg::context.max_level_nums = num_levels;
    Bcmg::context.max_coarse_size = (itype*)Malloc(num_levels * sizeof(int));
    assert(Bcmg::context.max_coarse_size != NULL);

    vectorCollection<vtype>* RHS_buffer = Vector::Collection::init<vtype>(num_levels);

    vectorCollection<vtype>* Xtent_buffer_local = Vector::Collection::init<vtype>(num_levels);
    vectorCollection<vtype>* Xtent_buffer_2_local = Vector::Collection::init<vtype>(num_levels);

    // !skip the first
    for (int i = 0; i < num_levels; i++) {
        itype n_i = hrrch->A_array[i]->n;
        itype n_i_full = hrrch->A_array[i]->full_n;
        Bcmg::context.max_coarse_size[i] = n_i;
        Vectorinit_CNT
            RHS_buffer->val[i]
            = Vector::init<vtype>(n_i, true, true);
        if (nprocs > 1) {
            n_i = (int)(hrrch->A_array[i]->n * XTENTFACT); /* Massimo March 13 2024. Was hrrch->A_array[i]->n; */
        }
        Vectorinit_CNT
            Xtent_buffer_local->val[i]
            = Vector::init<vtype>((i != num_levels - 1) ? n_i : n_i_full, true, true);
        Vectorinit_CNT
            Xtent_buffer_2_local->val[i]
            = Vector::init<vtype>((i != num_levels - 1) ? n_i : n_i_full, true, true);
        Vector::fillWithValue(Xtent_buffer_local->val[i], 0.);
        Vector::fillWithValue(Xtent_buffer_2_local->val[i], 0.);
    }

    Bcmg::context.RHS_buffer = RHS_buffer;

    Bcmg::context.Xtent_buffer_local = Xtent_buffer_local;
    Bcmg::context.Xtent_buffer_2_local = Xtent_buffer_2_local;

    POP_RANGE
}

void Bcmg::setHrrchBufferSize(hierarchy* hrrch)
{
    int num_levels = hrrch->num_levels;
    assert(num_levels <= Bcmg::context.max_level_nums);
    _MPI_ENV;
    for (int i = 0; i < num_levels; i++) {
        itype n_i = hrrch->A_array[i]->n;

        itype n_i_full = hrrch->A_array[i]->full_n;

        if (n_i > Bcmg::context.max_coarse_size[i]) {
            // make i-level's buffer bigger

            Bcmg::context.max_coarse_size[i] = n_i;
            Vector::free(Bcmg::context.RHS_buffer->val[i]);
            Vectorinit_CNT
                Bcmg::context.RHS_buffer->val[i]
                = Vector::init<vtype>(n_i, true, true);

            Vector::free(Bcmg::context.Xtent_buffer_local->val[i]);
            Vector::free(Bcmg::context.Xtent_buffer_2_local->val[i]);

            if (nprocs > 1) {
                n_i = (int)(hrrch->A_array[i]->n * XTENTFACT); /* Massimo March 13 2024. Was hrrch->A_array[i]->n; */
            }
            if (i == num_levels - 1) {
                Vectorinit_CNT
                    Bcmg::context.Xtent_buffer_local->val[i]
                    = Vector::init<vtype>(n_i_full, true, true);
                Vectorinit_CNT
                    Bcmg::context.Xtent_buffer_2_local->val[i]
                    = Vector::init<vtype>(n_i_full, true, true);
            } else {
                Vectorinit_CNT
                    Bcmg::context.Xtent_buffer_local->val[i]
                    = Vector::init<vtype>(n_i, true, true);
                Vectorinit_CNT
                    Bcmg::context.Xtent_buffer_2_local->val[i]
                    = Vector::init<vtype>(n_i, true, true);
            }

        } else {
            Bcmg::context.RHS_buffer->val[i]->n = n_i;
            if (nprocs > 1) {
                n_i = (int)(hrrch->A_array[i]->n * XTENTFACT); /* Massimo March 13 2024. Was hrrch->A_array[i]->n; */
            }
            Bcmg::context.Xtent_buffer_local->val[i]->n = n_i;
            Bcmg::context.Xtent_buffer_2_local->val[i]->n = n_i;
        }
    }
}

void Bcmg::freePreconditionContext()
{

    free(Bcmg::context.max_coarse_size);
    Vector::Collection::free(Bcmg::context.RHS_buffer);

    Vector::Collection::free(Bcmg::context.Xtent_buffer_local);
    Vector::Collection::free(Bcmg::context.Xtent_buffer_2_local);
}
