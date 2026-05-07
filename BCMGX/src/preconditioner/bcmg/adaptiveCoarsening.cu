#include "adaptiveCoarsening.h"

#include "halo_communication/halo_communication.h"
#include "preconditioner/bcmg/matchingAggregation.h"
#include "utility/col8.h"
#include "utility/memory.h"
#include "utility/profiling.h"

// -------- AH glob --------
itype* AH_glob_row = NULL;
itype* AH_glob_col = NULL;
gsstype* AH_glob_col8 = NULL;
vtype* AH_glob_val = NULL;

extern itype* iPtemp1;
extern vtype* vPtemp1;
extern itype* iAtemp1;
extern vtype* vAtemp1;
extern itype* idevtemp1;
extern vtype* vdevtemp1;
extern itype* idevtemp2;

#define DEBUG 0
#define CHECK_INDECES 0

bool shrink_col(CSR*, CSR*);

// return S = I - omega invD A
CSR *getSmoother(CSR *A, vector<vtype> *diag) {
    CSR *invDA = CSRm::diagonalScaling(A, diag);
    // CSRm::debug(invDA, stderr);
    
    double omega = 4.0 / (3.0 * CSRm::globalInfinityNorm(invDA));
    _MPI_ENV;
    if (ISMASTER) {
        printf("omega = %lf\n", omega);
    }

    CSRm::forEach(invDA, [omega]__device__(CSR *invDA, itype irow, size_t innz) {
        gsstype col8 = invDA->col8[innz];
        invDA->val[innz] *= -omega;
        if (irow == col8) {
            invDA->val[innz] += 1;
        }
        // vtype val = invDA->val[innz];
        // printf("CSR[%ld, %ld] = %lf\n", (gsstype)(irow + invDA->row_shift), col8 - invDA->col_shifted, val);
    });

    return invDA;
}

void relaxPrepare(handles* h, int level, hierarchy* hrrch, buildData* amg_data)
{
    BEGIN_PROF(__FUNCTION__);

    CSR* A = hrrch->A_array[level];

    RelaxType relax_type = amg_data->CRrelax_type;

    if (relax_type == RelaxType::L1_JACOBI) {
        // L1 smoother
        if (hrrch->D_array[level] != NULL) {
            Vector::free(hrrch->D_array[level]);
        }
        hrrch->D_array[level] = CSRm::diag(A);

        if (hrrch->M_array[level] != NULL) {
            Vector::free(hrrch->M_array[level]);
        }
        hrrch->M_array[level] = CSRm::absoluteRowSum(A, NULL);
    }

    END_PROF(__FUNCTION__);
}

void adjustShift(CSR *A) {
    _MPI_ENV;
    if (A->row_shift && !A->col_shifted) {
        unsigned long global_m = A->m;
        unsigned long local_m = global_m / nprocs;
        long shift = get_mapped_task(myid) * local_m;
        CSRm::shift_cols(A, -shift);
        A->col_shifted = -shift;
        A->row_shift = shift;
    }
}

void printHierarchyInfo(hierarchy *hrrch, int level) {
    _MPI_ENV;
    // int level = hrrch->num_levels;
    int j = 0;
    for (j = 0; j < level; j++) {
        CSR *oldP = hrrch->P_array[j];
        CSR *oldR = hrrch->R_array[j];
        CSR *oldA = hrrch->A_array[j];

        if (oldP) fprintf(stderr, "myid = %d, P[%d] = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, j, oldP->n,  oldP->full_n,  oldP->m,  oldP->row_shift,  oldP->col_shifted);
        if (oldR) fprintf(stderr, "myid = %d, R[%d] = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, j, oldR->n,  oldR->full_n,  oldR->m,  oldR->row_shift,  oldR->col_shifted);
        if (oldA) fprintf(stderr, "myid = %d, A[%d] = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, j, oldA->n,  oldA->full_n,  oldA->m,  oldA->row_shift,  oldA->col_shifted);
    }
}

/**
 * @brief Function for adaptive coarsening in a multilevel solver hierarchy.
 *
 * This function builds the multilevel hierarchy by performing adaptive coarsening based on the AMG (Algebraic Multigrid) method.
 * It allocates memory for various buffers, sets up the communication patterns for the solver, and computes the prolongation
 * and restriction operators for the AMG hierarchy.
 *
 * @param h A pointer to the `handles` structure which contains solver-related data.
 * @param amg_data A pointer to the `buildData` structure which contains the matrix and related data.
 * @param p A reference to the `InputParameters` structure which holds solver parameters such as memory allocation size and preconditioner type.
 *
 * @return A pointer to the `hierarchy` structure which holds the multilevel hierarchy and its components.
 *
 * @note This function also involves device memory management for CUDA-based operations, and it manages communication patterns
 *       when multiple processes are involved.
 */
hierarchy* adaptiveCoarsening(handles* h, buildData* amg_data, const InputParameters& p)
{
    // trace_enabled = true;

    BEGIN_PROF(__FUNCTION__);

    _MPI_ENV;

    // BEGIN_PROF("MEM");
    CSR* A = amg_data->A;
    assert(A->row);
    assert(A->col);
    assert(A->val);
    assert(A->col8);

    // init memory pool
    amg_data->ws_buffer = Vector::init<vtype>(A->n, true, true);
    amg_data->mutex_buffer = Vector::init<itype>(A->n, true, true);
    amg_data->_M = Vector::init<itype>(A->n, true, true);

    iPtemp1 = NULL;
    vPtemp1 = NULL;

    size_t size = p.mem_alloc_size;

    iAtemp1 = CUDA_MALLOC_HOST(itype, size, true);
    vAtemp1 = CUDA_MALLOC_HOST(vtype, size, true);

    idevtemp1 = CUDA_MALLOC(itype, size, true);
    vdevtemp1 = CUDA_MALLOC(vtype, size, true);
    idevtemp2 = CUDA_MALLOC(itype, size, true);

    // -------- AH glob --------
    AH_glob_row = CUDA_MALLOC(itype, A->n + 1, true);
    AH_glob_col = CUDA_MALLOC(itype, A->nnz, true);
    AH_glob_col8 = CUDA_MALLOC(gsstype, A->nnz, true);
    AH_glob_val = CUDA_MALLOC(vtype, A->nnz, true);
    // -------------------------

    vector<vtype>* w = amg_data->w;
    vector<vtype>* w_temp = NULL;

    if (w->on_the_device) {
        w_temp = Vector::init<vtype>(w->n, true, true);
        cudaError_t err = cudaMemcpy(w_temp->val, w->val, w_temp->n * sizeof(vtype), cudaMemcpyDeviceToDevice);
        CHECK_DEVICE(err);
    } else {
        w_temp = Vector::clone(w);
    }
    // ----------------------------//

    CSR *P = NULL, *R = NULL;
    hierarchy* hrrch = AMG::Hierarchy::init(amg_data->maxlevels + 1);
    hrrch->A_array[0] = A;
    sprintf(A->name, "A[0]");
    if (nprocs > 1) {
        TRACE("Before haloSetup(hrrch->A_array[0], NULL);");
        halo_info hi = haloSetup(hrrch->A_array[0], NULL);
        hrrch->A_array[0]->halo = hi;
        TRACE("After haloSetup(hrrch->A_array[0], NULL);");
    }
    relaxPrepare(h, 0, hrrch, amg_data);

    // {
    //     char fname[1024] = {0};
    //     snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), "", "A", 0, myid, nprocs);
    //     CSRm::printMM(hrrch->A_array[0], fname, false);
    // }

    // END_PROF("MEM");

    // compute comunication patterns for solver

    vtype avcoarseratio = 0.;
    amg_data->ftcoarse = 1;

    int level = 0;
    for (level = 1; level < amg_data->maxlevels;) {

        TRACE("Before matchingAggregation");
        hrrch->A_array[level] = matchingAggregation(h, amg_data, hrrch->A_array[level - 1], &w_temp, &P, &R, level - 1);
        sprintf(hrrch->A_array[level]->name, "A[%d]", level);
        TRACE("After matchingAggregation\n");

        hrrch->P_array[level - 1] = P;
        hrrch->R_array[level - 1] = R;
        sprintf(P->name, "P[%d]", level - 1);
        sprintf(R->name, "R[%d]", level - 1);

        #if 0
        {
            char fname[256] = {0};
            snprintf(fname, 256, "%s/%s%s_global.mtx", output_dir.c_str(), output_prefix.c_str(), P->name);
            CSRm::printGlobalMM(P, fname);

            snprintf(fname, 256, "%s/%s%s_local%d.mtx", output_dir.c_str(), output_prefix.c_str(), P->name, myid);
            CSRm::printMM(P, fname);
        }
        #endif

        // CSRm::printUniqueColumns(P, P->name, stderr);

        #if DEBUG
        if (log_file) {
            fprintf(log_file, "P[%d]: n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",
                level - 1, P->n, P->full_n, P->m, P->row_shift, P->col_shifted);
        }
        #endif

        // TRACE("Before shift_cols");
        
        // // if (myid != 0 && R->col_shifted == 0) {
        // //     R->bitcol = NULL;
        // //     R->bitcolsize = 0;
        // //     CSRm::shift_cols(R, -(hrrch->A_array[level - 1]->row_shift));
        // //     R->col_shifted = -(hrrch->A_array[level - 1]->row_shift);
        // // }
        // TRACE("After shift_cols");

        TRACE("Before shrink_col");
        shrink_col(hrrch->A_array[level - 1], NULL);
        shrink_col(hrrch->P_array[level - 1], hrrch->A_array[level]);
        if (level != hrrch->num_levels - 1) {
            shrink_col(hrrch->R_array[level - 1], hrrch->A_array[level - 1]);
        }
        TRACE("After shrink_col");

        if (nprocs > 1) {
            halo_info hi = haloSetup(hrrch->A_array[level], NULL);
            hrrch->A_array[level]->halo = hi;

            hi = haloSetup(hrrch->P_array[level - 1], NULL);
            hrrch->P_array[level - 1]->halo = hi;

            if (level != hrrch->num_levels - 1) {
                halo_info hi = haloSetup(hrrch->R_array[level - 1], NULL);
                hrrch->R_array[level - 1]->halo = hi;
            }
        }

        // {
        //     char fname[1024] = {0};
        //     snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), "", "P", level - 1, myid, nprocs);
        //     CSRm::printMM(P, fname, false);
        // }
        // {
        //     char fname[1024] = {0};
        //     snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), "", "R", level - 1, myid, nprocs);
        //     CSRm::printMM(R, fname, false);
        // }
        // {
        //     char fname[1024] = {0};
        //     snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), "", "A", level, myid, nprocs);
        //     CSRm::printMM(hrrch->A_array[level], fname, false);
        // }

        TRACE("Before relaxPrepare");
        if (!amg_data->agg_interp_type) {
            relaxPrepare(h, level, hrrch, amg_data);
        }
        TRACE("After relaxPrepare");

        vtype size_coarse = hrrch->A_array[level]->full_n;
        vtype coarse_ratio = hrrch->A_array[level - 1]->full_n / size_coarse;
        avcoarseratio = avcoarseratio + coarse_ratio;
        level++;

        if (size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize) {
            break;
        }
    }
    shrink_col(hrrch->A_array[level - 1], NULL);

    #if 0
    for (int j = 0; j < level - 1; j++) {
        {
            char fname[1024] = {0};
            snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "hrrch_A", j, myid, nprocs);
            CSRm::printMMsimple(hrrch->A_array[j], fname, false);
        }
        {
            char fname[1024] = {0};
            snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "hrrch_P", j, myid, nprocs);
            CSRm::printMMsimple(hrrch->P_array[j], fname, false);
        }
        {
            char fname[1024] = {0};
            snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "hrrch_R", j, myid, nprocs);
            CSRm::printMMsimple(hrrch->R_array[j], fname, false);
        }
    }
    #endif

    #if 0
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    if (ISMASTER) {
        fprintf(stderr, "Base hierarchy\n");
    }
    //printHierarchyInfo(hrrch, level);
    #endif

    if (p.smoothed_prolungator) {
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
        if (ISMASTER) {
            fprintf(stderr, "Smoothing prolungator\n");
        }

        for (int j = 0; j < level - 1; j++) {
            CSR *oldP = hrrch->P_array[j];
            CSR *oldR = hrrch->R_array[j];
            CSR *oldA = hrrch->A_array[j + 1];

            CSR* SA = getSmoother(hrrch->A_array[j], hrrch->D_array[j]);
            sprintf(SA->name, "SA[%d]", j);
            #if 0
            {
                char fname[256] = {0};
                snprintf(fname, 256, "%s/%s%s.mtx", output_dir.c_str(), output_prefix.c_str(), SA->name);
                CSRm::printGlobalMM(SA, fname);
            }
            #endif

            #if CHECK_INDECES
            ASSERT(CSRm::checkUniqueIndeces(SA));
            #endif


            if (ISMASTER) {
                printf("level = %d\n", j);
                printf("Computing newP = SA * oldP\n");
            }
            CSR* newP = CSRm::product(h, SA, hrrch->P_array[j]);
            #if CHECK_INDECES
            ASSERT(CSRm::checkUniqueIndeces(newP));
            #endif
            sprintf(newP->name, "newP[%d]", j);
            #if 0
            {
                char fname[256] = {0};
                snprintf(fname, 256, "%s/%s%s.mtx", output_dir.c_str(), output_prefix.c_str(), newP->name);
                CSRm::printGlobalMM(newP, fname);
            }
            #endif

            ASSERT(newP->full_n > newP->m);
            // fprintf(stderr, "myid = %d, %s = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, newP->name, newP->n,  newP->full_n,  newP->m,  newP->row_shift,  newP->col_shifted);
            // CSRm::printUniqueColumns(newP, newP->name, stderr);
            // ASSERT(CSRm::checkProduct(h, SA, hrrch->P_array[j], newP));
            // if (ISMASTER) {
            //     CSR *Aj = hrrch->A_array[j];
            //     printf("A[%d]: n=%d, full_n=%d, m=%d\n", j, Aj->n, Aj->full_n, Aj->m);
            //     printf("SA: n=%d, full_n=%d, m=%d\n", SA->n, SA->full_n, SA->m);
            //     printf("P = SA * P: n=%d, full_n=%d, m=%d\n", newP->n, newP->full_n, newP->m);
            // }
            CSRm::free(SA);

            if (ISMASTER) {
                printf("Computing newR = newP^T\n");
            }
            // Fix Giacomo 2026-04-15: pass coarseA to CSRm::transpose so it uses the actual
            // SUITOR-based coarse row distribution (row_shift and n from coarseA, plus the
            // per-process row counts gathered via MPI_Allgather by ProcessSelector) instead
            // of the uniform estimate full_n/nprocs * myid. The uniform estimate can be
            // off-by-one when SUITOR produces an uneven distribution (e.g. 198+197 vs 197+198),
            // causing matrix items to be routed to the wrong process and leading to OOB accesses
            // or silent index corruption that makes the solver diverge.
            CSR* coarseA = hrrch->A_array[j+1];
            CSR* newR = CSRm::transpose(newP, log_file, coarseA);
            #if CHECK_INDECES
            ASSERT(CSRm::checkUniqueIndeces(newR));
            #endif
            sprintf(newR->name, "newR[%d]", j);

            ASSERT(newR->full_n < newR->m);

            if (ISMASTER) {
                printf("Computing newAP = oldA * newP\n");
            }
            // Fix Giacomo 2026-04-15: reset shrinked_col and bitcol cache for A_array[j]
            // before computing A * newP. The cached shrinked_col was built in a P=NULL
            // context (column range [0, A->n-1]) during the original hierarchy setup.
            // For A * newP the relevant column range is [0, newP->n-1], which is different
            // when the smoother changes the number of active columns. Reusing the stale
            // cache causes out-of-bounds accesses in NSP SpGEMM (shrinked_col values
            // exceed completedP->n). The fix is inert when CUSPARSE is used for SpGEMM.
            {
                CSR* Aj = hrrch->A_array[j];
                if (Aj->shrinked_col) {
                    CUDA_FREE(Aj->shrinked_col);
                    Aj->shrinked_col = NULL;
                }
                Aj->shrinked_flag = false;
                if (Aj->bitcol) {
                    CUDA_FREE(Aj->bitcol);
                    Aj->bitcol = NULL;
                    Aj->bitcolsize = 0;
                    Aj->nonuniquesize = 0;
                }
            }
            CSR* newAP = CSRm::product(h, hrrch->A_array[j], newP);
            #if CHECK_INDECES
            if (j == 3) {
                {
                    char fname[1024] = {0};
                    snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "probA", j, myid, nprocs);
                    CSRm::printMMsimple(hrrch->A_array[j], fname, false);
                    if (log_file) {
                        fprintf(log_file, "%s\n", fname);
                        CSRm::printInfo(hrrch->A_array[j], log_file);
                    }
                }
                {
                    char fname[1024] = {0};
                    snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "probP", j, myid, nprocs);
                    CSRm::printMMsimple(newP, fname, false);
                    if (log_file) {
                        fprintf(log_file, "%s\n", fname);
                        CSRm::printInfo(newP, log_file);
                    }
                }
            }
            ASSERT(CSRm::checkUniqueIndeces(hrrch->A_array[j]));
            ASSERT(CSRm::checkUniqueIndeces(newP));
            ASSERT(CSRm::checkUniqueIndeces(newAP));
            #endif
            // ASSERT(CSRm::checkProduct(h, hrrch->A_array[j], newP, newAP));
            // if (ISMASTER) {
            //     printf("R = P^T: n=%d, full_n=%d, m=%d\n", newR->n, newR->full_n, newR->m);
            //     printf("AP = A * P: n=%d, full_n=%d, m=%d\n", newAP->n, newAP->full_n, newAP->m);
            // }

            #if DEBUG
            {
                FILE *old_log_file = log_file;
                log_file = stderr;
                if (log_file && myid) {
                    fprintf(log_file, "\nlevel = %d\n", j);
                    fprintf(log_file, "Before adjusting shifts\n", j);
                    fprintf(log_file, "myid = %d, oldP = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, oldP->n,  oldP->full_n,  oldP->m,  oldP->row_shift,  oldP->col_shifted);
                    fprintf(log_file, "myid = %d, newP = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, newP->n,  newP->full_n,  newP->m,  newP->row_shift,  newP->col_shifted);
                    fprintf(log_file, "myid = %d, oldR = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, oldR->n,  oldR->full_n,  oldR->m,  oldR->row_shift,  oldR->col_shifted);
                    fprintf(log_file, "myid = %d, newR = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, newR->n,  newR->full_n,  newR->m,  newR->row_shift,  newR->col_shifted);
                }
                log_file = old_log_file;
            }
            #endif

            // Begin Fix (Giacomo 2025-11-11)
            // After CSRm::transpose above, newR has:
            //   row_shift  = coarseA->row_shift          (coarse rows owned by this process)
            //   col_shifted = -coarseA->row_shift        => col8 = global_fine - coarseA->row_shift
            //
            // The product (getmct + compute_rows_to_rcv_CPU) that evaluates
            // newR * newAP requires a different convention for newR's columns:
            //   col8 = global_fine - newAP->row_shift
            //     => getmct marks col8 in [0, newAP->n-1] as local (correct fine-level range)
            //   row_shift = newAP->row_shift
            //     => compute_rows_to_rcv_CPU: search_index = row_shift + col8 = global_fine
            //        (used to determine which process owns each fine-level column)
            //
            // The fix re-shifts newR's columns and updates row_shift to fine-level values.
            // After CSRm::product(newR, newAP) => newA, newA->row_shift will equal
            // newAP->row_shift (fine-level); the block below (Fix 3) corrects it to the
            // actual SUITOR coarse shift.
            if (newR->col_shifted) {
                // newR_col_shifted = newR->col_shifted;
                CSRm::shift_cols(newR, -newR->col_shifted);      // undo coarse shift => global fine cols
                CSRm::shift_cols(newR, -(long)newAP->row_shift); // apply fine shift => col8 = fine - fine_shift
                newR->row_shift = newAP->row_shift;
                newR->col_shifted = -(long)newAP->row_shift;
            }
            // End Fix (Giacomo 2025-11-11)

            if (ISMASTER) {
                printf("Computing newA = newR * newAP\n");
            }
            CSR* newA = CSRm::product(h, newR, newAP);
            #if CHECK_INDECES
            ASSERT(CSRm::checkUniqueIndeces(newA));
            #endif
            sprintf(newA->name, "newA[%d]", j+1);

            ASSERT(newA->full_n == newA->m);

            #if DEBUG
            {
                FILE *old_log_file = log_file;
                log_file = stderr;
                if (log_file && myid) {
                    fprintf(log_file, "myid = %d, oldA = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, oldA->n,  oldA->full_n,  oldA->m,  oldA->row_shift,  oldA->col_shifted);
                    fprintf(log_file, "myid = %d, newA = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, newA->n,  newA->full_n,  newA->m,  newA->row_shift,  newA->col_shifted);
                }
                log_file = old_log_file;
            }
            #endif

            // ASSERT(CSRm::checkProduct(h, newR, newAP, newA));
            CSRm::free(newAP);

            // if (ISMASTER) {
            //     printf("newA = R * AP: n=%d, full_n=%d, m=%d\n", newA->n, newA->full_n, newA->m);
            // }

            // Fix Giacomo 2026-04-15: replace the two adjustShift() calls with an explicit
            // shift using the actual coarse row_shift from the hierarchy.
            //
            // adjustShift() estimates the first coarse global index as:
            //   shift = (A->m / nprocs) * myid
            // This is the uniform-distribution estimate and can be off-by-one when SUITOR
            // produces an uneven coarse distribution (e.g. process 0 aggregates 198 coarse
            // nodes and process 1 aggregates 197, but uniform gives 197 to both except the
            // last). The off-by-one shifts col8 by +/-1 for every column in newA and newP,
            // corrupting the coarse-level correction and causing the solver to diverge.
            //
            // hrrch->A_array[j+1]->row_shift is computed by matchingPairAggregation via
            // MPI_Allgather of the actual local coarse counts, so it is always exact.
            {
                gstype S_actual = hrrch->A_array[j+1]->row_shift;
                // newA: square coarse-level matrix — shift both rows and cols to coarse origin
                if (newA->row_shift && !newA->col_shifted) {
                    CSRm::shift_cols(newA, -(long)S_actual);
                    newA->col_shifted = -(long)S_actual;
                    newA->row_shift = S_actual;
                }
                // newP: rectangular (fine rows, coarse cols) — only col_shifted changes;
                // row_shift stays at the fine-level value set by CSRm::product.
                if (newP->row_shift && !newP->col_shifted) {
                    CSRm::shift_cols(newP, -(long)S_actual);
                    newP->col_shifted = -(long)S_actual;
                }
            }

            #if DEBUG
            {
                FILE *old_log_file = log_file;
                log_file = stderr;
                if (log_file && myid) {
                    fprintf(log_file, "\nlevel = %d\n", j);
                    fprintf(log_file, "After adjusting shifts\n", j);
                    fprintf(log_file, "myid = %d, oldP = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, oldP->n,  oldP->full_n,  oldP->m,  oldP->row_shift,  oldP->col_shifted);
                    fprintf(log_file, "myid = %d, newP = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, newP->n,  newP->full_n,  newP->m,  newP->row_shift,  newP->col_shifted);
                    fprintf(log_file, "myid = %d, oldR = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, oldR->n,  oldR->full_n,  oldR->m,  oldR->row_shift,  oldR->col_shifted);
                    fprintf(log_file, "myid = %d, newR = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, newR->n,  newR->full_n,  newR->m,  newR->row_shift,  newR->col_shifted);
                    fprintf(log_file, "myid = %d, oldA = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, oldA->n,  oldA->full_n,  oldA->m,  oldA->row_shift,  oldA->col_shifted);
                    fprintf(log_file, "myid = %d, newA = n=%d, full_n=%d, m=%d, row_shift=%d, col_shifted=%d\n",  myid, newA->n,  newA->full_n,  newA->m,  newA->row_shift,  newA->col_shifted);
                }
                log_file = old_log_file;
            }
            #endif

            // ============================================================================
            // Substitution
            // ============================================================================

            // fprintf(stderr, "myid = %d, A[%d] = %p, P[%d] = %p, R[%d] = %p\n", myid, j, hrrch->A_array[j], j, newP, j, newR);
            ASSERT(hrrch->P_array[j]->full_n == newP->full_n);
            ASSERT(hrrch->P_array[j]->m      == newP->m);

            ASSERT(hrrch->R_array[j]->full_n == newR->full_n);
            ASSERT(hrrch->R_array[j]->m      == newR->m);

            ASSERT(hrrch->A_array[j+1]->full_n    == newA->full_n);
            ASSERT(hrrch->A_array[j+1]->m         == newA->m);
            // ASSERT(hrrch->A_array[j+1]->row_shift == newA->row_shift);

            CSRm::free(hrrch->P_array[j]);
            hrrch->P_array[j] = newP;
            
            CSRm::free(hrrch->R_array[j]);
            hrrch->R_array[j] = newR;

            CSRm::free(hrrch->A_array[j+1]);
            hrrch->A_array[j+1]=newA;

            #if 0
            {
                char fname[1024] = {0};
                snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "hrrch_A_new", j, myid, nprocs);
                CSRm::printMMsimple(hrrch->A_array[j], fname, false);
            }
            {
                char fname[1024] = {0};
                snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "hrrch_P_new", j, myid, nprocs);
                CSRm::printMMsimple(hrrch->P_array[j], fname, false);
            }
            {
                char fname[1024] = {0};
                snprintf(fname, 1024, "%s/%s%s%d_id%d_nprocs%d.mtx", output_dir.c_str(), output_prefix.c_str(), "hrrch_R_new", j, myid, nprocs);
                CSRm::printMMsimple(hrrch->R_array[j], fname, false);
            }
            #endif

            relaxPrepare(h, j+1, hrrch, amg_data);
        }

        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
        if (ISMASTER) {
            fprintf(stderr, "Smoothing prolungator [DONE 1/2]\n");
        }

        #if 0
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
        if (ISMASTER) {
            fprintf(stderr, "Recomputed hierarchy\n");
        }
        printHierarchyInfo(hrrch, level);
        #endif

        for (int j = 0; j < level; j++) {
            CSR *newP = hrrch->P_array[j];
            CSR *newR = hrrch->R_array[j];
            CSR *newA = hrrch->A_array[j];

            // if (newP) {
            //     CSRm::printUniqueColumns(newP, newP->name, stderr);
            // }

            // if (newR) {
            //     auto shift = hrrch->A_array[j + 1]->row_shift;
            //     // CSRm::shift_cols(newR, -(long)newR->col_shifted - (long)shift);
            //     // newR->col_shifted = -shift;
            //     newR->row_shift = shift;
            // }
            // if (newP) {
            //     auto shift = hrrch->A_array[j]->row_shift;
            //     // CSRm::shift_cols(newP, -(long)newP->col_shifted - (long)shift);
            //     // newP->col_shifted = -shift;
            //     newP->row_shift = shift;
            // }

            // ============================================================================
            // haloSetup
            // ============================================================================

            if (newP)
            {
                // trace_enabled = true;
                #if DEBUG
                if (log_file) {
                    fprintf(log_file, "Before haloSetup(P)\n");
                    fprintf(log_file, "================================================\n");
                }
                // fprintf(stderr, "Before haloSetup(P)\n");
                // fprintf(stderr, "================================================\n");
                // fflush(stderr);
                #endif
                // Fix Giacomo 2026-04-15: pass hrrch->A_array[j+1] as the second argument
                // so that getMissing uses nr_of_cols = R->n and mypfirstrow = R->row_shift
                // from the actual SUITOR coarse distribution, instead of the uniform estimate
                // newP->m / nprocs that the R=NULL path would use. This ensures the halo
                // correctly identifies which coarse columns are remote for each process.
                halo_info hi = haloSetup(newP, hrrch->A_array[j+1]);
                newP->halo = hi;
                #if DEBUG
                // fprintf(stderr, "================================================\n");
                // fprintf(stderr, "After haloSetup(P)\n");
                // fflush(stderr);
                if (log_file) {
                    fprintf(log_file, "================================================\n");
                    fprintf(log_file, "After haloSetup(P)\n");
                }
                #endif
                // trace_enabled = false;
            }

            if (newR)
            {
                // trace_enabled = true;
                #if DEBUG
                if (log_file) {
                    fprintf(log_file, "Before haloSetup(R)\n");
                    fprintf(log_file, "================================================\n");
                }
                // fprintf(stderr, "Before haloSetup(R)\n");
                // fprintf(stderr, "================================================\n");
                // fflush(stderr);
                #endif
                // Fix Giacomo 2026-04-15 (revised 2026-04-15): pass hrrch->A_array[j] as the
                // second argument so that getMissing uses nr_of_cols = R->n (exact local fine
                // row count) instead of A->m / nprocs (integer-truncated uniform estimate).
                // When total_fine is not divisible by nprocs the truncated estimate is too small
                // by 1 for some ranks, causing their last local col8 to escape the filter and be
                // treated as remote => the lookup maps it back to myid => NULL-deref crash.
                //
                // In the active implementation (USE_GET_MISSING_NEW=1), R != NULL still uses
                // A->col8 (newR->col8) for the filter - NOT R->col8 - so passing A_array[j]
                // here only fixes the column-count estimate without changing what is filtered.
                // (The old getMissing in the #else branch did use R->col8, which was the
                // original concern; that code path is no longer active.)
                halo_info hi = haloSetup(newR, hrrch->A_array[j]);
                newR->halo = hi;
                #if DEBUG
                // fprintf(stderr, "================================================\n");
                // fprintf(stderr, "After haloSetup(R)\n");
                // fflush(stderr);
                if (log_file) {
                    fprintf(log_file, "================================================\n");
                    fprintf(log_file, "After haloSetup(R)\n");
                }
                #endif
                // trace_enabled = false;
            }

            if (newA)
            {
                // trace_enabled = true;
                #if DEBUG
                if (log_file) {
                    fprintf(log_file, "Before haloSetup(A)\n");
                    fprintf(log_file, "================================================\n");
                }
                // fprintf(stderr, "Before haloSetup(A)\n");
                // fprintf(stderr, "================================================\n");
                // fflush(stderr);
                #endif
                halo_info hi = haloSetup(newA, NULL);
                newA->halo = hi;
                #if DEBUG
                // fprintf(stderr, "================================================\n");
                // fprintf(stderr, "After haloSetup(A)\n");
                // fflush(stderr);
                if (log_file) {
                    fprintf(log_file, "================================================\n");
                    fprintf(log_file, "After haloSetup(A)\n");
                }
                #endif
                // trace_enabled = false;
            }

            // ============================================================================
            // shrink_col
            // ============================================================================

            if (nprocs > 1) {
                #if DEBUG
                if (log_file) {
                    fprintf(log_file, "\nBefore shrink_col\n");
                    fprintf(log_file, "----------------------------------------------\n");
                }
                #endif

                // ASSERT(!newA || !newA->shrinked_flag);
                // ASSERT(!newP || !newP->shrinked_flag);

                if (newA) shrink_col(newA, NULL);
                // Fix Giacomo 2026-04-16: use the correct reference matrix so that
                // get_shrinked_col uses P->n (exact local count) not Alocal->n (wrong for
                // rectangular P/R). Mirrors the first-pass pattern at lines 240-242.
                if (newP) shrink_col(newP, hrrch->A_array[j + 1]);
                if (newR) shrink_col(newR, hrrch->A_array[j]);

                ASSERT(!newA || newA->shrinked_m <= newA->m);
                ASSERT(!newP || newP->shrinked_m <= newP->m);
                ASSERT(!newR || newR->shrinked_m <= newR->m);

                #if DEBUG
                if (log_file) {
                    fprintf(log_file, "\nAfter shrink_col\n");
                    fprintf(log_file, "----------------------------------------------\n");
                }
                #endif
            }

            if (newR) {
                auto shift = newP->row_shift;
                // CSRm::shift_cols(newR, -(long)newR->col_shifted - (long)shift);
                // newR->col_shifted = -shift;
                newR->row_shift = shift;
            }
            if (newP) {
                auto shift = hrrch->A_array[j]->row_shift;
                // CSRm::shift_cols(newP, -(long)newP->col_shifted - (long)shift);
                // newP->col_shifted = -shift;
                newP->row_shift = shift;
            }
        }

        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
        if (ISMASTER) {
            fprintf(stderr, "Smoothing prolungator [DONE 2/2]\n");
        }

        #if 0
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
        if (ISMASTER) {
            fprintf(stderr, "Recomputed hierarchy\n");
        }
        //printHierarchyInfo(hrrch, level);
        #endif
    }

    AMG::Hierarchy::finalize_level(hrrch, level);
    AMG::Hierarchy::finalize_cmplx(hrrch);
    AMG::Hierarchy::finalize_wcmplx(hrrch);
    hrrch->avg_cratio = level > 1
        ? (avcoarseratio / (level - 1))
        : 1;

    CUDA_FREE_HOST(iPtemp1);
    FREE(vPtemp1);
    CUDA_FREE_HOST(iAtemp1);
    CUDA_FREE_HOST(vAtemp1);
    CUDA_FREE(idevtemp1);
    CUDA_FREE(idevtemp2);
    CUDA_FREE(vdevtemp1);
    // --------------- AH glob -------------------
    CUDA_FREE(AH_glob_row);
    CUDA_FREE(AH_glob_col);
    CUDA_FREE(AH_glob_col8);
    CUDA_FREE(AH_glob_val);
    // -------------------------------------------
    if (alloced_idx == true) {
        CUDA_FREE(idx_4shrink);
        alloced_idx = false;
    }

    Vector::free(w_temp);
    Vector::free(amg_data->ws_buffer);
    Vector::free(amg_data->mutex_buffer);
    Vector::free(amg_data->_M);

    // END_PROF("MEM");

    END_PROF(__FUNCTION__);
    return hrrch;
}
