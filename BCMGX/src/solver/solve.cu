#include "solver/solve.h"

#include "basic_kernel/halo_communication/halo_communication.h"
#include "basic_kernel/halo_communication/newoverlap.h"
#include "custom_cudamalloc/custom_cudamalloc.h"
#include "preconditioner/prec_apply.h"
#include "preconditioner/prec_finalize.h"
#include "preconditioner/prec_setup.h"
#include "solver/cghs/CG_HS.h"
#include "solver/fcg/FCG.h"
#include "utility/metrics.h"
#include "utility/timing.h"

vector<vtype>* solve(CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x0, const params& p, SolverOut* out)
{
    _MPI_ENV;

    vector<vtype>* sol;

    // -------------------------------------------------------------------------

    handles* h = Handles::init();

    // ------------------ custom_cudamalloc --------------------
    CustomCudaMalloc::init((Alocal->nnz) * 5, (Alocal->nnz) * 3);
    CustomCudaMalloc::init((Alocal->nnz) * 1, (Alocal->nnz) * 1, 1);
    CustomCudaMalloc::init((Alocal->nnz) * 3, (Alocal->nnz) * 3, 2);
    // ---------------------------------------------------------

    if (p.sprec != PreconditionerType::BCMG) {

        halo_info hi = haloSetup(Alocal, NULL);

        Alocal->halo = hi;

        shrink_col(Alocal, NULL);
    }

    cgsprec pr;
    pr.ptype = p.sprec;

    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Setting preconditioner %d\n", pr.ptype);
        TIME::start();
    }
    prec_setup(h, Alocal, &pr, p);
    if (ISMASTER) {
        float precsetup_time = TIME::stop();
        fprintf(stdout, "Preconditioner setup time: %f in seconds\n", precsetup_time / 1000.);
    }

    // -------------------------------------------------------------------------

    switch (p.solver_type) {
    case SolverType::CGHS:
        if (ISMASTER) {
            printf("Solving using CG-HS...\n");
        }
        sol = solve_cg_hs(h, Alocal, rhs, x0, &pr, p, out);
        break;

    case SolverType::FCG:
        if (ISMASTER) {
            printf("Solving using FCG...\n");
        }
        sol = solve_fcg(h, Alocal, rhs, x0, &pr, p, out);
        break;

    default:
        printf("Unhandled value for enum class SolverType\n");
        exit(1);
    }

    // -------------------------------------------------------------------------

    if (ISMASTER) {
        printf("Finalizing preconditioner %d\n", pr.ptype);
        TIME::start();
    }
    prec_finalize(h, Alocal, &pr, p);
    if (ISMASTER) {
        float prec_finalize_time = TIME::stop();
        fprintf(stdout, "Preconditioner finalize time: %f in seconds\n", prec_finalize_time / 1000.);
    }

    // ------------------ custom_cudamalloc --------------------
    CustomCudaMalloc::free(); // Freed in sample_main.cu
    // ---------------------------------------------------------

    return sol;
}
