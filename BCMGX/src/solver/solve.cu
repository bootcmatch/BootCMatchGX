#include "solver/solve.h"

#include "preconditioner/prec_setup.h"
#include "solver/cghs/CG_HS.h"
#include "solver/cgs/CGs.h"
#include "solver/cgs_cublas/CGscublas.h"
#include "solver/fcg/FCG.h"
#include "solver/pipelined_cgs/pipelinedCGs.h"
#include "utility/profiling.h"

vector<vtype>* solve(handles* h, CSR* Alocal, vector<vtype>* rhs, vector<vtype>* x0, const params& p, cgsprec& pr, SolverOut* out)
{
    _MPI_ENV;

    vector<vtype>* sol = NULL;

    beginProfiling(__FILE__, __FUNCTION__, __FUNCTION__);

    switch (p.solver_type) {
    case SolverType::CGHS:
        sol = solve_cg_hs(h, Alocal, rhs, x0, &pr, p, out);
        break;

    case SolverType::FCG:
        sol = solve_fcg(h, Alocal, rhs, x0, &pr, p, out);
        break;

    case SolverType::CGS:
        sol = solve_cgs(h, Alocal, rhs, x0, &pr, p, out);
        break;

    case SolverType::PIPELINED_CGS:
        sol = solve_pipelined_cgs(h, Alocal, rhs, x0, &pr, p, out);
        break;

    case SolverType::CGS_CUBLAS:
        sol = solve_cgs_cublas(h, Alocal, rhs, x0, &pr, p, out);
        break;

    default:
        printf("Unhandled value for enum class SolverType\n");
        exit(1);
    }

    cudaDeviceSynchronize();
    endProfiling(__FILE__, __FUNCTION__, __FUNCTION__);

    // -------------------------------------------------------------------------

    return sol;
}
