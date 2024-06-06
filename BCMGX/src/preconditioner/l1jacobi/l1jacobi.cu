#include "basic_kernel/halo_communication/halo_communication.h"
#include "op/addAbsoluteRowSumNoDiag.h"
#include "op/basic.h"
#include "op/diagScal.h"
#include "op/mydiag.h"
#include "preconditioner/l1jacobi/l1jacobi.h"
#include "preconditioner/prec_setup.h"

/*----------------------------------------------------------------------------------
 * Given the local A returns the local vector of the diagonal of l1-jacobi smoother
 *---------------------------------------------------------------------------------*/
void set_l1j(CSR* Alocal, vector<vtype>* pl1j_loc)
{
    _MPI_ENV;

    // D^-1: pl1j_loc per ogni riga i di A contiene la somma dell'elemento
    // diagonale della riga i + i valori assoluti di tutti gli elementi extra-diagonali

    mydiag(Alocal, pl1j_loc);
    addAbsoluteRowSumNoDiag(Alocal, pl1j_loc);
}

void l1jacobi_setup(handles* h, CSR* Alocal, cgsprec* pr, const params& p)
{
    pr->l1jacobi.pl1j = Vector::init<vtype>(Alocal->n, true, true);
    pr->l1jacobi.w_loc = Vector::init<vtype>(Alocal->n, true, true);
    pr->l1jacobi.rcopy_loc = Vector::init<vtype>(Alocal->n, true, true);

    set_l1j(Alocal, pr->l1jacobi.pl1j);
}

void l1jacobi_finalize(handles* h, CSR* Alocal, cgsprec* pr, const params& p)
{
    Vector::free(pr->l1jacobi.pl1j);
    Vector::free(pr->l1jacobi.w_loc);
    Vector::free(pr->l1jacobi.rcopy_loc);
}

// r_loc -> residual
// u_loc -> solution
void l1jacobi_iter(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, vector<vtype>* pl1j, vector<vtype>* rcopy_loc, vector<vtype>* w_loc)
{
    _MPI_ENV;

    CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, u_loc, w_loc, 1., 0.);
    cudaDeviceSynchronize();

    Vector::copyTo(rcopy_loc, r_loc);
    my_axpby(w_loc->val, w_loc->n, rcopy_loc->val, -1., 1.); // rcopy_loc = rcopy_loc - w_loc = r_loc - A u

    // Usa pl1j per fare il diagonal scaling del residuo
    diagScal(rcopy_loc, pl1j, w_loc);
    my_axpby(w_loc->val, w_loc->n, u_loc->val, 1., 1.); // u_loc = u_loc + w_loc = u_loc + D^-1(r_loc - A*u_loc)
}

void l1jacobi_iter(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, cgsprec* pr, const params p)
{
    l1jacobi_iter(h, Alocal, r_loc, u_loc, pr->l1jacobi.pl1j, pr->l1jacobi.rcopy_loc, pr->l1jacobi.w_loc);
}

void l1jacobi_apply(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, cgsprec* pr, const params& p, PrecOut* out)
{
    for (int i = 0; i < p.l1jacsweeps; i++) {
        l1jacobi_iter(h, Alocal, r_loc, u_loc, pr, p);
    }
}
