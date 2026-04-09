#include "mpkchebyshev.h"

#include "halo_communication/halo_communication.h"
#include "op/CSRVector_product_adaptive_miniwarp_splitted.h"
#include "preconditioner/prec_apply.h"
#include "utility/profiling.h"

#define USECM 1

__constant__ vtype abg_const[3];

void copyToCM(vtype maxval)
{

    vtype abgv[3];
    abgv[0] = 2.0 / maxval;
    abgv[1] = 1.0;
    abgv[2] = 1.0;

    cudaMemcpyToSymbol(abg_const, &abgv, 3 * sizeof(vtype));
}

__global__ void lcomb2cm(itype n, vtype* zp, vtype* p, vtype* zp1)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        zp[id] = abg_const[0] * p[id] - abg_const[1] * zp1[id];
    }
}

__global__ void lcomb3cm(itype n, vtype* zp, vtype* p, vtype* zp1, vtype* zp2)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        zp[id] = 2 * abg_const[0] * p[id] - 2 * abg_const[1] * zp1[id] - abg_const[2] * zp2[id];
    }
}

void mylcomb3cm(int n, double* zp, double* y, double* zp1, double* zp2)
{
    BEGIN_PROF(__FUNCTION__);
    GridBlock gb = gb1d(n, BLOCKSIZE);
    lcomb3cm<<<gb.g, gb.b>>>(n, zp, y, zp1, zp2);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

void mylcomb2cm(int n, double* zp, double* y, double* zp1)
{
    BEGIN_PROF(__FUNCTION__);
    GridBlock gb = gb1d(n, BLOCKSIZE);
    lcomb2cm<<<gb.g, gb.b>>>(n, zp, y, zp1);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

__global__ void lcomb2(itype n, vtype* zp, vtype* p, vtype* zp1, vtype a, vtype b)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        zp[id] = a * p[id] - b * zp1[id];
    }
}

__global__ void lcomb3(itype n, vtype* zp, vtype* p, vtype* zp1, vtype* zp2, vtype a, vtype b, vtype g)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        zp[id] = 2 * a * p[id] - 2 * b * zp1[id] - g * zp2[id];
    }
}

void mylcomb3(int n, double* zp, double* y, double* zp1, double* zp2, double a, double b, double g)
{
    BEGIN_PROF(__FUNCTION__);
    GridBlock gb = gb1d(n, BLOCKSIZE);
    lcomb3<<<gb.g, gb.b>>>(n, zp, y, zp1, zp2, a, b, g);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

void mylcomb2(int n, double* zp, double* y, double* zp1, double a, double b)
{
    BEGIN_PROF(__FUNCTION__);
    GridBlock gb = gb1d(n, BLOCKSIZE);
    lcomb2<<<gb.g, gb.b>>>(n, zp, y, zp1, a, b);
    cudaDeviceSynchronize();
    END_PROF(__FUNCTION__);
}

// Chebyshev mpk
void mpkc(handles* h, CSR* Alocal, vector<vtype>* x_loc, int s, vector<vtype>* sP, Preconditioner* pr, vector<vtype>* y_loc, vectordh<vtype>* abg, vector<vtype>* zp, const InputParameters& ip, const CurrentParameters& cp, SolverOut* out)
{
    _MPI_ENV;
    BEGIN_PROF(__FUNCTION__);

    int i, p_0, p_minus, p_plus, ptemp;
    int pToP, pToZ;
    stype n = Alocal->n;

    vtype* xtemp = x_loc->val;
    vtype* ytemp = y_loc->val;

    pToZ = 0;
    pToP = s;

    // vtype a = abg->val_[0];
    // vtype b = abg->val_[1];
    // vtype g = abg->val_[2];

    for (i = 0; i < s; i++) {

        if (i == 0) {
            Vector::copyTo(zp, x_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);

            y_loc->val = sP->val;
            Vector::fillWithValue(y_loc, 0.);
            cudaDeviceSynchronize();

        	//prec_apply(h, Alocal, x_loc, y_loc, pr, ip);
            if (pr->type == PreconditionerType::NONE) {
                Vector::copyTo(y_loc, x_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);
            } else {
            	prec_apply(h, Alocal, x_loc, y_loc, pr, ip);
            }

        } else if (i == 1) {
            x_loc->val = y_loc->val;
            y_loc->val = sP->val + pToP * n;
            CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, y_loc, 1., 0.);

#if USECM
            mylcomb2cm(n, zp->val + n, y_loc->val, zp->val);
#else
            mylcomb2(n, zp->val + n, y_loc->val, zp->val, a, b);
#endif

            x_loc->val = zp->val + n;
            y_loc->val = sP->val + ++pToZ * n;
            Vector::fillWithValue(y_loc, 0.);
            cudaDeviceSynchronize();

            //prec_apply(h, Alocal, x_loc, y_loc, pr, ip);
            if (pr->type == PreconditionerType::NONE) {
                Vector::copyTo(y_loc, x_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);
            } else {
            	prec_apply(h, Alocal, x_loc, y_loc, pr, ip);
            }

            p_0 = 0;
            p_plus = 1;
            p_minus = 2;

        } else {
            ptemp = p_plus;
            p_plus = p_minus;
            p_minus = p_0;
            p_0 = ptemp;

            x_loc->val = y_loc->val;
            y_loc->val = sP->val + ++pToP * n;
            CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, y_loc, 1., 0.);

#if USECM
            mylcomb3cm(n, zp->val + p_plus * n, y_loc->val, zp->val + p_0 * n, zp->val + p_minus * n);
#else
            mylcomb3(n, zp->val + p_plus * n, y_loc->val, zp->val + p_0 * n, zp->val + p_minus * n, a, b, g);
#endif

            x_loc->val = zp->val + p_plus * n;
            y_loc->val = sP->val + ++pToZ * n;
            Vector::fillWithValue(y_loc, 0.);
            cudaDeviceSynchronize();

            //prec_apply(h, Alocal, x_loc, y_loc, pr, ip);
            if (pr->type == PreconditionerType::NONE) {
                Vector::copyTo(y_loc, x_loc, (nprocs > 1) ? *(Alocal->os.streams->comm_stream) : 0);
            } else {
            	prec_apply(h, Alocal, x_loc, y_loc, pr, ip);
            }
        }
    }

    x_loc->val = y_loc->val;
    y_loc->val = sP->val + ++pToP * n;
    if (s == 1) {
        y_loc->val = sP->val + n;
    }
    CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, y_loc, 1., 0.);

    x_loc->val = xtemp;
    y_loc->val = ytemp;

    END_PROF(__FUNCTION__);
}
