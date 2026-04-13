#include "solver/cgsn/CGSN.h"

#include "halo_communication/halo_communication.h"
#include "op/LBfunctions.h"
#include "op/basic.h"
#include "preconditioner/prec_apply.h"
#include "solver/krylov_base/mpk.h"
#include "solver/krylov_base/mpkchebyshev.h"
#include "solver/krylov_base/mpk_mon_noprec.h"
#include "utility/handles.h"
#include "utility/profiling.h"

/**
 * @brief Performs an s-step version of the Conjugate Gradient method for solving linear systems as described in the paper:
 * P. D'Ambra, M. Bernaschi, M.G. Carrozzo, S. Thomas, Scalable s-step Preconditioned Conjugate Gradient with Chebyshev Basis and Gauss-Seidel Gram Solve, 
 * arXiv preprint arXiv:2603.09790, 2026.
 *
 * It performs a new algorithm wich relies on the iterative Gauss-Seidel method to solve the reduced Gram systems arising
 * from the projection of the original matrix on a s-dimensional Krylov based, in order to find the scalars/vectors for updating solution and directions
 * at each iteration. The Gram matrix P^TAP is obtained by using a preconditioned matrix power kernel, to obtain
 * the columns of P, i.e., the vectors of the s-step Krylov basis, and agglomerate all the needed scalar products in order
 * to reduce MPI global reductions.
 *
 * The function iterates until a convergence criterion is met (based on the residual norm) or the maximum number of
 * iterations is reached. The residuals are updated or recomputed periodically depending on the provided settings.
 *
 * @param h Handle to CUDA and other resources.
 * @param Alocal The local CSR matrix used in the matrix-vector products.
 * @param rhs_loc The local right-hand side vector.
 * @param x_loc The local initial guess vector.
 * @param p The parameters that control the behavior of the solver, including solver settings and limits.
 * @param pr Preconditioner storage.
 * @param out The output structure that will store the results, including residual history and final solution.
 *
 * @return The return code representing the status of the solver (0 for success, non-zero for failure).
 *
 * @note This function performs the iterative s-step CG method, which alternates between matrix-vector products and
 *       preconditioning, depending on the current iteration step and the solver configuration.
 *       The residuals are recalculated at specified intervals based on the provided parameters.
 *
 * @details The function follows the s-step approach, where the residual is computed and updated periodically,
 *          alternating between using the preconditioner and performing matrix-vector multiplications. The result
 *          is stored in the provided output structure `out`, and the solver iterates until convergence or the
 *          maximum number of iterations is reached.
 */

//void dumpMatColMajDH(int dh, double *m, int nr, int nc, int na, int iter, int pid, char *name, int stop);

void scale_Ab(vector<vtype>* Ab, vector<vtype>* d)
{
// Computes d (saved for later use) and scales Ab = (Gram | rhs)

    BEGIN_PROF(__FUNCTION__);

    int i, j, n;

    n = d->n;

    for (i = 0; i < n; i++) {
        if (Ab->val[i + n * i] != 0.0) {
            d->val[i] = 1.0 / Ab->val[i + n * i];
        } else {
            DIE("i=%d, Ab[i+n*i]= %lf, exiting.\n", i, Ab->val[i + n * i]);
        }
    }

	// Scale Ab
	for (i = 0; i < n; i++) {
		for (j = 0; j < n + 1; j++) {
			Ab->val[i + n * j] *= d->val[i];
		}
	}

    END_PROF(__FUNCTION__);
}


void scale_B(vectordh<vtype>* B, vector<vtype>* d)
{
// Scales B = multiple rhs (n,n) using d

    BEGIN_PROF(__FUNCTION__);

    int i, j, n;

    n = d->n;

	// Scale B (n,n)
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			B->val_[i + n * j] *= d->val[i];
		}
	}

    END_PROF(__FUNCTION__);
}


void solve_fgs_alpha(vector<vtype>* Ab, vectordh<vtype>* x, vtype gs_tol, int gs_iter)
{
// Ab = (Gram | rhs) (n, n+1) must be already scaled by d

    BEGIN_PROF(__FUNCTION__);

    int i, j, k, n;
    vtype s, so;

    n = x->n;

    vector<vtype>* ox = Vector::init<vtype>(n, true, false);

    for (i = 0; i < n; i++) {
        x->val_[i] = 0.0;
    }

    if (n == 1) {
        if (Ab->val[0] != 0.0) {
            x->val_[0] = Ab->val[1] / Ab->val[0];
        } else {
            DIE("s=1, Ab[0]= %lf, exiting.\n", Ab->val[0]);
        }
    } else {
        for (k = 0; k < gs_iter; k++) {
            for (i = 0; i < n; i++) {
                ox->val[i] = x->val_[i]; // ox = old x
            }
            for (i = 0; i < n; i++) {
                s = 0.0;
                so = 0.0;
                for (j = 0; j < i; j++) {
                    s += Ab->val[i + n * j] * x->val_[j];
                }
                for (j = i + 1; j < n; j++) {
                    so += Ab->val[i + n * j] * ox->val[j];
                }
                x->val_[i] = (Ab->val[n * n + i] - s - so) / Ab->val[i + n * i];
            }
            // check if norm(x-xold)/norm(x) < gs_tol
            s = 0.0;
            so = 0.0;
            for (i = 0; i < n; i++) {
                so += pow(x->val_[i] - ox->val[i], 2);
                s += pow(x->val_[i], 2);
            }
            if (sqrt(so / s) < gs_tol) {
                break;
            }
        }
    }

   	Vector::free(ox);

    END_PROF(__FUNCTION__);
}


void solve_fgs_beta(vector<vtype>* Ab, vectordh<vtype>* b, vtype gs_tol, int gs_iter, int n)
{
// Ab = (Gram | rhs) (n, n+1) and beta (n,n) must be already scaled by d
// Here will be used only the Gram part of Ab

    BEGIN_PROF(__FUNCTION__);

    int iter, all_below_threshold;
    vtype err_i;

    vector<vtype>* x = Vector::init<vtype>(n*n, true, false);
    vector<vtype>* ox = Vector::init<vtype>(n*n, true, false);
    vector<vtype>* s = Vector::init<vtype>(n, true, false);
    vector<vtype>* so = Vector::init<vtype>(n, true, false);

	Vector::fillWithValue(x,0.);

    if (n == 1) {
        if (Ab->val[0] != 0.0) {
            b->val_[0] = b->val_[0] / Ab->val[0];
        } else {
            DIE("s=1, Ab[0]= %lf, exiting.\n", Ab->val[0]);
        }
    } else {
        for (iter = 0; iter < gs_iter; iter++) {

        	memcpy(ox->val, x->val, n*n*sizeof(vtype)); // ox = old x

            for (int i = 0; i < n; i++) {

            	Vector::fillWithValue(s,0.);
            	Vector::fillWithValue(so,0.);

                for (int k=0; k<n; k++) {

					for (int j = 0; j < i; j++) {
						s->val[k] += Ab->val[i + n * j] * x->val[j + n*k];
					}

					for (int j = i + 1; j < n; j++) {
						so->val[k] += Ab->val[i + n * j] * ox->val[j + n*k];
					}

					x->val[i + n*k] = (b->val_[i + n*k] - s->val[k] - so->val[k]) / Ab->val[i + n * i];
               }

            }

            // Check if norm(x[i]-xold[i])/norm(x[i]) < gs_tol for all i
			Vector::fillWithValue(s,0.);
			Vector::fillWithValue(so,0.);
 			all_below_threshold = 0;
 			for (int i = 0; i < n; i++) {
				for (int k=0; k<n; k++) {
					so->val[i] += pow(x->val[i + n*k] - ox->val[i + n*k], 2);
					s->val[i] += pow(x->val[i + n*k], 2);
				}
				err_i = sqrt(so->val[i]/s->val[i]);
				if (err_i > gs_tol) break;
				all_below_threshold += 1;
            }
            if (all_below_threshold == n) {
            	break;
            }
        }
    	memcpy(b->val_, x->val, n*n*sizeof(vtype)); // so returns b = x
    }

   	Vector::free(x);
   	Vector::free(ox);
   	Vector::free(s);
   	Vector::free(so);

    END_PROF(__FUNCTION__);
}




int cgsstep_cgsn(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x_loc, const InputParameters& ip, const CurrentParameters& cp, Preconditioner* pr, SolverOut* out)
{

    _MPI_ENV;

    int iter; //, info;

    int excnlim = -1;
    int retval = -1;

    stype ln = Alocal->n;
    gstype fn = Alocal->full_n;

    int s = cp.sstep;

	// Chebyshev coefficients for Naumov
    vectordh<vtype>* abg = Vector::initdh<vtype>(3);
    if (ip.use_chebyshev) {
		abg->val_[0] = 2.0/ip.A_max_eigval;
		abg->val_[1] = 1.0;
		abg->val_[2] = 1.0;
		Vector::copydhToD<vtype>(abg);
		if (ISMASTER) {
			printf("Naumov\n alpha %.10lf\n beta %.10lf\n gamma %.10lf\n",abg->val_[0],abg->val_[1],abg->val_[2]);
		}
    }

    copyToCM(ip.A_max_eigval);

    out->resHist = MALLOC(vtype, ip.itnlim, true);

    vector<vtype>* u1_loc = Vector::init<vtype>(ln, true, true);
    vector<vtype>* r_loc = Vector::init<vtype>(ln, true, true);

    vector<vtype>* u_loc = NULL;
    if (pr->type != PreconditionerType::NONE) {
        u_loc = Vector::init<vtype>(ln, true, true);
    }

    Vector::copyTo(r_loc, rhs_loc); // r_loc = rhs_loc
    vector<vtype>* w_loc = CSRm::CSRVector_product_adaptive_miniwarp_witho(Alocal, x_loc, NULL, 1., 0.);

    my_axpby(w_loc->val, ln, r_loc->val, -1., 1.); // r_loc = r_loc - w_loc

    vector<vtype>* sP1 = Vector::init<vtype>(ln * 2 * s, true, true);
    vector<vtype>* sP2 = Vector::init<vtype>(ln * 2 * s, true, true);
    vectordh<vtype>* sGramRhs = Vector::initdh<vtype>(s * (s+1));
    vector<vtype>* GramRhs = Vector::init<vtype>(s * (s+1), true, false);
    vectordh<vtype>* alpha = Vector::initdh<vtype>(s);
    vectordh<vtype>* sb2 = Vector::initdh<vtype>(s * s);
    vectordh<vtype>* b2 = Vector::initdh<vtype>(s * s);

    vector<vtype>* zp = NULL;
    zp = Vector::init<vtype>(3*ln, true, true);

    vtype sone, sminusone, szero, delta0, l2_norm;
    sone = 1.0;
    sminusone = -1.0;
    szero = 0.0;
    delta0 = 1.0;

	itype* ipiv;
	ipiv = MALLOC(itype, s, true);

    vector<vtype>* d;
    if (ip.use_fgs) {
    	d = Vector::init<vtype>(s, true, false);
    }

    if (ip.stop_criterion == 1) {
        delta0 = Vector::norm_MPI(h->cublas_h, r_loc);
    }

    if (ISMASTER) {
        out->resHist[0] = delta0;
    }

    if (ip.use_chebyshev) {
        mpkc(h, Alocal, r_loc, s, sP2, pr, u1_loc, abg, zp, ip, cp, out);
    } else {
        if (pr->type == PreconditionerType::NONE) {
        	mpk_mon_noprec(h, Alocal, r_loc, s, sP2, pr, u1_loc, ip, cp, out);
        } else {
        	mpk(h, Alocal, r_loc, s, sP2, pr, u1_loc, ip, cp, out);
        }
    }

    for (iter = 0; iter < ip.itnlim; iter++) {

        BEGIN_PROF("cublasDgemm");
        cublasDgemm(h->cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, s, s, ln, &sone, sP2->val, ln, sP2->val+ln*s, ln, &szero, sGramRhs->val, s);
        END_PROF("cublasDgemm");
        BEGIN_PROF("cublasDgemv");
        cublasDgemv(h->cublas_h, CUBLAS_OP_T, ln, s, &sone, sP2->val, ln, r_loc->val, 1, &szero, sGramRhs->val+s*s, 1);
        END_PROF("cublasDgemv");

        cudaDeviceSynchronize();
        Vector::copydhToH<vtype>(sGramRhs);

        BEGIN_PROF("MPI_Allreduce");
        CHECK_MPI(MPI_Allreduce(sGramRhs->val_, GramRhs->val, s * (s+1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        END_PROF("MPI_Allreduce");

        // alpha
		memcpy(alpha->val_, GramRhs->val+s*s, s * sizeof(vtype));
        BEGIN_PROF("swork_CGSN");
        if (ip.use_fgs) {
        	scale_Ab(GramRhs, d);
        	solve_fgs_alpha(GramRhs, alpha, ip.gs_tol, ip.gs_iterations);
        } else {
        	/*info =*/ my_dgetrf(GramRhs->val, s, ipiv);
        	/*info =*/ my_dgetrs(GramRhs->val, alpha->val_, s, 1, ipiv);
        }
        END_PROF("swork_CGSN");

        Vector::copydhToD<vtype>(alpha);

        BEGIN_PROF("cublasDgemv-axpy");
        cublasDgemv(h->cublas_h, CUBLAS_OP_N, ln, s, &sone, sP2->val, ln, alpha->val, 1, &sone, x_loc->val, 1);
        cublasDgemv(h->cublas_h, CUBLAS_OP_N, ln, s, &sminusone, sP2->val + s * ln, ln, alpha->val, 1, &sone, r_loc->val, 1);
        END_PROF("cublasDgemv-axpy");

        if (ip.use_chebyshev) {
            mpkc(h, Alocal, r_loc, s, sP1, pr, u1_loc, abg, zp, ip, cp, out);
        } else {
            if (pr->type == PreconditionerType::NONE) {
            	mpk_mon_noprec(h, Alocal, r_loc, s, sP1, pr, u1_loc, ip, cp, out);
            } else {
            	mpk(h, Alocal, r_loc, s, sP1, pr, u1_loc, ip, cp, out);
            }
        }

        BEGIN_PROF("cublasDgemm");
        cublasDgemm(h->cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, s, s, ln, &sminusone, sP2->val, ln, sP1->val+ln*s, ln, &szero, sb2->val, s); //
        END_PROF("cublasDgemm");

        Vector::copydhToH<vtype>(sb2);

        BEGIN_PROF("MPI_Allreduce");
        CHECK_MPI(MPI_Allreduce(sb2->val_, b2->val_, s * s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        END_PROF("MPI_Allreduce");

        // beta
        BEGIN_PROF("swork_CGSN");
        if (ip.use_fgs) {
        	scale_B(b2, d);
        	solve_fgs_beta(GramRhs, b2, ip.gs_tol, ip.gs_iterations, s);
        } else {
        	// use previously (alpha) computed decomposition
        	/*info =*/ my_dgetrs(GramRhs->val, b2->val_, s, s, ipiv);
        }
        END_PROF("swork_CGSN");

        Vector::copydhToD<vtype>(b2);

        BEGIN_PROF("cublasDgemm");
        cublasDgemm(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, s, &sone, sP2->val, ln, b2->val, s, &szero, sP2->val, ln);
        END_PROF("cublasDgemm");
        BEGIN_PROF("cublasDgeam");
        cublasDgeam(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, &sone, sP2->val, ln, &sone, sP1->val, ln, sP2->val, ln); //
        END_PROF("cublasDgeam");

        BEGIN_PROF("cublasDgemm");
        cublasDgemm(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, s, &sone, sP2->val+s*ln, ln, b2->val, s, &szero, sP2->val+s*ln, ln);
        END_PROF("cublasDgemm");
        BEGIN_PROF("cublasDgeam");
        cublasDgeam(h->cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, ln, s, &sone, sP2->val+s*ln, ln, &sone, sP1->val+s*ln, ln, sP2->val+s*ln, ln);
        END_PROF("cublasDgeam");

        l2_norm = Vector::norm_MPI(h->cublas_h, r_loc);

        if (ISMASTER) {
            if (ip.dispnorm) {
                printf("%d) r norm: %.10lf\n", iter, l2_norm);
            }
            out->resHist[iter + 1] = l2_norm;
        }

        if (l2_norm < ip.rtol * delta0) {
            excnlim = 0;
            retval = 0;
            break;
        }

    }

    // Even if max_iter exceeded return the current status
    if (ISMASTER) {
        out->del0 = delta0;
        out->full_n = fn;
        out->local_n = ln;
        out->niter = iter + 1 + excnlim;
        out->exitRes = l2_norm;
    }

    Vector::free(r_loc);
    Vector::free(w_loc);
    Vector::free(sP1);
    Vector::free(sP2);
    Vector::free(u1_loc);
    Vector::free(zp);
    free(ipiv);
    if (pr->type != PreconditionerType::NONE) {
        Vector::free(u_loc);
    }

    if (ip.use_fgs) {
    	Vector::free(d);
    }

    return retval;
}

/**
 * @brief Solves the linear system \( A x = b \) using s-step Naumov method
 *
 * @param h Handle to CUDA and other resources.
 * @param Alocal The local CSR matrix used in the matrix-vector products.
 * @param rhs_loc The local right-hand side vector.
 * @param x_loc The local initial guess vector.
 * @param p The parameters that control the behavior of the solver, including solver settings and limits.
 * @param pr Preconditioner storage.
 * @param out The output structure that will store the results, including residual history and final solution.
 *
 * @return The solution vector after performing the CG s-step method, stored in `out->sol_local`.
 *
 * @note The final solution is returned in the output structure `out->sol_local`, and the result of the CG s-step
 *       method is captured in `out->retv`, which indicates the status of the solver.
 *
 * @details The function acts as a wrapper to invoke the `cgsstep` function, providing an interface for the user
 *          to solve the system. The initial guess `x0_loc` is updated in-place with the final solution, and the
 *          output structure contains additional information, such as the residual history and exit conditions.
 */

vector<vtype>* solve_cgsn(handles* h, CSR* Alocal, vector<vtype>* rhs_loc, vector<vtype>* x0_loc, Preconditioner* pr, const InputParameters& ip, const CurrentParameters& cp, SolverOut* out)
{

    out->retv = cgsstep_cgsn(h, Alocal, rhs_loc, x0_loc, ip, cp, pr, out);

    // To be removed, should be used the solution returned in x0. Instead, should be returned out->retv (or void then check out->retv in the calling function)
    out->sol_local = Vector::init<vtype>(x0_loc->n, true, true);
    Vector::copyTo(out->sol_local, x0_loc);

    return out->sol_local;
}
