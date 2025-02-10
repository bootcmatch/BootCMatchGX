#include "scalarWorkMO.h"
#include "utility/profiling.h"

#ifdef SW_USE_LIB

#include "LBfunctions.h"

int scalarWorkMO(vectordh<vtype>* vm, vector<vtype>* W, vectordh<vtype>* alpha, vectordh<vtype>* beta, int s, int iter)
{
    BEGIN_PROF(__FUNCTION__);

    int i, j, info;

    vector<vtype>* Wcopy = NULL;
    Wcopy = Vector::init<vtype>(s * s, true, false);
    for (i = 0; i < s * s; i++) {
        Wcopy->val[i] = 0.0;
    }

    if (iter == 0) {

        for (i = 0; i < s; i++) {
            for (j = 0; j < s; j++) {
                W->val[i * s + j] = vm->val_[i + j + 1];
            }
        }
        memcpy(Wcopy->val, W->val, s * s * sizeof(vtype));
        memcpy(alpha->val_, vm->val_, s * sizeof(vtype));
        info = LBsolve(Wcopy->val, alpha->val_, s);

    } else {

        vector<vtype>* b1 = NULL;
        b1 = Vector::init<vtype>(s * s, true, false);
        for (i = 0; i < s * s; i++) {
            b1->val[i] = 0.0;
        }
        vector<vtype>* rhs1 = NULL;
        rhs1 = Vector::init<vtype>(2 * s - 1, true, false);
        for (i = 0; i < 2 * s - 1; i++) {
            rhs1->val[i] = 0.0;
        }

        for (i = 0; i < s; i++) {
            rhs1->val[s - 1 + i] = vm->val_[i];
            for (j = 0; j < i; j++) {
                rhs1->val[s - 1 + i] += rhs1->val[s - 1 + j] * alpha->val_[s - i + j - 1];
            }
            rhs1->val[s - 1 + i] = -rhs1->val[s - 1 + i] / alpha->val_[s - 1];
        }
        for (i = 0; i < s; i++) {
            for (j = 0; j < s; j++) {
                b1->val[i + j * s] = -rhs1->val[i + j];
                beta->val_[i + j * s] = -rhs1->val[i + j];
            }
        }
        info = LBsolvem(W->val, beta->val_, s);
        if (info != 0) {
            END_PROF(__FUNCTION__);
            return info;
        }

        for (i = 0; i < s; i++) {
            for (j = 0; j < s; j++) {
                W->val[i * s + j] = vm->val_[i + j + 1];
            }
        }

        // W = W - b1*beta
        LBdgemm(W->val, beta->val_, b1->val, s);

        // alpha = W\vm[1:s]
        memcpy(Wcopy->val, W->val, s * s * sizeof(vtype));
        memcpy(alpha->val_, vm->val_, s * sizeof(vtype));
        info = LBsolve(Wcopy->val, alpha->val_, s);

        Vector::free(b1);
        Vector::free(rhs1);
    }

    Vector::free(Wcopy);

    END_PROF(__FUNCTION__);
    return info;
}

#else

void getNewW(vectordh<vtype>* vm, vector<vtype>* W, vectordh<vtype>* beta, vector<vtype>* b1, int s)
{
    // W stored col maj

    int i, j, k;

    for (i = 0; i < s; i++) {
        for (j = 0; j < s; j++) {
            W->val[i + s * j] = vm->val_[i + j + 1];
            for (k = 0; k < s; k++) {
                W->val[i + s * j] -= b1->val[k + s * i] * beta->val_[k + s * j];
            }
        }
    }
}

int myChol(vtype* W, int n)
{
    // A positive definite matrix (n,n) stored col maj
    // L Cholesky decomposition (lower triangular) of A (stored col maj)

    int i, j, k;
    vtype sum, sq;

    int err = 0;

    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            sum = 0.0;
            for (k = 0; k < j; k++) {
                sum += W[i + k * n] * W[j + k * n];
            }
            W[i + j * n] = (W[i + j * n] - sum) / W[j + j * n];
        }
        sum = 0.0;
        for (k = 0; k < i; k++) {
            sum += W[i + k * n] * W[i + k * n];
        }
        sq = W[i + i * n] - sum;
        if (sq < 0.) {
            printf("W non positive definite %d\n", i);
            err = 1;
            break;
        } else {
            W[i + i * n] = sqrtf(sq);
        }
    }

    return err;
}

void mysolvem(vtype* L, vtype* rhs, int n, int nrhs)
{
    // L lower triangular matrix (n,n)
    // rhs (n, nrhs) col maj. Solution returned in rhs.

    int i, j, k;
    vtype sum;

    vector<vtype>* y = NULL;
    y = Vector::init<vtype>(n, true, false);

    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            sum = 0.0;
            for (k = 0; k < i; k++) {
                sum += L[i + n * k] * y->val[k];
            }
            y->val[i] = (rhs[i + nrhs * j] - sum) / L[i + n * i];
        }

        for (i = n - 1; i >= 0; i--) {
            sum = 0.0;
            for (k = i + 1; k < n; k++) {
                sum += L[k + n * i] * rhs[k + nrhs * j];
            }
            rhs[i + nrhs * j] = (y->val[i] - sum) / L[i + n * i];
        }
    }

    Vector::free(y);
}

int scalarWorkMO(vectordh<vtype>* vm, vector<vtype>* W, vectordh<vtype>* alpha, vectordh<vtype>* beta, int s, int iter)
{
    BEGIN_PROF(__FUNCTION__);

    int i, j, info;

    // printf("sw no lib\n");

    if (iter == 0) {

        for (i = 0; i < s; i++) {
            for (j = 0; j < s; j++) {
                W->val[i * s + j] = vm->val_[i + j + 1];
            }
        }

        info = myChol(W->val, s);
        if (info != 0) {
            return info;
        }

        memcpy(alpha->val_, vm->val_, s * sizeof(vtype));
        mysolvem(W->val, alpha->val_, s, 1);

    } else {

        vector<vtype>* b1 = NULL;
        b1 = Vector::init<vtype>(s * s, true, false);
        for (i = 0; i < s * s; i++) {
            b1->val[i] = 0.0;
        }
        vector<vtype>* rhs1 = NULL;
        rhs1 = Vector::init<vtype>(2 * s - 1, true, false);
        for (i = 0; i < 2 * s - 1; i++) {
            rhs1->val[i] = 0.0;
        }

        for (i = 0; i < s; i++) {
            rhs1->val[s - 1 + i] = vm->val_[i];
            for (j = 0; j < i; j++) {
                rhs1->val[s - 1 + i] += rhs1->val[s - 1 + j] * alpha->val_[s - i + j - 1];
            }
            rhs1->val[s - 1 + i] = -rhs1->val[s - 1 + i] / alpha->val_[s - 1];
        }
        for (i = 0; i < s; i++) {
            for (j = 0; j < s; j++) {
                b1->val[i + j * s] = -rhs1->val[i + j];
                beta->val_[i + j * s] = -rhs1->val[i + j];
            }
        }

        mysolvem(W->val, beta->val_, s, s);

        // W = W - b1*beta
        getNewW(vm, W, beta, b1, s);

        // alpha = W\vm[1:s]
        info = myChol(W->val, s);
        if (info != 0) {
            END_PROF(__FUNCTION__);
            return info;
        }
        memcpy(alpha->val_, vm->val_, s * sizeof(vtype));
        mysolvem(W->val, alpha->val_, s, 1);

        Vector::free(b1);
        Vector::free(rhs1);
    }

    END_PROF(__FUNCTION__);
    return info;
}

#endif
