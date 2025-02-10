#include "computevm.h"

#include "op/basic.h"
#include "utility/profiling.h"

void computevm(vector<vtype>* sP, int s, vector<vtype>* r_loc, vectordh<vtype>* vm, bool use_prec)
{
    _MPI_ENV;
    BEGIN_PROF(__FUNCTION__);

    stype ln = r_loc->n;
    vectordh<vtype>* svm = Vector::initdh<vtype>(2 * s);

    if (use_prec == 0) {
        mynddot(s + 1, ln, sP->val, r_loc->val, svm->val); // vm(1:s+1)
        mynddot(s - 1, ln, sP->val + ln, sP->val + ln * s, svm->val + s + 1);
    } else {
        mynddot(s, ln, sP->val, r_loc->val, svm->val);
        mynddot(s, ln, sP->val, sP->val + ln * (2 * s - 1), svm->val + s); // vm(s+1:2s)
    }

    Vector::copydhToH<vtype>(svm);

    BEGIN_PROF("MPI_Allreduce");
    CHECK_MPI(MPI_Allreduce(svm->val_, vm->val_, 2 * s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    END_PROF("MPI_Allreduce");

    Vector::freedh(svm);

    END_PROF(__FUNCTION__);
}
