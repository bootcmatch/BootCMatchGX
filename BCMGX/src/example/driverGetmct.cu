#include "datastruct/vector.h"
#include "op/basic.h"
#include "op/getmct.h"
#include "utility/arrays.h"

#define USAGE "Usage:\n\t%s <FILE_NAME>\n"

// long* getmct(gsstype* Col, int nnz, int f, int l, int* uvs, long** bitcol, int* bitcolsize, int num_thr);

#define NPROCS 255

/**
 * Compute norm(sol1 - sol2) / norm(sol1) and
 * eventually verify it is under a given threshold
 */
int main(int argc, char** argv)
{
    if (argc != 2) {
        DIE(USAGE, argv[0]);
    }

    // handles* h = Handles::init();

    vector<long>* arr = Vector::load<long>(argv[1], true);

 //   debugArray("arr[%d] = %ld\n", arr->val, arr->n, true, stderr);

    gstype ends[NPROCS];
    gstype mypfirstrow;
#include "255.h"
    int uvs = 0;
    long* bitcol = nullptr;
    int bitcolsize = 0;
    int nonuniquesize = 0;
    int nrow=360*360*360;
    long* mct = getmct(arr->val, arr->n, 0, nrow, &uvs, &bitcol, &bitcolsize, &nonuniquesize, NUM_THR);

    vector<gsstype>* _bitcol;
    if (uvs == 0) {
        _bitcol = Vector::init<gsstype>(1, true, false);
    } else {
        _bitcol = Vector::init<gsstype>(uvs, false, false);
        _bitcol->val = mct;
    }

    stype J = 0;
    for (itype i = 0; i < uvs; i++) {
	    gsstype j = _bitcol->val[i];
  	    fprintf(stderr,"trying %ld (%d/%d) (J=%d)\n",j,i,uvs,J);
        CHECK_AGAIN:
            if ((j + mypfirstrow) >= ends[J]) {
                J++;
                goto CHECK_AGAIN;
            }

            if (J >= NPROCS) {
                fprintf(stderr, "unexpected missing source :%d for %ld (%lu)\n",J,j,mypfirstrow);
	    }
// 	    printf("%ld\n",j);
    }


/*    debugArray("mct[%d] = %ld\n", mct, uvs, false, stderr); */

    FREE(mct);
    Vector::free(arr);

    return 0;
}
