#include "datastruct/CSR.h"
#include "datastruct/vector.h"
#include "preconditioner/PrecOut.h"
#include "preconditioner/prec_setup.h"

void prec_apply(handles* h, CSR* Alocal, vector<vtype>* r_loc, vector<vtype>* u_loc, cgsprec* pr, const params& p, PrecOut* out);
