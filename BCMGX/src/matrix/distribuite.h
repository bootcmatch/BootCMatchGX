
#include "utility/myMPI.h"
#include <cuda_runtime.h>
#include "utility/cudamacro.h"

CSR* split_MatrixMPI(CSR *A);

CSR* join_MatrixMPI(CSR *Alocal);


