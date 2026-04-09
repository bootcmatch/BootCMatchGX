#include "Anorm.h"
#include "utility/mpi.h"
#include "utility/profiling.h"

vtype Anorm(cublasHandle_t handle, CSR* A, vector<vtype>* v)
{
    BEGIN_PROF(__FUNCTION__);

    // Vector::debug(v, "\nv");

    TRACE("Before SpMV");
    cudaDeviceSynchronize();
    vector<vtype>* temp = CSRm::CSRVector_product_adaptive_miniwarp_witho(A, v, NULL, 1., 0.);
    cudaDeviceSynchronize();
    TRACE("After SpMV");

    // Vector::debug(temp, "\ntemp");
    // Vector::debug(v, "\nv");

    TRACE("Before dot");
    cudaDeviceSynchronize();
    double result_local = Vector::dot(handle, temp, v);
    cudaDeviceSynchronize();
    TRACE("After dot");

    double result = 0.;
    TRACE("Before MPI_Allreduce");
    BEGIN_PROF("MPI_Allreduce");
    CHECK_MPI(
        MPI_Allreduce(
            &result_local,
            &result,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD));
    END_PROF("MPI_Allreduce");
    TRACE("After MPI_Allreduce");

    result = sqrt(result);

    END_PROF(__FUNCTION__);

    return result;
}
