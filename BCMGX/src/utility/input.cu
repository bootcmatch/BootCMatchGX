#include "input.h"

#include "generator/laplacian.h"
#include "utility/distribuite.h"

generator_t get_generator(const char* str)
{
    if (!strcmp(str, "7p")) {
        return LAP_7P;
    } else if (!strcmp(str, "27p")) {
        return LAP_27P;
    } else {
        return INVALIG_GEN;
    }
}

CSR* read_local_matrix_from_mtx_host(const char* mtx_file)
{
    _MPI_ENV;

    CSR* Alocal_master = NULL;
    if (ISMASTER) {
        Alocal_master = read_matrix_from_file(mtx_file, 0, false);
        check_and_fix_order(Alocal_master);
    }

    taskmap = MALLOC(int, nprocs);
    itaskmap = MALLOC(int, nprocs);

    for (int i = 0; i < nprocs; i++) {
        taskmap[i] = i;
        itaskmap[i] = i;
    }

    CSR* Alocal = split_matrix_mpi_host(Alocal_master);
    if (ISMASTER) {
        CSRm::free(Alocal_master);
    }

    snprintf(idstring, sizeof(idstring), "1_1_1");
    CSRm::shift_cols_nogpu(Alocal, -Alocal->row_shift);
    Alocal->col_shifted = -Alocal->row_shift;

    return Alocal;
}

CSR* read_local_matrix_from_mtx(const char* mtx_file)
{
    _MPI_ENV;

    CSR* Alocal_master = NULL;
    if (ISMASTER) {
        Alocal_master = read_matrix_from_file(mtx_file, 0, false);
        check_and_fix_order(Alocal_master);
    }

    taskmap = MALLOC(int, nprocs);
    itaskmap = MALLOC(int, nprocs);

    for (int i = 0; i < nprocs; i++) {
        taskmap[i] = i;
        itaskmap[i] = i;
    }

    CSR* Alocal = split_matrix_mpi(Alocal_master);
    if (ISMASTER) {
        CSRm::free(Alocal_master);
    }

    snprintf(idstring, sizeof(idstring), "1_1_1");
    CSRm::shift_cols(Alocal, -Alocal->row_shift);
    Alocal->col_shifted = -Alocal->row_shift;

    return Alocal;
}

CSR* generate_lap_local_matrix_host(itype n)
{
    CSR* Alocal_host = generateLocalLaplacian3D(n);
    check_and_fix_order(Alocal_host);
    Alocal_host->col_shifted = -Alocal_host->row_shift;
    return Alocal_host;
}

CSR* generate_lap_local_matrix(itype n)
{
    CSR* Alocal_host = generate_lap_local_matrix_host(n);
    CSR* Alocal = CSRm::copyToDevice(Alocal_host);
    CSRm::free(Alocal_host);
    return Alocal;
}

CSR* generate_lap3d_local_matrix_host(generator_t generator, const char* lap_3d_file)
{
    _MPI_ENV;
    enum lap_params { nx = 0,
        ny = 1,
        nz = 2,
        P = 3,
        Q = 4,
        R = 5 };
    int* parms = read_laplacian_file(lap_3d_file);
    if (nprocs != (parms[P] * parms[Q] * parms[R])) {
        fprintf(stderr, "Nproc must be equal to P*Q*R\n");
        exit(EXIT_FAILURE);
    }
    CSR* Alocal_host = Alocal_host = NULL;
    switch (generator) {
    case LAP_7P:
        if (ISMASTER) {
            fprintf(stderr, "Using laplacian 3d 7 points generator.\n");
        }
        Alocal_host = generateLocalLaplacian3D_7p(parms[nx], parms[ny], parms[nz], parms[P], parms[Q], parms[R]);
        break;
    case LAP_27P:
        if (ISMASTER) {
            fprintf(stderr, "Using laplacian 3d 27 points generator.\n");
        }
        Alocal_host = generateLocalLaplacian3D_27p(parms[nx], parms[ny], parms[nz], parms[P], parms[Q], parms[R]);
        break;
    default:
        printf("Invalid generator\n");
        exit(1);
    }
    snprintf(idstring, sizeof(idstring), "%dx%dx%d", parms[P], parms[Q], parms[R]);
    FREE(parms);
    check_and_fix_order(Alocal_host);
    Alocal_host->col_shifted = -Alocal_host->row_shift;
    return Alocal_host;
}

CSR* generate_lap3d_local_matrix(generator_t generator, const char* lap_3d_file)
{
    CSR* Alocal_host = generate_lap3d_local_matrix_host(generator, lap_3d_file);
    CSR* Alocal = CSRm::copyToDevice(Alocal_host);
    CSRm::free(Alocal_host);
    return Alocal;
}
