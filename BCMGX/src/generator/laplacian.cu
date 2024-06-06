#include "generator/laplacian.h"

#include "datastruct/CSR.h"
#include "utility/cudamacro.h"
#include "utility/globals.h"
#include "utility/mpi.h"
#include "utility/setting.h"
#include "utility/utils.h"

#define MAX_NNZ_PER_ROW_LAP 5
#define LAP_N_PARAMS 6

long int internal_index(int gi, int gj, int gk, int nx, int ny, int nz, int P, int Q, int R)
{
    gstype i = gi % nx; // Position in x
    gstype j = gj % ny; // Position in y
    gstype k = gk % nz; // Position in z

    gstype p = gi / nx; // Position in x direction
    gstype q = gj / ny; // Position in y
    gstype r = gk / nz; // Position in z

    return (r * ((gstype)P) * ((gstype)Q) + q * ((gstype)P) + p) * ((gstype)nx) * ((gstype)ny) * ((gstype)nz) + (k * ((gstype)nx) * ((gstype)ny) + j * ((gstype)nx) + i);
}

CSR* generateLocalLaplacian3D(itype n)
{
    _MPI_ENV;
    // global number of rows
    gstype N = ((gstype)n) * ((gstype)n) * ((gstype)n);
    /* Each processor knows only of its own rows - the range is denoted by ilower
       and upper.  Here we partition the rows. We account for the fact that
       N may not divide evenly by the number of processors. */
    stype local_size = N / ((gstype)nprocs);
    // local row start
    gstype ilower = ((gstype)local_size) * ((gstype)myid);
    // local row end
    gstype iupper = ((gstype)local_size) * ((gstype)(myid + 1));
    // last takes all

    if (myid == nprocs - 1) {
        iupper = N;
        local_size = N - ((gstype)local_size) * ((gstype)myid);
    }

    assert(local_size == (iupper - ilower));
    CSR* Alocal = CSRm::init(local_size, N, local_size * 7, true, false, false, N, ilower);

    vtype values[7];
    itype cols[7];
    itype NNZ = 0;
    gstype I = 0;

    for (gsstype i = ilower; i < iupper; i++, I++) {
        itype nnz = 0;
        gstype k = floor(i / (((gstype)n) * ((gstype)n)));
        /* The left identity block:position i-n*n */
        if ((i - n * n) >= 0) {
            cols[nnz] = (i - ((gstype)n) * ((gstype)n)) - ilower;
            values[nnz] = -1.0;
            nnz++;
        }
        /* The left identity block:position i-n */
        if (i >= n + k * ((gstype)n) * ((gstype)n) && i < (k + 1) * ((gstype)n) * ((gstype)n)) {
            cols[nnz] = i - n - ilower;
            values[nnz] = -1.0;
            nnz++;
        }
        /* The left -1: position i-1 */
        if (i % n) {
            cols[nnz] = i - 1 - ilower;
            values[nnz] = -1.0;
            nnz++;
        }
        /* Set the diagonal: position i */
        cols[nnz] = i - ilower;
        values[nnz] = 6.0;
        nnz++;
        /* The right -1: position i+1 */
        if ((i + 1) % n) {
            cols[nnz] = i + 1 - ilower;
            values[nnz] = -1.0;
            nnz++;
        }
        /* The right identity block:position i+n */
        if (i >= k * ((gstype)n) * ((gstype)n) && i < (k + 1) * (((gstype)n) * ((gstype)n)) - (gstype)n) {
            cols[nnz] = i + (gstype)n - ilower;
            values[nnz] = -1.0;
            nnz++;
        }
        /* The right identity block:position i+n*n */
        if ((i + ((gstype)n) * ((gstype)n)) < N) {
            cols[nnz] = i + ((gstype)n) * ((gstype)n) - ilower;
            values[nnz] = -1.0;
            nnz++;
        }
        if (I == 0) {
            Alocal->row[0] = 0;
        }
        // set row index
        Alocal->row[I + 1] = Alocal->row[I] + nnz;
        for (itype j = 0; j < nnz; j++) {
            Alocal->col[NNZ] = cols[j];
            Alocal->val[NNZ] = values[j];
            NNZ++;
        }
    }
    taskmap = (int*)Malloc(nprocs * sizeof(*taskmap));
    itaskmap = (int*)Malloc(nprocs * sizeof(*itaskmap));
    if (taskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for taskmap\n", nprocs * sizeof(*taskmap));
        exit(1);
    }
    if (itaskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for itaskmap\n", nprocs * sizeof(*itaskmap));
        exit(1);
    }
    for (int i = 0; i < nprocs; i++) {
        taskmap[i] = i;
        itaskmap[i] = i;
    }
    Alocal->nnz = NNZ;
    return Alocal;
}

CSR* generateLocalLaplacian3D_7p(itype nx, itype ny, itype nz, itype P, itype Q, itype R)
{
    PUSH_RANGE(__func__, 2)

    itype local_size = nx * ny * nz;

    gstype gnx = (gstype)nx * (gstype)P; // Boundary on x side
    gstype gny = (gstype)ny * (gstype)Q; // Boundary on y side
    gstype gnz = (gstype)nz * (gstype)R; // Boundary on z side

    gstype num_rows = gnx * gny * gnz; // global number of rows
    gstype num_nonzeros = num_rows * 7; // Ignoring any boundary, 7 nnz per row
    gstype num_substract = 0;

    num_substract += gny * gnz;
    num_substract += gny * gnz;
    num_substract += gnx * gnz;
    num_substract += gnx * gnz;
    num_substract += gnx * gny;
    num_substract += gnx * gny;

    num_nonzeros -= num_substract; // global

    // ---------------------------------------------------------------------------

    _MPI_ENV;
    MPI_Comm NEWCOMM;

    int dims[3] = { 0, 0, 0 };
    int periods[3] = { false, false, false };
    int coords[3] = { 0, 0, 0 };
    int my3id;

    dims[0] = P;
    dims[1] = Q;
    dims[2] = R;

    MPI_Dims_create(nprocs, 3, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, false, &NEWCOMM);
    MPI_Comm_rank(NEWCOMM, &my3id);
    MPI_Cart_coords(NEWCOMM, my3id, 3, coords);

    // ---------------------------------------------------------------------------

    int p = coords[0];
    int q = coords[1];
    int r = coords[2];

    gstype ilower = (((gstype)r) * ((gstype)Q) * ((gstype)P)
                        + ((gstype)q) * ((gstype)P)
                        + ((gstype)p))
        * ((gstype)local_size);

    // ---------------------------------------------------------------------------

    int allcoords[3 * P * Q * R] = { 0 };

    taskmap = (int*)Malloc(nprocs * sizeof(*taskmap));
    if (taskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for taskmap\n", nprocs * sizeof(*taskmap));
        exit(1);
    }

    itaskmap = (int*)Malloc(nprocs * sizeof(*itaskmap));
    if (itaskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for itaskmap\n", nprocs * sizeof(*itaskmap));
        exit(1);
    }

    CHECK_MPI(MPI_Allgather(
        coords,
        sizeof(coords),
        MPI_BYTE,
        allcoords,
        sizeof(coords),
        MPI_BYTE,
        MPI_COMM_WORLD));

    for (int i = 0; i < nprocs; i++) {
        taskmap[(allcoords[i * 3 + 2] * Q * P) + (allcoords[i * 3 + 1] * P) + (allcoords[i * 3])] = i;
        itaskmap[i] = (allcoords[i * 3 + 2] * Q * P) + (allcoords[i * 3 + 1] * P) + (allcoords[i * 3]);
    }

    // ---------------------------------------------------------------------------

    CSR* Alocal = CSRm::init(
        local_size,
        num_rows,
        (local_size * 7),
        true,
        false,
        false,
        num_rows,
        ilower);

    // alloc COO
    itype* Arow = (itype*)Malloc(sizeof(itype*) * (local_size * 7));
    itype* Acol = (itype*)Malloc(sizeof(itype*) * (local_size * 7));
    vtype* Aval = (vtype*)Malloc(sizeof(vtype*) * (local_size * 7));

    itype count = 0;
    itype nz_count = 0;
    itype nnz = 0;
    Alocal->row[0] = 0;

    // ---------------------------------------------------------------------------

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int gi = p * nx + i;
                int gj = q * ny + j;
                int gk = r * nz + k;

                // Diagonal term
                Arow[nz_count] = count;
                Acol[nz_count] = count + (ilower - ilower);
                Aval[nz_count] = 6.;
                nnz++;
                nz_count++;

                // Given gi, gj, gk, find p, q, r, i_loc, j_loc, k_loc

                if (i == 0 && p == 0) {
                    // do nothing
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi - 1, gj, gk, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;
                }

                if (i == nx - 1 && p == P - 1) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi + 1, gj, gk, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;
                }

                if ((j == 0) && (q == 0)) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi, gj - 1, gk, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;
                }

                if (j == ny - 1 && q == Q - 1) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi, gj + 1, gk, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;
                }

                if (k == 0 && r == 0) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi, gj, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;
                }

                if ((k == nz - 1) && (r == R - 1)) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi, gj, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;
                }

                Alocal->row[count + 1] = Alocal->row[count] + nnz;
                bubbleSort(&Acol[nz_count - nnz], &Aval[nz_count - nnz], nnz);
                nnz = 0;
                count++;
            }
        }
    }

    // ---------------------------------------------------------------------------

    Alocal->row[count] = nz_count; // check if
    for (itype j = 0; j < nz_count; j++) {
        Alocal->col[j] = Acol[j];
        Alocal->val[j] = Aval[j];
    }
    Alocal->nnz = nz_count;

#if defined(PRINT_COO3D)
    FILE* fout = NULL;
    char fname[256];
    snprintf(fname, 256, "matrix-rank%d-pqr-%d-%d-%d.mtx", myid, p, q, r);
    fout = fopen(fname, "w+");
    if (fout == NULL) {
        fprintf(stderr, "in function %s: error opening %s\n", __func__, fname);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nz_count; i++) {
        fprintf(fout, "%d %d %lf\n", Arow[i] + (ilower + 1), Acol[i] + ilower + 1, Aval[i]);
    }
    fclose(fout);
#endif

    Free(Arow);
    Free(Acol);
    Free(Aval);

    POP_RANGE
    return Alocal;
}

CSR* generateLocalLaplacian3D_27p(itype nx, itype ny, itype nz, itype P, itype Q, itype R)
{
    PUSH_RANGE(__func__, 2)

    itype local_size = nx * ny * nz;

    gstype gnx = (gstype)nx * (gstype)P; // Boundary on x side
    gstype gny = (gstype)ny * (gstype)Q; // Boundary on y side
    gstype gnz = (gstype)nz * (gstype)R; // Boundary on z side

    gstype num_rows = gnx * gny * gnz; // global number of rows
    gstype num_nonzeros = num_rows * 27; // Ignoring any boundary, 7 nnz per row
    gstype num_substract = 0;

    num_substract += gny * gnz;
    num_substract += gny * gnz;
    num_substract += gnx * gnz;
    num_substract += gnx * gnz;
    num_substract += gnx * gny;
    num_substract += gnx * gny;

    num_nonzeros -= num_substract; // global

    // ---------------------------------------------------------------------------

    _MPI_ENV;
    MPI_Comm NEWCOMM;

    int dims[3] = { 0, 0, 0 };
    int periods[3] = { false, false, false };
    int coords[3] = { 0, 0, 0 };
    int my3id;

    dims[0] = P;
    dims[1] = Q;
    dims[2] = R;

    MPI_Dims_create(nprocs, 3, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, false, &NEWCOMM);
    MPI_Comm_rank(NEWCOMM, &my3id);
    MPI_Cart_coords(NEWCOMM, my3id, 3, coords);

    // ---------------------------------------------------------------------------

    int p = coords[0];
    int q = coords[1];
    int r = coords[2];

    gstype ilower = (((gstype)r) * ((gstype)Q) * ((gstype)P)
                        + ((gstype)q) * ((gstype)P)
                        + ((gstype)p))
        * ((gstype)local_size);

    // ---------------------------------------------------------------------------

    int allcoords[3 * P * Q * R] = { 0 };

    taskmap = (int*)Malloc(nprocs * sizeof(*taskmap));
    if (taskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for taskmap\n", nprocs * sizeof(*taskmap));
        exit(1);
    }

    itaskmap = (int*)Malloc(nprocs * sizeof(*itaskmap));
    if (itaskmap == NULL) {
        fprintf(stderr, "Could not get %d byte for itaskmap\n", nprocs * sizeof(*itaskmap));
        exit(1);
    }

    CHECK_MPI(MPI_Allgather(
        coords,
        sizeof(coords),
        MPI_BYTE,
        allcoords,
        sizeof(coords),
        MPI_BYTE,
        MPI_COMM_WORLD));

    for (int i = 0; i < nprocs; i++) {
        taskmap[(allcoords[i * 3 + 2] * Q * P) + (allcoords[i * 3 + 1] * P) + (allcoords[i * 3])] = i;
        itaskmap[i] = (allcoords[i * 3 + 2] * Q * P) + (allcoords[i * 3 + 1] * P) + (allcoords[i * 3]);
    }

    // ---------------------------------------------------------------------------

    CSR* Alocal = CSRm::init(
        local_size,
        num_rows,
        (local_size * 27),
        true,
        false,
        false,
        num_rows,
        ilower);

    // alloc COO
    itype* Arow = (itype*)Malloc(sizeof(itype*) * (local_size * 27));
    itype* Acol = (itype*)Malloc(sizeof(itype*) * (local_size * 27));
    vtype* Aval = (vtype*)Malloc(sizeof(vtype*) * (local_size * 27));

    itype count = 0;
    itype nz_count = 0;
    itype nnz = 0;
    Alocal->row[0] = 0;

    // ---------------------------------------------------------------------------

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int gi = p * nx + i;
                int gj = q * ny + j;
                int gk = r * nz + k;

                // Diagonal term
                Arow[nz_count] = count;
                Acol[nz_count] = count;
                Aval[nz_count] = 26.;
                nnz++;
                nz_count++;

                // Given gi, gj, gk, find p, q, r, i_loc, j_loc, k_loc

                // *** 1.0
                if (i == 0 && p == 0) {
                    // do nothing
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi - 1, gj, gk, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;
                }

                // *** 2.0
                if (i == nx - 1 && p == P - 1) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi + 1, gj, gk, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;
                }

                // *** 3.0
                if ((j == 0) && (q == 0)) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi, gj - 1, gk, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;

                    // *** 3.1
                    if ((i == 0) && (p == 0)) {
                        // do nothing
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi - 1, gj - 1, gk, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;
                    }

                    // *** 3.2
                    if ((i == nx - 1) && (p == P - 1)) {
                        // do nothing, no right neighbor
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi + 1, gj - 1, gk, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;
                    }
                }

                // *** 4.0
                if (j == ny - 1 && q == Q - 1) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi, gj + 1, gk, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;

                    // *** 4.1
                    if ((i == 0) && (p == 0)) {
                        // do nothing
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi - 1, gj + 1, gk, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;
                    }

                    // *** 4.2
                    if ((i == nx - 1) && (p == P - 1)) {
                        // do nothing, no right neighbor
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi + 1, gj + 1, gk, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;
                    }
                }

                // *** 5.0
                if (k == 0 && r == 0) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi, gj, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;

                    // *** 5.1
                    if ((i == 0) && (p == 0)) {
                        // do nothing
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi - 1, gj, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;
                    }

                    // *** 5.2
                    if ((i == nx - 1) && (p == P - 1)) {
                        // do nothing, no right neighbor
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi + 1, gj, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;
                    }

                    // *** 5.3
                    if ((j == 0) && (q == 0)) {
                        // do nothing
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi, gj - 1, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;

                        // *** 5.3.1
                        if ((i == 0) && (p == 0)) {
                            // do nothing
                        } else {
                            Arow[nz_count] = count;
                            Acol[nz_count] = internal_index(gi - 1, gj - 1, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                            Aval[nz_count] = -1.;
                            nnz++;
                            nz_count++;
                        }

                        // *** 5.3.2
                        if ((i == nx - 1) && (p == P - 1)) {
                            // do nothing
                        } else {
                            Arow[nz_count] = count;
                            Acol[nz_count] = internal_index(gi + 1, gj - 1, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                            Aval[nz_count] = -1.;
                            nnz++;
                            nz_count++;
                        }
                    }

                    // *** 5.4
                    if ((j == ny - 1) && (q == Q - 1)) {
                        // do nothing, no right neighbor
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi, gj + 1, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;

                        // *** 5.4.1
                        if ((i == 0) && (p == 0)) {
                            // do nothing
                        } else {
                            Arow[nz_count] = count;
                            Acol[nz_count] = internal_index(gi - 1, gj + 1, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                            Aval[nz_count] = -1.;
                            nnz++;
                            nz_count++;
                        }

                        // *** 5.4.2
                        if ((i == nx - 1) && (p == P - 1)) {
                            // do nothing
                        } else {
                            Arow[nz_count] = count;
                            Acol[nz_count] = internal_index(gi + 1, gj + 1, gk - 1, nx, ny, nz, P, Q, R) - ilower;
                            Aval[nz_count] = -1.;
                            nnz++;
                            nz_count++;
                        }
                    }
                }

                // *** 6.0
                if ((k == nz - 1) && (r == R - 1)) {
                    // do nothing, no right neighbor
                } else {
                    Arow[nz_count] = count;
                    Acol[nz_count] = internal_index(gi, gj, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                    Aval[nz_count] = -1.;
                    nnz++;
                    nz_count++;

                    // *** 6.1
                    if ((i == 0) && (p == 0)) {
                        // do nothing
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi - 1, gj, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;
                    }

                    // *** 6.2
                    if ((i == nx - 1) && (p == P - 1)) {
                        // do nothing, no right neighbor
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi + 1, gj, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;
                    }

                    // *** 6.3
                    if ((j == 0) && (q == 0)) {
                        // do nothing
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi, gj - 1, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;

                        // *** 6.3.1
                        if ((i == 0) && (p == 0)) {
                            // do nothing
                        } else {
                            Arow[nz_count] = count;
                            Acol[nz_count] = internal_index(gi - 1, gj - 1, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                            Aval[nz_count] = -1.;
                            nnz++;
                            nz_count++;
                        }

                        // *** 6.3.2
                        if ((i == nx - 1) && (p == P - 1)) {
                            // do nothing
                        } else {
                            Arow[nz_count] = count;
                            Acol[nz_count] = internal_index(gi + 1, gj - 1, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                            Aval[nz_count] = -1.;
                            nnz++;
                            nz_count++;
                        }
                    }

                    // *** 6.4
                    if ((j == ny - 1) && (q == Q - 1)) {
                        // do nothing, no right neighbor
                    } else {
                        Arow[nz_count] = count;
                        Acol[nz_count] = internal_index(gi, gj + 1, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                        Aval[nz_count] = -1.;
                        nnz++;
                        nz_count++;

                        // *** 6.4.1
                        if ((i == 0) && (p == 0)) {
                            // do nothing
                        } else {
                            Arow[nz_count] = count;
                            Acol[nz_count] = internal_index(gi - 1, gj + 1, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                            Aval[nz_count] = -1.;
                            nnz++;
                            nz_count++;
                        }

                        // *** 6.4.2
                        if ((i == nx - 1) && (p == P - 1)) {
                            // do nothing
                        } else {
                            Arow[nz_count] = count;
                            Acol[nz_count] = internal_index(gi + 1, gj + 1, gk + 1, nx, ny, nz, P, Q, R) - ilower;
                            Aval[nz_count] = -1.;
                            nnz++;
                            nz_count++;
                        }
                    }
                }

                Alocal->row[count + 1] = Alocal->row[count] + nnz;
                bubbleSort(&Acol[nz_count - nnz], &Aval[nz_count - nnz], nnz);
                nnz = 0;
                count++;
            }
        }
    }

    // ---------------------------------------------------------------------------

    Alocal->row[count] = nz_count; // check if
    for (itype j = 0; j < nz_count; j++) {
        Alocal->col[j] = Acol[j];
        Alocal->val[j] = Aval[j];
    }
    Alocal->nnz = nz_count;

    Free(Arow);
    Free(Acol);
    Free(Aval);

    POP_RANGE
    return Alocal;
}

int* read_laplacian_file(const char* file_name)
{
    char buffer[BUFSIZE];
    const char* params_list[] = { "nx", "ny", "nz", "P", "Q", "R" };

    int* lap_3d_parm = (int*)Malloc(sizeof(int) * LAP_N_PARAMS);
    FILE* fp = fopen(file_name, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] - Laplacia file not found! (%s)\n", file_name);
        exit(EXIT_FAILURE);
    }

    while (fgets(buffer, BUFSIZE, fp)) { /* READ a LINE */
        int buflen = 0;
        buflen = strlen(buffer);

        // Check that the buffer is big enough to read a line
        if (buffer[buflen - 1] != '\n' && !feof(fp)) {
            fprintf(stderr, "[ERROR] File %s. The line is too long, increase the BUFSIZE! Exit\n", file_name);
            exit(EXIT_FAILURE);
        }

        // skip empty lines and comments
        if (buflen > 0 && buffer[0] != '#') {
            char opt[20];
            int value, err;

            err = sscanf(buffer, "%s = %d\n", opt, &value);
            if (err != 2 || err == EOF) {
                fprintf(stderr, "[ERROR] Error reading file %s.\n", file_name);
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < LAP_N_PARAMS; i++) {
                if (strstr(opt, params_list[i]) != NULL) {
                    lap_3d_parm[i] = value;
                    break;
                }
            }
        }
    }
    fclose(fp);
    return lap_3d_parm;
}
