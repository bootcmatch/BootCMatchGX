#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NX 60 // numero di elementi in x
#define NY 30 // numero di elementi in y
#define NNODES ((NX + 1) * (NY + 1))
#define NELEMS (2 * NX * NY)
#define DOF (2 * NNODES) // 2 gradi di liberta' per nodo (ux, uy)
typedef struct {
    double x, y;
} Node;

typedef struct {
    int n[3]; // nodi di ogni elemento triangolare
} Element;
// Massimo numero di valori non zero stimati (6*6*NELEMS)
#define MAX_NNZ (36 * NELEMS)
typedef struct {
    int row;
    int col;
    double val;
} Triplet;

// Parametri materiale (acciaio tipico)
const double E = 210e9; // Modulo di Young (Pa)
const double nu = 0.3; // Coefficiente di Poisson
double lambda, mu; // costanti di Lamé
//
// Prototipi
void compute_lame_constants();
void generate_mesh(Node* nodes, Element* elems);
void print_mesh_info(Node* nodes, Element* elems);
void assemble_global(Node* nodes, Element* elems, double D[3][3], int* row_ptr, int* col_idx, double* val);
void apply_load(double* rhs, Node* nodes);
void apply_dirichlet_bc(double* rhs, int* row_ptr, int* col_idx, double* val, Node* nodes);
void export_matrix_market(const char* filename, int nrows, int* row_ptr, int* col_idx, double* val);
void compute_D_matrix(double D[3][3]);
//
//
int main()
{
    printf("NNODES=%d, NELEMS=%d, DOF=%d, MAX_NNZ=%d\n", NNODES, NELEMS, DOF, MAX_NNZ);
    Node* nodes = (Node*)malloc(NNODES * sizeof(Node));
    Element* elems = (Element*)malloc(NELEMS * sizeof(Element));

    if (!nodes || !elems) {
        printf("Errore allocazione memoria\n");
        return 1;
    }

    printf("compute lame constants\n");
    compute_lame_constants();

    printf("generate mesh\n");
    generate_mesh(nodes, elems);

    // print_mesh_info(nodes, elems);

    int* row_ptr = (int*)malloc((DOF + 1) * sizeof(int));
    int* col_idx = (int*)malloc(MAX_NNZ * sizeof(int));
    double* val = (double*)malloc(MAX_NNZ * sizeof(double));

    if (!row_ptr || !col_idx || !val) {
        printf("Errore allocazione memoria matrice\n");
        return 1;
    }

    double D[3][3];
    compute_D_matrix(D);

    assemble_global(nodes, elems, D, row_ptr, col_idx, val);

    double* rhs = (double*)calloc(DOF, sizeof(double));

    apply_dirichlet_bc(rhs, row_ptr, col_idx, val, nodes);

    // Primo passaggio: conta quanti non-zero restano
    int new_nnz = 0;
    for (int i = 0; i < row_ptr[DOF]; i++) {
        if (val[i] != 0.0) {
            new_nnz++;
        }
    }
    // Alloca nuove strutture compatte
    int* new_row_ptr = (int*)malloc((DOF + 1) * sizeof(int));
    int* new_col_idx = (int*)malloc(new_nnz * sizeof(int));
    double* new_val = (double*)malloc(new_nnz * sizeof(double));
    int k = 0;
    new_row_ptr[0] = 0;
    for (int i = 0; i < DOF; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            if (val[j] != 0.0) {
                new_col_idx[k] = col_idx[j];
                new_val[k] = val[j];
                k++;
            }
        }
        new_row_ptr[i + 1] = k;
    }

    apply_load(rhs, nodes);

    const char* filename = "K_matrix.mtx";

    export_matrix_market(filename, DOF, new_row_ptr, new_col_idx, new_val);

    const char* filename1 = "rhs.mtx";
    FILE* f1 = fopen(filename1, "w");
    if (!f1) {
        printf("Errore apertura file f1 \n");
        return 1;
    }
    fprintf(f1, "%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f1, "%d \n", DOF);
    for (int i = 0; i < DOF; i++) {
        fprintf(f1, "%.3f\n", rhs[i]);
    }
    fclose(f1);

    free(nodes);
    free(elems);
    free(row_ptr);
    free(col_idx);
    free(val);
    free(new_row_ptr);
    free(new_col_idx);
    free(new_val);
    free(rhs);
    return 0;
}

// Calcola le costanti di Lamé da E e nu
void compute_lame_constants()
{
    lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
    printf("Costanti di Lamé: lambda = %.3e, mu = %.3e\n", lambda, mu);
}
//
// Genera mesh triangolare regolare su quadrato [0,1]^2
void generate_mesh(Node* nodes, Element* elems)
{
    // Nodi
    for (int j = 0; j <= NY; j++) {
        for (int i = 0; i <= NX; i++) {
            int n = i + j * (NX + 1);
            nodes[n].x = (double)i / NX;
            nodes[n].y = (double)j / NY;
        }
    }

    // Elementi (due triangoli per ogni cella rettangolare)
    int e = 0;
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            int n1 = i + j * (NX + 1);
            int n2 = n1 + 1;
            int n3 = n1 + (NX + 1);
            int n4 = n3 + 1;

            // Primo triangolo
            elems[e].n[0] = n1;
            elems[e].n[1] = n2;
            elems[e].n[2] = n3;
            e++;
            // Secondo triangolo
            elems[e].n[0] = n2;
            elems[e].n[1] = n4;
            elems[e].n[2] = n3;
            e++;
        }
    }
}
// Ordina triplets per row,col (qsort)
int compare_triplets(const void* A, const void* B)
{
    Triplet* a = (Triplet*)A;
    Triplet* b = (Triplet*)B;
    if (a->row != b->row) {
        return a->row - b->row;
    }
    return a->col - b->col;
}

// Stampa sommaria mesh
void print_mesh_info(Node* nodes, Element* elems)
{
    printf("Numero nodi: %d\n", NNODES);
    printf("Coordinate nodi:\n");
    for (int i = 0; i < NNODES; i++) {
        printf("  Nodo %3d: (%.3f, %.3f)\n", i, nodes[i].x, nodes[i].y);
    }
    printf("\nNumero elementi: %d\n", NELEMS);
    printf("Elementi (nodi):\n");
    for (int e = 0; e < NELEMS; e++) {
        printf("  Elem %3d: %d, %d, %d\n", e, elems[e].n[0], elems[e].n[1], elems[e].n[2]);
    }
}

// Matrice materiale D (3x3) in piano di deformazione
void compute_D_matrix(double D[3][3])
{
    double c = E / ((1 + nu) * (1 - 2 * nu));
    D[0][0] = c * (1 - nu);
    D[0][1] = c * nu;
    D[0][2] = 0.0;
    D[1][0] = c * nu;
    D[1][1] = c * (1 - nu);
    D[1][2] = 0.0;
    D[2][0] = 0.0;
    D[2][1] = 0.0;
    D[2][2] = c * (1 - 2 * nu) / 2.0;
}

// Calcola area triangolo e matrici B e Ke
void element_stiffness(Node* nodes, Element* elem, double D[3][3], double Ke[6][6])
{
    int* n = elem->n;
    double x1 = nodes[n[0]].x, y1 = nodes[n[0]].y;
    double x2 = nodes[n[1]].x, y2 = nodes[n[1]].y;
    double x3 = nodes[n[2]].x, y3 = nodes[n[2]].y;

    double area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
    if (area <= 0) {
        printf("Warning: element with non-positive area!\n");
    }

    // Derivate funzioni forma rispetto a x,y
    double b[3], c[3];
    b[0] = y2 - y3;
    b[1] = y3 - y1;
    b[2] = y1 - y2;

    c[0] = x3 - x2;
    c[1] = x1 - x3;
    c[2] = x2 - x1;

    // Matrice B (3x6)
    double B[3][6] = { 0 };
    for (int i = 0; i < 3; i++) {
        B[0][2 * i] = b[i] / (2 * area);
        B[1][2 * i + 1] = c[i] / (2 * area);
        B[2][2 * i] = c[i] / (2 * area);
        B[2][2 * i + 1] = b[i] / (2 * area);
    }

    // Calcola Ke = area * B' * D * B
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            Ke[i][j] = 0.0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    Ke[i][j] += B[k][i] * D[k][l] * B[l][j];
                }
            }
            Ke[i][j] *= area;
        }
    }
}

// Assemblaggio matrice globale
void assemble_global(Node* nodes, Element* elems, double D[3][3], int* row_ptr, int* col_idx, double* val)
{
    Triplet* triplets;
    int triplet_count = 0;
    triplets = (Triplet*)malloc(MAX_NNZ * sizeof(Triplet));

    double Ke[6][6];

    for (int e = 0; e < NELEMS; e++) {
        element_stiffness(nodes, &elems[e], D, Ke);

        int* n = elems[e].n;

        for (int i = 0; i < 3; i++) {
            int gi = 2 * n[i];
            for (int j = 0; j < 3; j++) {
                int gj = 2 * n[j];

                for (int a = 0; a < 2; a++) {
                    for (int b = 0; b < 2; b++) {
                        int row = gi + a;
                        int col = gj + b;
                        double value = Ke[2 * i + a][2 * j + b];
                        if (value != 0.0) {
                            triplets[triplet_count].row = row;
                            triplets[triplet_count].col = col;
                            triplets[triplet_count].val = value;
                            triplet_count++;
                        }
                    }
                }
            }
        }
    }

    // Ora convertiamo COO in CSR

    qsort(triplets, triplet_count, sizeof(Triplet), compare_triplets);

    // Costruzione CSR (rimuove duplicati sommando valori)
    int nnz = 0;
    row_ptr[0] = 0;

    for (int i = 0; i < DOF; i++) {
        // riempie row_ptr per righe vuote (in seguito)
        row_ptr[i + 1] = -1;
    }

    int prev_row = -1;
    int prev_col = -1;
    for (int i = 0; i < triplet_count; i++) {
        if (triplets[i].row != prev_row) {
            // Completa row_ptr righe saltate
            for (int r = prev_row + 1; r <= triplets[i].row; r++) {
                row_ptr[r] = nnz;
            }
            prev_row = triplets[i].row;
            prev_col = -1;
        }
        if (triplets[i].col == prev_col) {
            val[nnz - 1] += triplets[i].val; // somma duplicati
        } else {
            col_idx[nnz] = triplets[i].col;
            val[nnz] = triplets[i].val;
            prev_col = triplets[i].col;
            nnz++;
        }
    }
    // Completa row_ptr fino a DOF
    for (int r = prev_row + 1; r <= DOF; r++) {
        row_ptr[r] = nnz;
    }

    free(triplets);

    printf("Matrice globale assemblata con %d valori non zero\n", nnz);
}

// Vincoliamo tutti i gradi di libertà dei nodi sul bordo sinistro (x=0) a spostamento nullo (u = v = 0).
void apply_dirichlet_bc(double* rhs, int* row_ptr, int* col_idx, double* val, Node* nodes)
{
    for (int i = 0; i < NNODES; i++) {
        if (nodes[i].x == 0.0) {
            int dof_x = 2 * i;
            int dof_y = 2 * i + 1;

            // azzera riga e colonna, inserisce 1 sulla diagonale
            for (int d = 0; d < 2; d++) {
                int dof = d == 0 ? dof_x : dof_y;
                rhs[dof] = 0.0;

                for (int j = row_ptr[dof]; j < row_ptr[dof + 1]; j++) {
                    val[j] = 0.0;
                }

                for (int irow = 0; irow < DOF; irow++) {
                    for (int j = row_ptr[irow]; j < row_ptr[irow + 1]; j++) {
                        if (col_idx[j] == dof) {
                            val[j] = 0.0;
                        }
                    }
                }

                // inserisce 1 sulla diagonale
                for (int j = row_ptr[dof]; j < row_ptr[dof + 1]; j++) {
                    if (col_idx[j] == dof) {
                        val[j] = 1.0;
                        break;
                    }
                }
            }
        }
    }
}

// Qui applichiamo una forza verticale (fy) distribuita sull'estremita' destra (x = 1):
void apply_load(double* rhs, Node* nodes)
{
    for (int i = 0; i < NNODES; i++) {
        if (nodes[i].x == 1.0) {
            int dof_y = 2 * i + 1;
            rhs[dof_y] = -1000.0; // forza verso il basso (N)
        }
    }
}

void export_matrix_market(const char* filename, int nrows, int* row_ptr, int* col_idx, double* val)
{
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Errore apertura file\n");
        return;
    }

    // Conta i non zero
    int nnz = row_ptr[nrows];

    fprintf(f, "%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%d %d %d\n", nrows, nrows, nnz);

    for (int i = 0; i < nrows; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            fprintf(f, "%d %d %.16e\n", i + 1, col_idx[j] + 1, val[j]); // 1-based
        }
    }

    fclose(f);
}
