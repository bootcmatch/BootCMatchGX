#include "fem.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utility/globals.h"
#include "utility/mpi.h"
#include "utility/utils.h"

#define DEBUG 0

CSR *resize_nnz(CSR *A, int new_nnz)
{
    int *new_col = MALLOC(int, new_nnz);
    long *new_col8 = MALLOC(long, new_nnz);
    double *new_val = MALLOC(double, new_nnz);

    int k = 0;
    for (int i = 0; i < A->n; i++) {
        int start = A->row[i];
        int end   = A->row[i+1];
        A->row[i] = k;

        for (int j = start; j < end; j++) {
            if (!IS_ZERO(A->val[j])) {
                new_col[k] = A->col[j];
                new_col8[k] = A->col8[j];
                new_val[k] = A->val[j];
                k++;
            }
        }
    }
    A->row[A->n] = k;

    free(A->col);
    free(A->col8);
    free(A->val);
    A->col = new_col;
    A->col8 = new_col8;
    A->val = new_val;
    A->nnz = k;

    return A;
}


// Matrice materiale D (3x3) in piano di deformazione
void compute_D_matrix(double E, double nu, double D[3][3])
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

// Calcola le costanti di Lamé lambda e mu da E e nu
void compute_lame_constants(double E, double nu, double *lambda, double *mu)
{
  *lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
  *mu = E / (2 * (1 + nu));
  #if DEBUG
  if (log_file) {
    fprintf(log_file, "Costanti di Lamé: lambda = %.3e, mu = %.3e\n", *lambda, *mu);
  }
  #endif
}

// Genera mesh triangolare regolare su quadrato [0,1]^2
void generate_mesh_1proc(int nx, int ny, int P, int Q, vector<Node> **nodesRet, vector<Element> **elemsRet)
{
  const int NODES_X = (nx + 1) * P;
  const int NODES_Y = (ny + 1) * Q;
  const int NNODES = NODES_X * NODES_Y;
  const int NELEMS = 2 * (NODES_X - 1) * (NODES_Y - 1);

  #if DEBUG
  if (log_file) {
    fprintf(log_file, "NNODES=%d, NELEMS=%d\n", NNODES, NELEMS);
  }
  #endif

  vector<Node> *nodes = Vector::init<Node>(NNODES, true, false);
  vector<Element> *elems = Vector::init<Element>(NELEMS, true, false);

  // Nodi
  for (int j = 0; j < NODES_Y; j++)
  {
    for (int i = 0; i < NODES_X; i++)
    {
      int n = i + j * (NODES_X);
      nodes->val[n].x = (double)i / (NODES_X - 1);
      nodes->val[n].y = (double)j / (NODES_Y - 1);
      nodes->val[n].gn = n;

      #if DEBUG
      if (log_file) {
        fprintf(log_file, "Node %2d (%d, %d) = (%lf, %lf)\n", n, i, j, nodes->val[n].x, nodes->val[n].y);
      }
      #endif
    }
  }

  // Elementi (due triangoli per ogni cella rettangolare)
  int e = 0;
  for (int j = 0; j < NODES_Y - 1; j++)
  {
    for (int i = 0; i < NODES_X - 1; i++)
    {
      int n1 = i + j * NODES_X;
      int n2 = n1 + 1;
      int n3 = n1 + NODES_X;
      int n4 = n3 + 1;

      // Primo triangolo
      #if DEBUG
      if (log_file) {
          int rec = 2 * i + 2 * j * (NODES_X - 1);
          fprintf(log_file, "Triangle %2d, rec %2d [%2d, %2d, %2d]\n", e, rec, n1, n2, n3);
      }
      #endif
      elems->val[e].n[0] = n1;
      elems->val[e].n[1] = n2;
      elems->val[e].n[2] = n3;
      elems->val[e].gn[0] = n1;
      elems->val[e].gn[1] = n2;
      elems->val[e].gn[2] = n3;
      e++;

      // Secondo triangolo
      #if DEBUG
      if (log_file) {
        int rec = 2 * i + 2 + 2 * j * (NODES_X - 1);
        fprintf(log_file, "Triangle %2d, rec %2d [%2d, %2d, %2d]\n", e, rec, n2, n4, n3);
      }
      #endif
      elems->val[e].n[0] = n2;
      elems->val[e].n[1] = n4;
      elems->val[e].n[2] = n3;
      elems->val[e].gn[0] = n2;
      elems->val[e].gn[1] = n4;
      elems->val[e].gn[2] = n3;
      e++;
    }
  }

  *nodesRet = nodes;
  *elemsRet = elems;
}

CSR *get_matrix_from_triplets_1proc(Triplet *triplets, int triplet_count, int dof)
{
    qsort(triplets, triplet_count, sizeof(Triplet), compare_triplets);
    triplet_count = compact_triplets(triplets, triplet_count);

    CSR *A = CSRm::init(dof, dof, triplet_count, true, false, false, dof, 0);

    int nnz = 0;
    int cur_row = 0;
    A->row[0] = 0;

    for (int i = 0; i < triplet_count; i++) {
        int row = triplets[i].row;
        int col = triplets[i].col;

        // riempi righe vuote
        while (cur_row < row) {
            cur_row++;
            A->row[cur_row] = nnz;
        }

        A->col[nnz] = col;
        A->col8[nnz] = col;
        A->val[nnz] = triplets[i].val;
        nnz++;
    }

    // completa le righe fino a dof
    while (cur_row < dof) {
        cur_row++;
        A->row[cur_row] = nnz;
    }

    A->nnz = nnz;
    return A;
}


void generate_mesh(int nx, int ny, int p, int q, int P, int Q, int row_start, int row_end, vector<Node> **nodesRet, vector<Element> **elemsRet)
{
    const int NODES_X = (nx + 1) * P;
    const int NODES_Y = (ny + 1) * Q;
    const int NNODES = NODES_X * NODES_Y;
    const int NELEMS = 2 * (NODES_X - 1) * (NODES_Y - 1);

    // const int NX_GLB = nx * P;
    // const int NY_GLB = ny * Q;
    // const int NNODES_GLB = (NX_GLB + 1) * (NY_GLB + 1);
    // const int NELEMS_GLB = 2 * NX_GLB * NY_GLB;

    // Allocazione strutture (sovradimensionate per sicurezza, verranno popolate selettivamente)
    vector<Node> *nodes = Vector::init<Node>(NNODES, true, false);
    vector<Element> *elems = Vector::init<Element>(NELEMS, true, false);

    #if DEBUG
    if (log_file) {
        fprintf(log_file, "(p, q) = [%d, %d]\n", p, q);
    }
    #endif

    // 1. Generazione di tutti i nodi (necessario per avere coordinate e GN per i Dirichlet globali)
    for (int j = 0; j < NODES_Y; j++)
    {
      for (int i = 0; i < NODES_X; i++)
      {
        int gn = i + j * (NODES_X);
        nodes->val[gn].x = (double)i / (NODES_X - 1);
        nodes->val[gn].y = (double)j / (NODES_Y - 1);
        nodes->val[gn].gn = gn;

        #if DEBUG
        if (log_file) {
          fprintf(log_file, "Node %2d (%d, %d) = (%lf, %lf)\n", gn, i, j, nodes->val[gn].x, nodes->val[gn].y);
        }
        #endif
      }
    }

    // 2. Generazione Elementi "Interessanti" per questo processo
    int e_count = 0;
    for (int j = 0; j < NODES_Y - 1; j++) {
        for (int i = 0; i < NODES_X - 1; i++) {
            // Nodi della cella
            int n1 = i + j * NODES_X;
            int n2 = n1 + 1;
            int n3 = n1 + NODES_X;
            int n4 = n3 + 1;

            int triangles[2][3] = {{n1, n2, n3}, {n2, n4, n3}};

            for (int t = 0; t < 2; t++) {
                bool touches_my_rows = false;
                for (int n_idx = 0; n_idx < 3; n_idx++) {
                    int node_id = triangles[t][n_idx];
                    int dof_x = 2 * node_id;
                    int dof_y = 2 * node_id + 1;
                    // Se almeno un DOF dell'elemento cade nelle mie righe, devo calcolarlo
                    if ((dof_x >= row_start && dof_x < row_end) || 
                        (dof_y >= row_start && dof_y < row_end)) {
                        touches_my_rows = true;
                        break;
                    }
                }

                if (touches_my_rows) {
                    #if DEBUG
                    if (log_file) {
                        // int rec = 2 * i + 2 * j * (NODES_X - 1);
                        // fprintf(log_file, "Triangle %2d, rec %2d [%2d, %2d, %2d]\n", e_count, rec, triangles[t][0], triangles[t][1], triangles[t][2]);
                        fprintf(log_file, "Triangle %2d [%2d, %2d, %2d]\n", e_count, triangles[t][0], triangles[t][1], triangles[t][2]);
                    }
                    #endif

                    elems->val[e_count].n[0] = triangles[t][0];
                    elems->val[e_count].n[1] = triangles[t][1];
                    elems->val[e_count].n[2] = triangles[t][2];
                    elems->val[e_count].gn[0] = triangles[t][0];
                    elems->val[e_count].gn[1] = triangles[t][1];
                    elems->val[e_count].gn[2] = triangles[t][2];
                    e_count++;
                }
            }
        }
    }
    elems->n = e_count;
    *nodesRet = nodes;
    *elemsRet = elems;
}

Triplet *assemble_local(vector<Node> *nodes, vector<Element> *elems, double D[3][3], int *count, int row_start, int row_end)
{
    int max_nnz = 36 * elems->n;
    Triplet *triplets = MALLOC(Triplet, max_nnz);
    int triplet_count = 0;
    double Ke[6][6];

    for (int e = 0; e < elems->n; e++) {
        element_stiffness(nodes, &elems->val[e], D, Ke);
        int *gn = elems->val[e].gn;

        for (int i = 0; i < 3; i++) {
            int gi = 2 * gn[i]; // Global index nodo i
            for (int a = 0; a < 2; a++) {
                int row_global = gi + a;

                // FILTRO FONDAMENTALE: Processo solo se possiedo questa riga
                if (row_global >= row_start && row_global < row_end) {
                    
                    for (int j = 0; j < 3; j++) {
                        int gj = 2 * gn[j]; // Global index nodo j
                        for (int b = 0; b < 2; b++) {
                            int col_global = gj + b; // INDICE COLONNA GLOBALE
                            
                            double value = Ke[2 * i + a][2 * j + b];
                            if (value != 0.0) {
                                triplets[triplet_count].row = row_global;
                                triplets[triplet_count].col = col_global; // Resta globale!
                                triplets[triplet_count].val = value;
                                triplet_count++;
                            }
                        }
                    }
                }
            }
        }
    }
    *count = triplet_count;
    return triplets;
}

// Compatta il vettore (già ordinato), sommando i valori degli elementi (vicini)
// che hanno la stessa posizione (riga, colonna)
int compact_triplets(Triplet *triplets, int triplet_count) {
    if (triplet_count == 0) return 0;

    int j = 0;  // indice di scrittura nel vettore compatto

    for (int i = 1; i < triplet_count; i++) {
        if (triplets[i].row == triplets[j].row && triplets[i].col == triplets[j].col) {
            // stessa posizione: sommo i valori
            triplets[j].val += triplets[i].val;
        } else {
            // nuova posizione: avanzo l'indice di scrittura
            j++;
            triplets[j] = triplets[i];
        }
    }

    return j + 1;  // numero di elementi nel vettore compatto
}

CSR* get_matrix_from_triplets(Triplet* triplets, int triplet_count,
                              int row_start, int row_end, int total_dof_global)
{
    int local_nrows = row_end - row_start;
    
    // Inizializza la matrice locale:
    // N. righe = local_nrows
    // N. colonne = total_dof_global (IMPORTANTE: usiamo dimensione globale per le colonne)
    CSR* A = CSRm::init(local_nrows, total_dof_global, triplet_count,
                        true, false, false,
                        total_dof_global, row_start);

    // Inizializza row[]
    for (int i = 0; i <= local_nrows; i++)
        A->row[i] = 0;

    // Ordina i triplet per (row, col)
    qsort(triplets, triplet_count, sizeof(Triplet), compare_triplets);

    int nnz = 0;
    int prev_row = row_start - 1;
    int prev_col = -1;

    // Flag per indicare che non stiamo shiftando le colonne
    // A->col_shifted = 0; 

    for (int i = 0; i < triplet_count; i++)
    {
        int r = triplets[i].row; // Globale
        int c = triplets[i].col; // Globale

        // Filtra solo le righe locali (sicurezza aggiuntiva)
        if (r < row_start || r >= row_end)
            continue;

        // Aggiorna row[] (basato su indici locali)
        while (prev_row < r)
        {
            int lr = prev_row - row_start + 1;
            if (lr >= 0 && lr <= local_nrows)
                A->row[lr] = nnz;
            prev_row++;
            prev_col = -1;
        }

        // Colonna: USIAMO DIRETTAMENTE L'INDICE GLOBALE
        // Nessuna sottrazione di row_start qui!
        int cl = c; 

        // Somma duplicati
        if (cl == prev_col && nnz > 0)
        {
            A->val[nnz - 1] += triplets[i].val;
        }
        else
        {
            A->col[nnz] = cl;
            A->col8[nnz] = cl;
            A->val[nnz] = triplets[i].val;
            prev_col = cl;
            nnz++;
        }
    }

    // Completa row[] per eventuali righe vuote finali
    for (int r = prev_row + 1; r < row_end; r++)
    {
        int lr = r - row_start;
        if (lr >= 0 && lr <= local_nrows)
            A->row[lr] = nnz;
    }

    A->row[local_nrows] = nnz;
    A->nnz = nnz;

    #if DEBUG
    if (log_file) {
      fprintf(log_file, "Matrice locale assemblata: Rows [%d,%d), Cols Globali [0,%d), NNZ=%d\n",
             row_start, row_end, total_dof_global, nnz);
    }
    #endif

    return A;
}

// Calcola area triangolo e matrici B e Ke
void element_stiffness(vector<Node> *nodes, Element *elem, double D[3][3], double Ke[6][6] /*, int li, int lj*/)
{
  int *n = elem->n;
  
  // double x1 = nodes->val[n[0]].x + li, y1 = nodes->val[n[0]].y + lj;
  // double x2 = nodes->val[n[1]].x + li, y2 = nodes->val[n[1]].y + lj;
  // double x3 = nodes->val[n[2]].x + li, y3 = nodes->val[n[2]].y + lj;

  double x1 = nodes->val[n[0]].x;
  double y1 = nodes->val[n[0]].y;

  double x2 = nodes->val[n[1]].x;
  double y2 = nodes->val[n[1]].y;

  double x3 = nodes->val[n[2]].x;
  double y3 = nodes->val[n[2]].y;

  double area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
  if (area <= 0)
  {
    DIE("Element with non-positive area!\n");
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
  double B[3][6] = {0};
  for (int i = 0; i < 3; i++)
  {
    B[0][2 * i] = b[i] / (2 * area);
    B[1][2 * i + 1] = c[i] / (2 * area);
    B[2][2 * i] = c[i] / (2 * area);
    B[2][2 * i + 1] = b[i] / (2 * area);
  }

  // Calcola Ke = area * B' * D * B
  for (int i = 0; i < 6; i++)
  {
    for (int j = 0; j < 6; j++)
    {
      Ke[i][j] = 0.0;
      for (int k = 0; k < 3; k++)
      {
        for (int l = 0; l < 3; l++)
        {
          Ke[i][j] += B[k][i] * D[k][l] * B[l][j];
        }
      }
      Ke[i][j] *= area;
    }
  }
}

// Ordina triplets per row,col (qsort)
int compare_triplets(const void *A, const void *B)
{
  Triplet *a = (Triplet *)A;
  Triplet *b = (Triplet *)B;
  if (a->row != b->row)
    return a->row - b->row;
  return a->col - b->col;
}

int apply_dirichlet_bc(vector<double> *rhs, CSR *A, vector<Node> *nodes)
{
    int row_start = A->row_shift;
    int row_end   = row_start + A->n; // A->n qui è local_nrows

    // Iteriamo su TUTTI i nodi globali per trovare quelli Dirichlet
    for (int i = 0; i < nodes->n; i++) {
        
        // Condizione Dirichlet (es. x == 0)
        if (IS_ZERO(nodes->val[i].x)) {
            
            // I gradi di libertà globali da vincolare
            int gdof_x = 2 * nodes->val[i].gn;
            int gdof_y = 2 * nodes->val[i].gn + 1;
            int dofs[2] = {gdof_x, gdof_y};

            for (int k = 0; k < 2; k++) {
                int target_global_dof = dofs[k];

                // 1. AZZERAMENTO RIGA (Solo se la riga è mia)
                if (target_global_dof >= row_start && target_global_dof < row_end) {
                    int local_row = target_global_dof - row_start;
                    
                    if (rhs) rhs->val[local_row] = 0.0;

                    // Scorro la riga per mettere 1 sulla diagonale e 0 altrove
                    for (int j = A->row[local_row]; j < A->row[local_row + 1]; j++) {
                        if (A->col[j] == target_global_dof)
                            A->val[j] = 1.0;
                        else
                            A->val[j] = 0.0;
                    }
                }

                // 2. AZZERAMENTO COLONNA (Simmetria)
                // Devo controllare in TUTTA la mia matrice locale se c'è
                // qualche riferimento a questa colonna globale 'target_global_dof'
                // e azzerarlo (a meno che non sia l'elemento diagonale della riga vincolata)
                
                for (int r = 0; r < A->n; r++) { // loop su righe locali
                    int current_global_row = r + row_start;
                    
                    // Non azzeriamo la diagonale della riga Dirichlet stessa
                    if (current_global_row == target_global_dof) continue;

                    for (int j = A->row[r]; j < A->row[r+1]; j++) {
                        if (A->col[j] == target_global_dof) {
                            A->val[j] = 0.0;
                        }
                    }
                }
            }
        }
    }

    // Ricalcola NNZ effettivi (opzionale, solo per statistica)
    int new_nnz = 0;
    for (int i = 0; i < A->row[A->n]; i++) {
        if (!IS_ZERO(A->val[i])) new_nnz++;
    }
    return new_nnz;
}

// Qui applichiamo una forza verticale (fy) distribuita sull'estremita' destra (x = 1):
// TODO: ricontrollare
void apply_load(vector<double> *rhs, vector<Node> *nodes, CSR* A)
{
  int row_start = A->row_shift;
  int row_end   = row_start + A->n;

  for (int i = 0; i < nodes->n; i++)
  {
    if (IS_ZERO(nodes->val[i].x - 1.0))
    {
      int dof_y = 2 * nodes->val[i].gn + 1;
      if (dof_y >= row_start && dof_y < row_end) {
        rhs->val[dof_y - row_start] = -1000.0;
      }
    }
  }
}

void sort_csr_rows(CSR *A) {
  for (int i = 0; i < A->n; i++) {
    int start = A->row[i];
    int end = A->row[i+1];
    for (int j = start; j < end - 1; j++) {
      for (int k = j + 1; k < end; k++) {
        if (A->col[k] < A->col[j]) {
          int tc = A->col[j];
          A->col[j] = A->col[k];
          A->col[k] = tc;
          double tv = A->val[j];
          A->val[j] = A->val[k];
          A->val[k] = tv;
        }
      }
    }
  }
}

void export_matrix_market(const char *filename, CSR *A)
{
  int nrows = A->n;
  int *row_ptr = A->row;
  int *col_idx = A->col;
  double *val = A->val;

  FILE *f = fopen(filename, "w");
  if (!f)
  {
    DIE("Errore apertura file %s\n", filename);
  }

  // Conta i non zero
  int nnz = row_ptr[nrows];

  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%d %d %d\n", nrows, nrows, nnz);

  for (int i = 0; i < nrows; i++)
  {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
    {
      fprintf(f, "%d %d %.16e\n", i + 1, col_idx[j] + 1, val[j]); // 1-based
    }
  }

  fclose(f);
}

void export_matrix_market(const char *filename, vector<double> *rhs) {
  FILE *f = fopen(filename, "w");
  if (!f)
  {
    DIE("Errore apertura file %s\n", filename);
  }

  fprintf(f, "%%%%MatrixMarket matrix array real general\n");
  fprintf(f, "%d 1\n", rhs->n);
  for (int i = 0; i < rhs->n; i++)
  {
    fprintf(f, "%.16e\n", rhs->val[i]);
  }

  fclose(f);
}

CSR* generateLocalFem(itype nx, itype ny, itype P, itype Q, double E, double nu, vector<double>** rhs_local_out) {
    _MPI_ENV;

    printf("compute D matrix\n");
    double D[3][3];
    compute_D_matrix(E, nu, D);

    // costanti di Lamé
    printf("compute lame constants\n");
    double lambda, mu;
    compute_lame_constants(E, nu, &lambda, &mu);

    // ---------------------------------------------------------------------------
    // Creazione communicator 2D
    // ---------------------------------------------------------------------------

    MPI_Comm NEWCOMM;

    int dims[2] = { P, Q };
    int periods[2] = { false, false };
    int coords[2] = { 0, 0 };
    int my2id;

    CHECK_MPI(MPI_Dims_create(nprocs, 2, dims));
    CHECK_MPI(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, false, &NEWCOMM));
    CHECK_MPI(MPI_Comm_rank(NEWCOMM, &my2id));
    CHECK_MPI(MPI_Cart_coords(NEWCOMM, my2id, 2, coords));

    int p = coords[0];
    int q = coords[1];

    int allcoords[2 * P * Q] = { 0 };

    taskmap = MALLOC(int, nprocs);
    itaskmap = MALLOC(int, nprocs);

    CHECK_MPI(MPI_Allgather(
        coords,
        sizeof(coords),
        MPI_BYTE,
        allcoords,
        sizeof(coords),
        MPI_BYTE,
        MPI_COMM_WORLD));

    for (int i = 0; i < nprocs; i++) {
        taskmap[(allcoords[i * 2 + 1] * P) + (allcoords[i * 2])] = i;
        itaskmap[i] = (allcoords[i * 2 + 1] * P) + (allcoords[i * 2]);
    }

    // ---------------------------------------------------------------------------
    // Divisione righe
    // ---------------------------------------------------------------------------

    int dof = 2 * (nx + 1) * P * (ny + 1) * Q;

    int rows_per_proc = dof / nprocs;

    int logical_id = itaskmap[myid];

    int row_start = logical_id * rows_per_proc;
    int row_end = (logical_id == nprocs - 1) ? dof : row_start + rows_per_proc;

    ASSERT(row_end - row_start);

    // ---------------------------------------------------------------------------
    // Mesh locale
    // ---------------------------------------------------------------------------

    printf("generate mesh (p = %d, q = %d)\n", p, q);
    vector<Node>* nodes = NULL;
    vector<Element>* elems = NULL;
    generate_mesh(nx, ny, p, q, P, Q, row_start, row_end, &nodes, &elems);

    // ---------------------------------------------------------------------------
    // Assemblaggio locale
    // ---------------------------------------------------------------------------

    int triplet_count = 0;

    Triplet* triplets = assemble_local(nodes, elems, D, &triplet_count, row_start, row_end);

    CSR* A_local = get_matrix_from_triplets(triplets, triplet_count, row_start, row_end, dof);
    // sort_csr_rows(A_local);
    free(triplets);

    vector<double>* rhs_local = Vector::init<double>(A_local->n, true, false);
    Vector::fillWithValue<double>(rhs_local, 0);

    int new_nnz = apply_dirichlet_bc(rhs_local, A_local, nodes);
    A_local = resize_nnz(A_local, new_nnz);
    apply_load(rhs_local, nodes, A_local);
    if (A_local->row_shift) {
        CSRm::shift_cols(A_local, -A_local->row_shift);
        A_local->col_shifted = -A_local->row_shift;
    }

    Vector::free(nodes);
    Vector::free(elems);

    if (rhs_local_out) {
      *rhs_local_out = rhs_local;
    } else {
      Vector::free(rhs_local);
    }
    return A_local;
}

CSR* generateLocalFem_1proc(itype nx, itype ny, itype P, itype Q, double E, double nu, vector<double>** rhs_out) {
    // _MPI_ENV;

    printf("compute D matrix\n");
    double D[3][3];
    compute_D_matrix(E, nu, D);

    // costanti di Lamé
    printf("compute lame constants\n");
    double lambda, mu;
    compute_lame_constants(E, nu, &lambda, &mu);

    printf("generate mesh\n");
    vector<Node>* nodes = NULL;
    vector<Element>* elems = NULL;
    generate_mesh_1proc(nx, ny, P, Q, &nodes, &elems);
    // print_mesh_info(nodes, elems);

    int triplet_count = 0;
    int dof = 2 * nodes->n; // 2 gradi di liberta' per nodo (ux, uy)
    Triplet* triplets = assemble_local(nodes, elems, D, &triplet_count, /*0, 0,*/ 0, dof);
    CSR* A = get_matrix_from_triplets_1proc(triplets, triplet_count, dof);
    free(triplets);
    // sort_csr_rows(A);

    printf("vincoliamo tutti i gradi di libertà dei nodi sul bordo sinistro (x=0) a spostamento nullo (u = v = 0).\n");
    vector<double>* rhs = Vector::init<double>(A->n /* DOF */, true, false);
    Vector::fillWithValue<double>(rhs, 0);
    int new_nnz = apply_dirichlet_bc(rhs, A, nodes);
    A = resize_nnz(A, new_nnz);

    printf("Qui applichiamo una forza verticale (fy) distribuita sull'estremita' destra (x = 1)\n");
    apply_load(rhs, nodes, A);

    Vector::free(nodes);
    Vector::free(elems);

    if (rhs_out) {
      *rhs_out = rhs;
    } else {
      Vector::free(rhs);
    }
    return A;
}

