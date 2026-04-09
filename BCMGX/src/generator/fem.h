#pragma once

#include "datastruct/vector.h"
#include "datastruct/CSR.h"
#include "datastruct/FiniteElements.h"

#define DEFAULT_FEM_NX 10 //800
#define DEFAULT_FEM_NY 10 //800
#define DEFAULT_FEM_P 2
#define DEFAULT_FEM_Q 2
#define DEFAULT_FEM_E 210e9 // Modulo di Young (Pa)
#define DEFAULT_DEM_NU 0.3 // Coefficiente di Poisson

void compute_D_matrix(double E, double nu, double D[3][3]);
void compute_lame_constants(double E, double nu, double *lambda, double *mu);

void generate_mesh_1proc(int nx, int ny, int P, int Q, vector<Node> **nodes, vector<Element> **elems);
void generate_mesh(int nx, int ny, int p, int q, int P, int Q, int row_start, int row_end, vector<Node> **nodesRet, vector<Element> **elemsRet);

Triplet *assemble_local(vector<Node> *nodes, vector<Element> *elems, double D[3][3], int *count, int row_start, int row_end);
CSR *get_matrix_from_triplets_1proc(Triplet *triplets, int triplet_count, int dof);
CSR *get_matrix_from_triplets(Triplet *triplets, int triplet_count, int row_start, int row_end, int dof);
void element_stiffness(vector<Node> *nodes, Element *elem, double D[3][3], double Ke[6][6]);
int compare_triplets(const void *A, const void *B);
int apply_dirichlet_bc(vector<double> *rhs, CSR *A, vector<Node> *nodes);
void apply_load(vector<double> *rhs, vector<Node> *nodes, CSR* A);
void sort_csr_rows(CSR *A);
void export_matrix_market(const char *filename, CSR *A);
void export_matrix_market(const char *filename, vector<double> *rhs);
CSR *resize_nnz(CSR *A, int new_nnz);
int compact_triplets(Triplet *triplets, int triplet_count);
CSR* generateLocalFem(itype nx, itype ny, itype P, itype Q, double E, double nu, vector<double>** rhs_local_out);
CSR* generateLocalFem_1proc(itype nx, itype ny, itype P, itype Q, double E, double nu, vector<double>** rhs_out);

