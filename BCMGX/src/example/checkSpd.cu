#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    int n;
    double **a;
} Matrix;

/* Alloca matrice densa n x n inizializzata a 0 */
Matrix *alloc_matrix(int n) {
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    m->n = n;
    m->a = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        m->a[i] = (double *)calloc(n, sizeof(double));
    }
    return m;
}

void free_matrix(Matrix *m) {
    for (int i = 0; i < m->n; i++)
        free(m->a[i]);
    free(m->a);
    free(m);
}

/* Legge un file .mtx */
Matrix *read_mtx(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    char line[256];
    int symmetric = 0;

    /* Header */
    fgets(line, sizeof(line), f);
    if (strstr(line, "symmetric"))
        symmetric = 1;

    /* Skip commenti */
    do {
        fgets(line, sizeof(line), f);
    } while (line[0] == '%');

    int n, m, nnz;
    sscanf(line, "%d %d %d", &n, &m, &nnz);
    if (n != m) {
        fprintf(stderr, "La matrice non è quadrata\n");
        exit(1);
    }

    Matrix *A = alloc_matrix(n);

    for (int k = 0; k < nnz; k++) {
        int i, j;
        double v;
        fscanf(f, "%d %d %lf", &i, &j, &v);
        i--; j--;
        A->a[i][j] = v;
        if (symmetric && i != j)
            A->a[j][i] = v;
    }

    fclose(f);
    return A;
}

/* Controllo simmetria */
int is_symmetric(Matrix *A, double tol) {
    for (int i = 0; i < A->n; i++)
        for (int j = i + 1; j < A->n; j++)
            if (fabs(A->a[i][j] - A->a[j][i]) > tol) {
                fprintf(stderr, "Gli elementi (%d, %d) = %lf e (%d, %d) = %lf  differiscono\n",
                    i, j, A->a[i][j],
                    j, i, A->a[j][i]);
                return 0;
            }
    return 1;
}

/* Cholesky: ritorna 1 se SPD, 0 altrimenti */
int is_spd(Matrix *A) {
    int n = A->n;
    double **L = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        L[i] = (double *)calloc(n, sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A->a[i][j];
            for (int k = 0; k < j; k++)
                sum -= L[i][k] * L[j][k];

            if (i == j) {
                if (sum <= 0.0) {
                    return 0;
                }
                L[i][j] = sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }

    for (int i = 0; i < n; i++)
        free(L[i]);
    free(L);

    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Uso: %s matrix.mtx\n", argv[0]);
        return 1;
    }

    Matrix *A = read_mtx(argv[1]);

    if (!is_symmetric(A, 1e-4)) {
        printf("Matrice NON simmetrica\n");
    } else if (!is_spd(A)) {
        printf("Matrice simmetrica ma NON definita positiva\n");
    } else {
        printf("Matrice simmetrica definita positiva (SPD)\n");
    }

    free_matrix(A);
    return 0;
}
