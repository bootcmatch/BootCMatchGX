#define BLOCKSIZE 512 // 1024C
// Matrix's value type
#define vtype double
// Matrix's index type
#define itype int
#define MPI_ITYPE MPI_INT

// Matrix's sizes  type
#define stype unsigned int
// Matrix's global size and row shift
#define gstype unsigned long
#define gsstype long int

#define VTYPE_MPI MPI_DOUBLE
#define ITYPE_MPI MPI_INT
#define STYPE_MPI MPI_UINT32_T
#define GSTYPE_MPI MPI_UINT64_T

#define VERBOSE 0

// #define LOCAL_COARSEST 0

#define SMART_VECTOR_AGGREGATION 1
#define SMART_AGGREGATE_GETSET_GPU 1
#define DEBUG_SMART_AGG 0
#define R_SMART_VECTOR_AGGREGATION 1

#define MATRIX_MATRIX_MUL_TYPE 1

#define FULL_MASK 0xffffffff

#define CG_VERSION 2
#define CSR_JACOBI_TYPE 0

#define CSR_VECTOR_MUL_GENERAL_TYPE 1
#define CSR_VECTOR_MUL_A_TYPE 1
#define CSR_VECTOR_MUL_P_TYPE 1 // 2
#define CSR_VECTOR_MUL_R_TYPE 1

#define MAXIMUM_PRODUCT_MATRIX_OP 1
#define GALERKIN_PRODUCT_TYPE 1

#define USECOLSHIFT 1
#define BUFSIZE 1024

#define NUM_THR 1024
// TODO should we get the max number of threads from device?
#define MAX_THREADS 1024
