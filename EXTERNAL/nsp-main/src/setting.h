#if KERNEL_TYPE == 0
  #define makeC_BLOCKSIZE 1024
  #define auction_BLOCKSIZE 1024
  #define assignWtoM_BLOCKSIZE 1024
  #define makeAH_BLOCKSIZE 1024
  #define make_c_BLOCKSIZE 1024
  #define make_w_BLOCKSIZE 1024
  #define write_T_BLOCKSIZE 1024
#elif KERNEL_TYPE == 1
  #define write_T_BLOCKSIZE 1024
  #define makeC_BLOCKSIZE 1024
  #define auction_BLOCKSIZE 1024
  #define assignWtoM_BLOCKSIZE 1024
  #define makeAH_BLOCKSIZE 1024
  #define make_c_BLOCKSIZE 1024
  #define make_w_BLOCKSIZE 1024
  #define aggregate_unsymmetric_BLOCKSIZE 1024
#endif
#define BLOCKSIZE 512
// Matrix's value type
#define vtype double
// Matrix's index type
#define itype  int
// Matrix's sizes  type
#define stype unsigned long //art unsigned int
// Matrix's global size and row shift
#define gstype unsigned long
#define gsstype long int

#define VTYPE_MPI MPI_DOUBLE
#define ITYPE_MPI MPI_INT


#define VERBOSE 0
#define HARD_DEBUG 0

//----__---__----__---__-----__----__----___---__-
#define PAIR_AGG_CPU 2
#define CENTRALIZED_RAP 0
#define CENTRALIZED_SUITOR 0

#define LOCAL_COARSEST 1

#define SMART_VECTOR_AGGREGATION 1
#define SMART_AGGREGATE_GETSET_GPU 1
#define DEBUG_SMART_AGG 0
#define R_SMART_VECTOR_AGGREGATION 1
//----__---__----__---__-----__----__----___---__-

//---- TO CHECK:
#define MATRIX_MATRIX_MUL_TYPE 1
// 0 cuSPARSE 1 nsparse

#define CG_VERSION 2
#define CSR_JACOBI_TYPE 0

#define CSR_VECTOR_MUL_GENERAL_TYPE 1
#define CSR_VECTOR_MUL_A_TYPE 1
#define CSR_VECTOR_MUL_P_TYPE 1 //2
#define CSR_VECTOR_MUL_R_TYPE 1

#define MAXIMUM_PRODUCT_MATRIX_OP 1
#define GALERKIN_PRODUCT_TYPE 1
