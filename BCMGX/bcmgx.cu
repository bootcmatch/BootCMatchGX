#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <unistd.h>
#include <getopt.h>

#define DIE CHECK_DEVICE(cudaDeviceSynchronize());MPI_Finalize();exit(0);

#include "matrix/scalar.h"
#include "matrix/vector.h"
#include "matrix/matrixIO.h"
#include "utility/myMPI.h"
#include "utility/handles.h"
#include "AMG/AMG.h"
#include "solver/solver.h"
#include <string>
#include "matrix/distribuite.h"

#define MAX_NNZ_PER_ROW_LAP 5
#define MPI 1
#ifndef __GNUC__
typedef int (*__compar_fn_t)(const void *, const void *);
#endif

using namespace std;

CSR* generateLocalLaplacian3D(itype n){
  _MPI_ENV;
  // global number of rows
  itype N = n * n * n;
  /* Each processor knows only of its own rows - the range is denoted by ilower
     and upper.  Here we partition the rows. We account for the fact that
     N may not divide evenly by the number of processors. */
  itype local_size = N / nprocs;
  // local row start
  itype ilower = local_size * myid;
  // local row end
  itype iupper = local_size*(myid+1);
  // last takes all

  if(myid == nprocs-1){
    iupper = N;
    local_size = N - local_size*(myid);
  }

  assert(local_size == (iupper-ilower));
  CSR *Alocal = CSRm::init(local_size, N, local_size*7, true, false, false, N, ilower);

  vtype values[7];
  itype cols[7];
  itype NNZ = 0;
  itype I = 0;

  for(itype i = ilower; i < iupper; i++, I++){
    itype nnz = 0;
    itype k=floor(i/(n*n));
      /* The left identity block:position i-n*n */
      if ((i-n*n)>=0){
        cols[nnz] = i-n*n;
        values[nnz] = -1.0;
        nnz++;
      }
      /* The left identity block:position i-n */
      if (i>=n+k*(n*n) && i < (k+1)*(n*n)){
        cols[nnz] = i-n;
        values[nnz] = -1.0;
        nnz++;
      }
      /* The left -1: position i-1 */
      if (i%n){
        cols[nnz] = i-1;
        values[nnz] = -1.0;
        nnz++;
      }
      /* Set the diagonal: position i */
      cols[nnz] = i;
      values[nnz] = 6.0;
      nnz++;
      /* The right -1: position i+1 */
      if ((i+1)%n){
        cols[nnz] = i+1;
        values[nnz] = -1.0;
        nnz++;
      }
      /* The right identity block:position i+n */
      if(i>=k*(n*n) && i < (k+1)*(n*n)-n){
        cols[nnz] = i+n;
        values[nnz] = -1.0;
        nnz++;
      }
      /* The right identity block:position i+n*n */
      if ((i+n*n)< N){
        cols[nnz] = i+n*n;
        values[nnz] = -1.0;
        nnz++;
      }
      if(I == 0){
        Alocal->row[0] = 0;
      }
      // set row index
      Alocal->row[I+1] = Alocal->row[I] + nnz;
      for(itype j=0; j<nnz; j++){
        Alocal->col[NNZ] = cols[j];
        Alocal->val[NNZ] = values[j];
        NNZ++;
      }
   }
   Alocal->nnz = NNZ;
   return Alocal;
}

static int stringCmp(const void *a, const void *b){
	return strcmp((const char *)a, (const char *)b);
}

int assignDeviceToProcess()
{
#ifdef MPI
    char host_name[MPI_MAX_PROCESSOR_NAME];
    char(*host_names)[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm nodeComm;
#else
    char host_name[20];
#endif
    int myrank;
    int gpu_per_node;
    int n, namelen, color, rank, nprocs;
    size_t bytes;
#ifdef MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Get_processor_name(host_name, &namelen);
    bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    host_names = (char(*)[MPI_MAX_PROCESSOR_NAME])malloc(bytes);
    strcpy(host_names[rank], host_name);
    for (n = 0; n < nprocs; n++){
        MPI_Bcast(&(host_names[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n,
                  MPI_COMM_WORLD);
    }
    qsort(host_names, nprocs, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);
    color = 0;
    for (n = 0; n < nprocs; n++){
        if (n > 0 && strcmp(host_names[n - 1], host_names[n]))
            color++;
        if (strcmp(host_name, host_names[n]) == 0)
            break;
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
    MPI_Comm_rank(nodeComm, &myrank);
    MPI_Comm_size(nodeComm, &gpu_per_node);
#else
    return 0;
#endif
    return myrank;
}

void swap(itype *xcol, itype *ycol, vtype *xval, vtype *yval) { 
    itype temp = *xcol; 
    vtype tempf = *xval;
    *xcol = *ycol; 
    *xval = *yval;
    *ycol = temp;
    *yval = tempf; 
}
 
void bubbleSort(itype arr[], vtype val[], itype n) { 
    itype i, j; 
    for (i = 0; i < n-1; i++)     
      
    // Last i elements are already in place 
    for (j = 0; j < n-i-1; j++) 
        if (arr[j] > arr[j+1]) 
            swap(&arr[j], &arr[j+1], &val[j], &val[j+1]); 
}
 
int internal_index(int gi, int gj, int gk, int nx, int ny, int nz, int P, int Q, int R){
    int i = gi % nx; // Position in x
    int j = gj % ny; // Position in y
    int k = gk % nz; // Position in z

    int p = gi / nx; // Position in x direction
    int q = gj / ny; // Position in y
    int r = gk / nz; // Position in z

    return (r*P*Q + q*P + p)*nx*ny*nz + (k*nx*ny + j*nx + i);
}

CSR* generateLocalLaplacian3D_mesh(itype nx, itype ny, itype nz, itype P, itype Q, itype R){
  _MPI_ENV;
  MPI_Comm NEWCOMM; 
  int dims[3] = {0, 0, 0};
  int periods[3] = {false, false, false};
  int coords[3] = {0, 0, 0};
  int my3id;
  // global number of rows
  itype N = nx * ny * nz * P * Q * R;

  itype local_size = nx * ny * nz;
  int nx_glob = nx*P;
  int ny_glob = ny*Q;
  int nz_glob = nz*R;

  int num_rows = nx_glob*ny_glob*nz_glob;
  int num_nonzeros = num_rows*7; // Ignoring any boundary, 7 nnz per row
  int num_substract = 0;

  num_substract += ny_glob*nz_glob;
  num_substract += ny_glob*nz_glob;
  num_substract += nx_glob*nz_glob;
  num_substract += nx_glob*nz_glob;
  num_substract += nx_glob*ny_glob;
  num_substract += nx_glob*ny_glob;

  num_nonzeros -= num_substract; //global

  dims[0] =P;
  dims[1] =Q;
  dims[2] =R;
  MPI_Dims_create(nprocs, 3, dims);

  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, false, &NEWCOMM);

  MPI_Comm_rank(NEWCOMM, &my3id);
  MPI_Cart_coords(NEWCOMM, my3id, 3, coords);

  int p = coords[0];
  int q = coords[1];
  int r = coords[2]; 
  itype ilower=(r*Q*P +q*P + p)*local_size;
  CSR *Alocal = CSRm::init(local_size, N, (local_size*7), true, false, false, N, ilower);
// alloc COO 
  itype * Arow = (itype*) malloc( sizeof(itype *) * (local_size*7) );
  itype * Acol = (itype*) malloc( sizeof(itype *) * (local_size*7) );
  vtype * Aval = (vtype*) malloc( sizeof(vtype *) * (local_size*7) );
  itype count = 0;
  itype nz_count = 0;
  itype nnz = 0;
  Alocal->row[0] = 0;

        for (int k = 0 ; k < nz ; k++)
        {
        for (int j = 0 ; j < ny ; j++)
        {
        for (int i = 0 ; i < nx ; i++)
        {
           int gi = p*nx + i;
           int gj = q*ny + j;
           int gk = r*nz + k;

           //Diagonal term
           Arow[nz_count] = count;
           Acol[nz_count] = count+ ilower;
           Aval[nz_count] = 6.;
           nnz++;
           nz_count++;

           // Given gi,gj, gk, find p,q,r, i_loc, j_loc, k_loc

           if ((i==0) && (p==0)) {}
            // do nothing 
           else
           {
              Arow[nz_count] = count;
              Acol[nz_count] = internal_index(gi-1,gj,gk,nx,ny,nz,P,Q,R);
              Aval[nz_count] = -1.;
              nnz++;
              nz_count++;
           }

           if ((i==nx-1) && (p==P-1)) {}
            // do nothing, no right neighbor
           else
           {
              Arow[nz_count] = count;
              Acol[nz_count] = internal_index(gi+1,gj,gk,nx,ny,nz,P,Q,R);
              Aval[nz_count] = -1.;
              nnz++;
              nz_count++;
           }

           if ((j==0) && (q==0)) {}
            // do nothing, no right neighbor
           else
           {
              Arow[nz_count] = count;
              Acol[nz_count] = internal_index(gi,gj-1,gk,nx,ny,nz,P,Q,R);
              Aval[nz_count] = -1.;
              nnz++;
              nz_count++;
           }

           if ((j==ny-1) && (q==Q-1)) {}
            // do nothing, no right neighbor
           else
           {
              Arow[nz_count] = count;
              Acol[nz_count] = internal_index(gi,gj+1,gk,nx,ny,nz,P,Q,R);
              Aval[nz_count] = -1.;
              nnz++;
              nz_count++;
           }

           if ((k==0) && (r==0)) {}
            // do nothing, no right neighbor
           else
           {
              Arow[nz_count] = count;
              Acol[nz_count] = internal_index(gi,gj,gk-1,nx,ny,nz,P,Q,R);
              Aval[nz_count] = -1.;
              nnz++;
              nz_count++;
           }

           if ((k==nz-1) && (r==R-1)) {}
            // do nothing, no right neighbor
           else
           {
              Arow[nz_count] = count;
              Acol[nz_count] = internal_index(gi,gj,gk+1,nx,ny,nz,P,Q,R);
              Aval[nz_count] = -1.;
              nnz++;
              nz_count++;
           }
           Alocal->row[count+1] = Alocal->row[count] + nnz;
           bubbleSort(&Acol[nz_count-nnz], &Aval[nz_count-nnz],nnz); 
           nnz = 0;
           count++;
           
        }
        }
        }
       
   Alocal->row[count] =  nz_count; //check if 
   for(itype j=0; j<nz_count; j++){
         Alocal->col[j] = Acol[j];
         Alocal->val[j] = Aval[j];
   }
   Alocal->nnz = nz_count;

#ifdef PRINT_COO3D
  FILE *fout = NULL;
  char fname[256];
  snprintf(fname, 256, "matrix-rank%d-pqr-%d-%d-%d.mtx", myid, p, q, r);
  fout = fopen(fname, "w+");
  if (fout == NULL) {
    fprintf(stderr, "in function %s: error opening %s\n", __func__, fname);
    exit(EXIT_FAILURE);
  }
  for (int i =0; i < nz_count; i++){
    fprintf(fout,"%d %d %lf\n",Arow[i]+(ilower*1), Acol[i], Aval[i] );     
  }
   fclose(fout);  
#endif   

   free(Arow);
   free(Acol);
   free(Aval);
   return Alocal;
}

#define LAP_N_PARAMS	6
int * read_laplacian_file(const char *file_name){
    FILE *fp=NULL;
    char buffer[BUFSIZE];
    const char *params_list[] = {"nx","ny","nz","P","Q","R"};
    int *lap_3d_parm;
	
    lap_3d_parm = (int*) Malloc(sizeof(int)*LAP_N_PARAMS);
	
    fp = fopen(file_name, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] - Laplacia file not found! (%s)\n", file_name);
        exit(EXIT_FAILURE);
    }
	
    while ( fgets(buffer, BUFSIZE, fp)){ /* READ a LINE */
	int buflen = 0;
	buflen = strlen(buffer);
       	if ( buffer[buflen-1] != '\n' && !feof(fp) ) { // Check that the buffer is big enough to read a line
            fprintf(stderr, "[ERROR] File %s. The line is too long, increase the BUFSIZE! Exit\n", file_name);
			exit(EXIT_FAILURE);
        }
	if ( buflen > 0 && buffer[0] != '#'){ // skip empty lines and comments
	    char opt[20];
	    int value, err;
	    err = sscanf(buffer, "%s = %d\n", opt, &value);
	    if (err != 2 || err == EOF){
		fprintf(stderr, "[ERROR] Error reading file %s.\n", file_name);
		exit(EXIT_FAILURE);
	    }
	    for(int i=0; i<LAP_N_PARAMS; i++){
		if ( strstr( opt, params_list[i] ) != NULL ) {
		    lap_3d_parm[i] = value;
		    break;
		}	
	    }
	}
    }
    fclose(fp);
    return lap_3d_parm;
}


#define USAGE "\nUsage: bcmg [--matrix <FILE_NAME> | --laplacian-3d <FILE_NAME> | --laplacian <SIZE>] --settings <FILE_NAME>\n\n"\
                   "\tYou can specify only one out of the three available options: --matrix, --laplacian-3d and --laplacian\n\n"\
	           "\t-m, --matrix <FILE_NAME>         Read the matrix from file <FILE_NAME>.\n"\
	           "\t-l, --laplacian-3d <FILE_NAME>   Read generation parameters from file <FILE_NAME>.\n"\
		   "\t-a, --laplacian <SIZE>           Generate a matrix whose size is <SIZE>^3.\n"\
	           "\t-s, --settings <FILE_NAME>       Read settings from file <FILE_NAME>.\n\n"


int main(int argc, char **argv){

  enum opts {MTX, LAP_3D, LAP, NONE} opt = NONE;
  char *mtx_file = NULL;
  char *lap_3d_file = NULL;
  char *settings_file = NULL;
  char ch;
  itype n = 0;

  static struct option long_options[] ={
	{"matrix", required_argument, NULL, 'm'},
	{"laplacian-3d", required_argument, NULL, 'l'},
	{"laplacian", required_argument, NULL, 'a'},
	{"settings", required_argument, NULL, 's'},
	{"help", no_argument, NULL, 'h'},
	{NULL, 0, NULL, 0}
	};

  while ((ch = getopt_long(argc, argv, "m:l:a:s:h", long_options, NULL)) != -1){
    switch (ch){
      case 'm':
        mtx_file = strdup(optarg);
	opt = MTX;
	break;		    
      case 'l':
        lap_3d_file = strdup(optarg);
	opt = LAP_3D;
	break;		    
      case 'a':
        n = atoi(optarg);
	opt = LAP;
	break;		    
      case 's':
        settings_file = strdup(optarg);
	break;		    
      case 'h':
      default:
        printf(USAGE);
	exit(EXIT_FAILURE);
    }
 }

 if ( opt == NONE || settings_file == NULL){
   printf(USAGE);
   exit(EXIT_FAILURE);
 }

  // setup AMG:
  int myid, nprocs, device_id;
  StartMpi(&myid, &nprocs, &argc, &argv);

  device_id = assignDeviceToProcess();
  //SetDevice
  cudaSetDevice(device_id);
  handles *h = Handles::init();

  CSR *Alocal;
  vector<vtype> *rhs;
  params p = AMG::Params::initFromFile(settings_file);

  if ( opt == MTX ){ // The master reads the matrix and distributes it.
    CSR *Alocal_master = NULL;
    if(ISMASTER){
      Alocal_master = readMatrixFromFile(mtx_file,0,false);
    }
    Alocal = split_MatrixMPI(Alocal_master);
    if (ISMASTER){
       CSRm::free(Alocal_master);
    }
  }else if ( opt == LAP_3D || opt == LAP){
    CSR *Alocal_host = NULL;
    if( opt == LAP_3D){ // Each process generates its portion of the matrix.
      enum lap_params {nx=0, ny=1, nz=2, P=3, Q=4, R=5};
      int *parms;
      parms = read_laplacian_file(lap_3d_file);
      if (nprocs != (parms[P]*parms[Q]*parms[R]) ) {
        fprintf(stderr,"Nproc must be equal to P*Q*R\n");  
        exit(EXIT_FAILURE);
      }
      Alocal_host = generateLocalLaplacian3D_mesh(parms[nx], parms[ny], parms[nz], parms[P], parms[Q], parms[R]);
      free(parms);
    }else if( opt == LAP){
      Alocal_host = generateLocalLaplacian3D(n);
    }
    Alocal = CSRm::copyToDevice(Alocal_host);
    char fname[256];
    snprintf(fname, 256, "Alocal-%d.mtx", myid);
    CSRMatrixPrintMM(Alocal_host, fname);
    CSRm::free(Alocal_host);
  }

  rhs = Vector::init<vtype>(Alocal->n, true, true);
  Vector::fillWithValue(rhs, 1.);

  if(ISMASTER){
    printf("STARTING...:\n");
  }
  vector<vtype> *sol = solve(Alocal, rhs, p);
  Vector::free(sol);
  if(ISMASTER){
    printf("DONE!\n");
  }
  MPI_Finalize();
  return 0;
}


