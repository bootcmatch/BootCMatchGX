#include "basic_kernel/matrix/matrixIO.h"
#include "utility/cudamacro.h"

CSR* readMatrixFromFile(const char *matrix_path, int m_type, bool loadOnDevice){

  CSR* A_host = NULL;

  switch(m_type){
    case 0:
      A_host = readMTXDouble(matrix_path);
      break;
    case 1:
      A_host = readMTX2Double(matrix_path);
      break;
    default:
      std::cout << "You need to specify an input matrix type with the argument -F/--inputype\n";
      exit(1);
  }

  assert(A_host != NULL);

  if(loadOnDevice){
    CSR *A = CSRm::copyToDevice(A_host);
    CSRm::free(A_host);
    return A;
  }

  return A_host;
}

// stolen from BootCMatch CPU
CSR* readMTXDouble(const char *file_name){


    FILE    *fp;
    char    banner[64], mtx[64], crd[64], data_type[64], storage_scheme[64];
    char    buffer[BUFSIZE+1];
    double  *matrix_value, *matrix_data, val;
    int     *matrix_cooi, *matrix_i;
    int     *matrix_cooj, *matrix_j;
    int      num_rows, num_cols, ri, cj;
    int      num_nonzeros, fr_nonzeros, allc_nonzeros;
    int      max_col = 0, is_general=0, is_symmetric=0;

    int      file_base = 1;

    int      i, j, k, k0, iad;
    double   x;

    /*----------------------------------------------------------
     * Read in the data (matrix in MM format)
     *----------------------------------------------------------*/

    fp = fopen(file_name, "r");
  	if(fp == NULL){
  		printf("FILE NOT FOUND!\n");
  		exit(1);
  	}

    fscanf(fp, "%s %s %s %s %s\n", banner, mtx, crd, data_type, storage_scheme);
    fgets(buffer,BUFSIZE,fp);
    //for ( ; buffer[0]=='\%';  fgets(buffer,BUFSIZE,fp) );
    for ( ; buffer[0]=='%';  fgets(buffer,BUFSIZE,fp) );

    sscanf(buffer, "%d %d %d", &num_rows, &num_cols, &fr_nonzeros);

    if (strcmp(data_type,"real") !=0) {
      fprintf(stderr,"Error: we only read real matrices, not '%s'\n",data_type);
      fclose(fp);
      return(NULL);
    }

    if (strcmp(storage_scheme,"general")==0) {
      allc_nonzeros = fr_nonzeros;
      is_general=1;
    } else if (strcmp(storage_scheme,"symmetric")==0) {
      allc_nonzeros = 2*fr_nonzeros;
      is_symmetric=1;
    } else {
      fprintf(stderr,"Error: unhandled storage scheme '%s'\n",storage_scheme);
      fclose(fp);
      return(NULL);
    }

    matrix_cooi = (int *) calloc(allc_nonzeros, sizeof(int));
    matrix_cooj = (int *) calloc(allc_nonzeros, sizeof(int));
    matrix_value = (double *) calloc(allc_nonzeros, sizeof(double));
    if (is_general) {
      num_nonzeros = fr_nonzeros;
      for (j = 0; j < fr_nonzeros; j++)
        {
  	if (fgets(buffer,BUFSIZE,fp) != NULL) {
  	  sscanf(buffer, "%d %d %le", &matrix_cooi[j], &matrix_cooj[j], &matrix_value[j]);
  	  matrix_cooi[j] -= file_base;
  	  matrix_cooj[j] -= file_base;
  	  if (matrix_cooj[j] > max_col)
  	    {
  	      max_col = matrix_cooj[j];
  	    }
  	} else {
  	  fprintf(stderr,"Reading from MatrixMarket file failed\n");
  	  fprintf(stderr,"Error while trying to read record %d of %d from file %s\n",
  		  j,fr_nonzeros,file_name);
  	  exit(-1);
  	}
        }
    } else if (is_symmetric) {
      k = 0;
      for (j = 0; j < fr_nonzeros; j++)   {
        if (fgets(buffer,BUFSIZE,fp) != NULL) {
  	sscanf(buffer, "%d %d %le", &ri, &cj, &val);
  	ri -= file_base;
  	cj -= file_base;
  	if (cj > max_col)
  	  max_col = cj;
  	matrix_cooi[k]  = ri;
  	matrix_cooj[k]  = cj;
  	matrix_value[k] = val;
  	k++;
  	if (ri != cj) {
  	  matrix_cooi[k]  = cj;
  	  matrix_cooj[k]  = ri;
  	  matrix_value[k] = val;
  	  k++;
  	}
        } else {
  	fprintf(stderr,"Reading from MatrixMarket file failed\n");
  	fprintf(stderr,"Error while trying to read record %d of %d from file %s\n",
  		j,fr_nonzeros,file_name);
  	fclose(fp);
  	return(NULL);
        }
      }
      num_nonzeros = k;
    } else {
      fprintf(stderr,"Internal error: neither symmetric nor general ? \n");
      fclose(fp);
      return(NULL);
    }
    /*----------------------------------------------------------
     * Transform matrix from COO to CSR format
     *----------------------------------------------------------*/

    matrix_i = (int *) calloc(num_rows+1, sizeof(int));

    /* determine row lenght */
    for (j=0; j<num_nonzeros; j++) {
      if ((0<=matrix_cooi[j])&&(matrix_cooi[j]<num_rows)){
        matrix_i[matrix_cooi[j]]=matrix_i[matrix_cooi[j]]+1;
      } else {
        fprintf(stderr,"Wrong row index %d at position %d\n",matrix_cooi[j],j);
      }
    }

    /* starting position of each row */
    k=0;
    for(j=0; j<= num_rows; j++)
      {
        k0=matrix_i[j];
        matrix_i[j]=k;
        k=k+k0;
      }
    matrix_j = (int *) calloc(num_nonzeros, sizeof(int));
    matrix_data = (double *) calloc(num_nonzeros, sizeof(double));

    /* go through the structure once more. Fill in output matrix */
    for(k=0; k<num_nonzeros; k++)
      {
        i=matrix_cooi[k];
        j=matrix_cooj[k];
        x=matrix_value[k];
        iad=matrix_i[i];
        matrix_data[iad]=x;
        matrix_j[iad]=j;
        matrix_i[i]=iad+1;
      }
    /* shift back matrix_i */
    for(j=num_rows-1; j>=0; j--) matrix_i[j+1]=matrix_i[j];
    matrix_i[0]=0;

    assert(num_rows > 0 && num_cols > 0 && num_nonzeros >= 0);
    CSR *A = CSRm::init(num_rows, num_cols, num_nonzeros, false, false, false, num_rows);
    A->val = matrix_data;
    A->row = matrix_i;
    A->col = matrix_j;

    free(matrix_cooi);
    free(matrix_cooj);
    free(matrix_value);
    fclose(fp);

    return A;
  }

CSR* readMTX2Double(const char *file_name){

    FILE    *fp;

    double  *matrix_value, *matrix_data;
    int     *matrix_cooi, *matrix_i;
    int     *matrix_cooj, *matrix_j;
    int      num_rows;
    int      num_nonzeros;
    int      max_col = 0;

    int      file_base = 1;

    int      i, j, k, k0, iad;
    double   x;

    /*----------------------------------------------------------
     * Read in the data (matrix in COO format)
     *----------------------------------------------------------*/

    fp = fopen(file_name, "r");
    if(fp == NULL){
  		printf("FILE NOT FOUND!\n");
  		exit(1);
  	}

    fscanf(fp, "%d", &num_rows);
    fscanf(fp, "%d", &num_nonzeros);

    matrix_cooi = (int *) calloc(num_nonzeros, sizeof(int));
    for (j = 0; j < num_nonzeros; j++)
      {
        fscanf(fp, "%d", &matrix_cooi[j]);
        matrix_cooi[j] -= file_base;
      }
    matrix_cooj = (int *) calloc(num_nonzeros, sizeof(int));
    for (j = 0; j < num_nonzeros; j++)
      {
        fscanf(fp, "%d", &matrix_cooj[j]);
        matrix_cooj[j] -= file_base;
        if (matrix_cooj[j] > max_col)
  	{
  	  max_col = matrix_cooj[j];
  	}
      }
    matrix_value = (double *) calloc(num_nonzeros, sizeof(double));
    for (j = 0; j < num_nonzeros; j++) fscanf(fp, "%le", &matrix_value[j]);

    /*----------------------------------------------------------
     * Transform matrix from COO to CSR format
     *----------------------------------------------------------*/

    matrix_i = (int *) calloc(num_rows+1, sizeof(int));

    /* determine row lenght */
    for (j=0; j<num_nonzeros; j++) matrix_i[matrix_cooi[j]]=matrix_i[matrix_cooi[j]]+1;

    /* starting position of each row */
    k=0;
    for(j=0; j<= num_rows; j++)
      {
        k0=matrix_i[j];
        matrix_i[j]=k;
        k=k+k0;
      }
    matrix_j = (int *) calloc(num_nonzeros, sizeof(int));
    matrix_data = (double *) calloc(num_nonzeros, sizeof(double));

    /* go through the structure once more. Fill in output matrix */
    for(k=0; k<num_nonzeros; k++)
      {
        i=matrix_cooi[k];
        j=matrix_cooj[k];
        x=matrix_value[k];
        iad=matrix_i[i];
        matrix_data[iad]=x;
        matrix_j[iad]=j;
        matrix_i[i]=iad+1;
      }
    /* shift back matrix_i */
    for(j=num_rows-1; j>=0; j--) matrix_i[j+1]=matrix_i[j];
    matrix_i[0]=0;

    assert(num_rows > 0 && num_rows > 0 && num_nonzeros >= 0);
    CSR *A = CSRm::init(num_rows, num_rows, num_nonzeros, false, false, false, num_rows);
    A->val = matrix_data;
    A->row = matrix_i;
    A->col = matrix_j;

    free(matrix_cooi);
    free(matrix_cooj);
    free(matrix_value);
    fclose(fp);

    return A;
  }

void CSRMatrixPrintMM(CSR *A_, const char *file_name){
  PUSH_RANGE(__func__,2)

  CSR *A = NULL;
  if(A_->on_the_device)
    A = CSRm::copyToHost(A_);
  else
    A = A_;

  FILE    *fp;

  double  *matrix_data;
  int     *matrix_i;
  int     *matrix_j;
  int      num_rows;
  int      num_cols, nnz;

  int      file_base = 1;

  int      i,j;


  matrix_data = A->val;
  matrix_i    = A->row;
  matrix_j    = A->col;
  num_rows    = A->n;
  num_cols    = A->m;
  nnz         = A->nnz;

  fp = fopen(file_name, "w");
  fprintf(fp,"%s\n","%%MatrixMarket matrix coordinate real general");

  fprintf(fp, "%d  %d %d \n", num_rows, num_cols, nnz);

  for (i = 0; i < num_rows; i++)     {
    for (j=matrix_i[i]; j<matrix_i[i+1]; j++) {
      fprintf(fp, "%d   %d  %lg\n", i+file_base,matrix_j[j] + file_base, matrix_data[j]);
    }
  }
  fclose(fp);
  
  POP_RANGE
}
