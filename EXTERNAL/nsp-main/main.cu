#include <CSR.h>
#include <handles.h>
#include <matrixIO.h>
#include <scalar.cu>
#include <distribute.cu>
#include <unistd.h>
#include <stdlib.h>
#include <spspmpi.h>

#define ASSIGNGPU

#define MINARG 3

int xsize;
double *xvalstat=NULL;
#if defined(COMPUTECOLINDICES)
itype *hProw, *hPcol, hPnnz, hPn, hPshift;
#endif  
void device_query(){

cudaDeviceProp deviceProp;
int count;
int driverVersion = 0, runtimeVersion = 0;
cudaGetDeviceCount( &count ) ;

    for (int i=0; i< count; i++) {
      // Console log
      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);
      printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
      driverVersion / 1000, (driverVersion % 100) / 10,
      runtimeVersion / 1000, (runtimeVersion % 100) / 10);
      printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
      deviceProp.major, deviceProp.minor);

      printf("\n\n");


        cudaGetDeviceProperties( &deviceProp, i ) ;
        printf( " --- General Information for device %d ---\n", i );
        printf( "Name: %s\n", deviceProp.name );
        printf( "Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor );
        printf( "Clock rate: %d\n", deviceProp.clockRate );
        printf( "Device copy overlap: " );
        if (deviceProp.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );
        printf( "Kernel execition timeout : " );
        if (deviceProp.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( " --- Memory Information for device %d ---\n", i );
        printf( "Total global mem: %ld\n", deviceProp.totalGlobalMem );
        printf( "Total constant Mem: %ld\n", deviceProp.totalConstMem );
        printf( "Max mem pitch: %ld\n", deviceProp.memPitch );
        printf( "Texture Alignment: %ld\n", deviceProp.textureAlignment );
        printf( " --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count: %d\n",
                deviceProp.multiProcessorCount );
        printf( "Shared mem per mp: %ld\n", deviceProp.sharedMemPerMultiprocessor );
        printf( "Registers per mp: %d\n", deviceProp.regsPerBlock );
        printf( "Threads in warp: %d\n", deviceProp.warpSize );
        printf( "Max threads per block: %d\n",
                deviceProp.maxThreadsPerBlock );
        printf( "Max thread dimensions: (%d, %d, %d)\n",
                deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
                deviceProp.maxThreadsDim[2] );
        printf( "Max grid dimensions: (%d, %d, %d)\n",
                deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
                deviceProp.maxGridSize[2] );
        printf( "\n" );

    }

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

void check_and_fix_order(CSR * A) {

  itype * Arow = A->row;
  itype * Acol = A->col;
  vtype * Aval = A->val;
  itype prev;
  int wrongo;
  for(int i=0; i<A->n; i++){
      wrongo=0;
      prev=A->col[Arow[i]];
      for (int j = Arow[i]+1; j< Arow[i+1]; j++) {
           if (A->col[j] < prev) {
	       wrongo=1;
	       break;
           } else {
	       prev=A->col[j];
	   }
      }
      if(wrongo) {
           bubbleSort(&Acol[Arow[i]], &Aval[Arow[i]],(Arow[i+1]-Arow[i])); 
      }
  }

     
}

template <typename mtype,typename otype>
void write_binary_matrix(const char *filename, mtype *iat, mtype *ja, double *coef,
                         mtype nr, mtype nc, mtype nt);


int main(int argc, char *argv[]){

  device_query();

  int myid, nprocs;

  int printout=0, printinfo=0;
  bool ascii,header;
  //  il prodotto e' R(AP); le matrici sono tutte su device
  if(argc<MINARG) {
        fprintf(stderr,"Usage: %s FileMatrixA FileMatrixP [printout] [printinfo]\n",argv[0]);
        exit(1);
  }
  argc-=MINARG;
  if(argc>0) {
    ascii     = atoi(argv[3]);
    header    = atoi(argv[4]);
    printout  = atoi(argv[5]);
    printinfo = atoi(argv[6]);
    argc--;
  }

#if !defined(ADDCOLSHIFT) && !defined(USESHRINKEDMATRIX)
  fprintf(stderr,"Either ADDCOLSHIFT or USESHRINKEDMATRIX must be defined\n");
  exit(1);
#endif
  StartMpi(&myid, &nprocs,&argc,&argv);
#if defined(ASSIGNGPU)
   assignDeviceToProcess(myid);
#endif

  handles *h = Handles::init();

  // load matrix to device
  CSR *Alocal = load_MatrixMPI<long int,double,long int>(argv[1], ascii, header);
  CSR *Plocal = load_MatrixMPI<long int,double,long int>(argv[2], ascii, header);
  
#if defined(COMPUTECOLINDICES)
  
  cudaError_t err;
  itype *hArow, *hAcol;
  itype *hCcol;
  hArow=(itype *)malloc((Alocal->n + 1) * sizeof(itype));
  if(hArow==NULL) {
     fprintf(stderr,"Could not get %d bytes for Arow\n",(Alocal->n + 1) * sizeof(itype));
     exit(1);
  }
  hAcol=(itype *)malloc(Alocal->nnz * sizeof(itype));
  if(hAcol==NULL) {
     fprintf(stderr,"Could not get %d bytes for Acol\n",Alocal->nnz * sizeof(itype));
     exit(1);
  }
  err = cudaMemcpy(hArow, Alocal->row, (Alocal->n + 1) * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
  err = cudaMemcpy(hAcol, Alocal->col, Alocal->nnz * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);

  if(nprocs==1) {
    hProw=(itype *)malloc((Plocal->n + 1) * sizeof(itype));
    if(hProw==NULL) {
     fprintf(stderr,"Could not get %d bytes for Prow\n",(Plocal->n + 1) * sizeof(itype));
     exit(1);
    }
    hPcol=(itype *)malloc(Plocal->nnz * sizeof(itype));
    if(hPcol==NULL) {
       fprintf(stderr,"Could not get %d bytes for Pcol\n",Plocal->nnz * sizeof(itype));
       exit(1);
     }
     err = cudaMemcpy(hProw, Plocal->row, (Plocal->n + 1) * sizeof(itype), cudaMemcpyDeviceToHost);
     CHECK_DEVICE(err);
     err = cudaMemcpy(hPcol, Plocal->col, Plocal->nnz * sizeof(itype), cudaMemcpyDeviceToHost);
     CHECK_DEVICE(err);
     hPnnz=Plocal->nnz,
     hPn=Plocal->n,
     hPshift=Plocal->row_shift;      
  }
#endif

  Alocal->col_shifted=-Alocal->row_shift;
  Plocal->col_shifted=-Plocal->row_shift;

  if(printinfo){
    CSRm::printInfo(Alocal);
    CSRm::printInfo(Plocal);
  }

  double TOT_TIMEM, loc_time;
  stype tot_nnz,tot_n;
  double l2n_loc = 0.,l2n = 0.;
  
  MPI_Barrier(MPI_COMM_WORLD);
  cudaDeviceSynchronize();
  if(ISMASTER)TOT_TIMEM = MPI_Wtime();
  CSR* APlocal = nsparseMGPU_version1(h, Alocal, Plocal, loc_time);
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  if(ISMASTER) TOT_TIMEM = MPI_Wtime() - TOT_TIMEM;



  if(APlocal->on_the_device) {
    CSR* h_APlocal = CSRm::copyToHost(APlocal);
    CSRm::free(APlocal);
    APlocal = h_APlocal;
  }

  MPI_Reduce(&(APlocal->nnz), &tot_nnz, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(APlocal->n),   &tot_n,   1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  for (long int i=0;i<APlocal->nnz;i++) l2n_loc += APlocal->val[i]*APlocal->val[i];
  MPI_Reduce(&(l2n_loc), &l2n, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(ISMASTER) l2n = sqrt(l2n);
  if(ISMASTER) printf("TOTAL_TIME: %f\t\tlocal time: %f\t\tnnz = %lu\t\tl2norm: %f\n", TOT_TIMEM, loc_time, tot_nnz,l2n);

  if(ISMASTER){
    FILE *fid = fopen("outtime.txt", "a");
    fprintf(fid,"%15lu %15lu %15lu ",tot_n,(gstype)APlocal->m,tot_nnz);
    fprintf(fid,"%10f %10f %50f\n",loc_time,TOT_TIMEM,l2n);
    fclose(fid);
  }
  
  CSRm::free(APlocal);
  MPI_Finalize();
  return 0;
  
#if defined(COMPUTECOLINDICES)
  itype minhPcol=hPcol[0], maxhPcol=hPcol[0];
  for(int i=1; i<hPnnz; i++) {
      if(hPcol[i]<minhPcol) { minhPcol=hPcol[i]; }
      if(hPcol[i]>maxhPcol) { maxhPcol=hPcol[i]; }      
  }
  fprintf(stderr,"task %d, minhPcol=%d, maxhPcol=%d, Pnnz=%d\n",myid,minhPcol, maxhPcol,hPnnz);  
  
  itype *hPp2C=(itype *)malloc(((maxhPcol-minhPcol)+2) * sizeof(itype));
  if(hPp2C==NULL) {
     fprintf(stderr,"Could not get %d bytes for hPp2C\n",(maxhPcol-minhPcol) * sizeof(itype));
     exit(1);
  }
  itype *hPp2Cshifted=&hPp2C[1];
  itype *hPp2R=(itype *)malloc(hPnnz * sizeof(itype));
  if(hPp2R==NULL) {
     fprintf(stderr,"Could not get %d bytes for hPp2R\n",hPnnz * sizeof(itype));
     exit(1);
  }
  for(int i=0; i<((maxhPcol-minhPcol)+2); i++) {
      hPp2C[i]=0;
  }
  for(int i=0; i<hPnnz; i++) {
      hPp2Cshifted[hPcol[i]-minhPcol]++;
  }
  int savecnt=hPp2Cshifted[0], tmpcnt;
  hPp2Cshifted[0]=0;
  for(int i=1; i<=(maxhPcol-minhPcol); i++) {
      tmpcnt=hPp2Cshifted[i];
      hPp2Cshifted[i]=hPp2Cshifted[i-1]+savecnt;
      savecnt=tmpcnt;
  }
  for(int i=0; i<hPn; i++) {
  	  int c;
  	  for(int j=hProw[i]; j<hProw[i+1]; j++) {
	  	  c=hPp2Cshifted[(hPcol[j]-minhPcol)];
		  if(c>=hPnnz) {
		     fprintf(stderr,"task %d: unexpected value %d max should be %d hpcol=%d, \n",myid,c,hPnnz,hPcol[j]);
		     continue;
		  }
	      	  hPp2R[c]=i;
		  hPp2Cshifted[(hPcol[j]-minhPcol)]++;
	  }
  }
#if 0
  if(myid==0) {
     for(int j=0; j<((maxhPcol-minhPcol)+1); j++) {
            fprintf(stderr,"col %d: ",j);
        for(int l=hPp2C[j]; (l<hPp2C[j+1]); l++) {
            fprintf(stderr,"%d ",hPp2R[l]+hPshift);
	}
	fprintf(stderr,"\n");
     }
  }
  if(myid==0) {
     for(int j=0; j<hPn; j++) {
        fprintf(stderr,"row %d: ",j);
        for(int l=hProw[j]; (l<hProw[j+1]); l++) {
            fprintf(stderr,"%d ",hPcol[l]+hPshift);
	}
	fprintf(stderr,"\n");
     }
  }
#endif
  hCcol=(itype *)malloc(APlocal->nnz * sizeof(itype));
  memset(hCcol,0,APlocal->nnz * sizeof(itype));
  if(hCcol==NULL) {
     fprintf(stderr,"Could not get %d bytes for Ccol\n",APlocal->nnz * sizeof(itype));
     exit(1);
  }
  int found, m=0, mold=0;
  for(int j=0; j<(maxhPcol-minhPcol); j++) {
	       if(hPp2C[j+1]<hPp2C[j]) { hPp2C[j+1]=hPp2C[j]; }
  }
  for(int i=0; i< Alocal->n; i++) {
   mold=m;

   for(int j=0; j<((maxhPcol-minhPcol)+1); j++) {
      found=0;
      for(int k=hArow[i]; (k<hArow[i+1]) && !found; k++) {
        for(int l=hPp2C[j]; (l<hPp2C[j+1]) && !found; l++) {
           if((hPp2R[l]+hPshift)==(hAcol[k]+Alocal->row_shift)) {
	      if(m>=APlocal->nnz) {
	        fprintf(stderr,"something wrong %d %d %d %d\n",myid,hPshift,Alocal->row_shift,minhPcol); exit(1);
	      }
	      found=1; hCcol[m]=j+minhPcol; m++;
	   }
        }	   
      }
   }
   for(int k=mold; k<(m-1); k++) {
      	  if(hCcol[k]>=hCcol[k+1]) {
	     fprintf(stderr,"unexpected col %d %d on row %d\n",hCcol[k],hCcol[k+1],i);
	     exit(1);
	  }
   }
  }
#endif  

  CSRtype<stype,gstype,itype>* AP = NULL;
  if(nprocs > 1){
    // Join local results (ONLY THE MASTER GETS THE MATRIX!!!)
    CSR* h_APlocal = CSRm::copyToHost(APlocal);
    #if defined(COMPUTECOLINDICES)    
        free(h_APlocal->col);
        h_APlocal->col=hCcol;
    #endif    
    // AP = join_MatrixMPI(h_APlocal);
    AP = join_MatrixMPI<stype,gstype,itype>(h_APlocal);
    CSRm::free(APlocal);
    CSRm::free(h_APlocal);
  } else {
    #if defined(COMPUTECOLINDICES)  
        AP = CSRm::copyToHost(APlocal);
        free(AP->col);
        AP->col=hCcol;
    #endif
    AP=APlocal;
  }
    //art check correctness
      double l2norm = 0.0;
      if(nprocs == 1 && ISMASTER){
        double *C_coef = (double*) malloc( AP->nnz * sizeof(double));
        itype  *C_row  = (itype *) malloc( (AP->n+1) * sizeof(itype));
        itype  *C_col  = (itype *) malloc( AP->nnz * sizeof(itype));
        cudaMemcpy( C_coef, AP->val,  AP->nnz * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy( C_col,  AP->col,  AP->nnz * sizeof(itype ), cudaMemcpyDeviceToHost);
        cudaMemcpy( C_row,  AP->row, (AP->n+1)* sizeof(itype ), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        for (long int i=0;i<AP->nnz;i++) l2norm += C_coef[i]*C_coef[i];
        l2norm = sqrt(l2norm);
        printf("\n\n\tl2norm of C = %f, nterms = %lu\n\n",l2norm,AP->nnz);

        printf("print output matrix:\n");
        for (int i = 0; i < 5; i++)                     printf("\t%lu %lu %f\n",(unsigned long)C_row[i],(unsigned long)C_col[i],C_coef[i]);
        printf("\n");
        for (unsigned long i = AP->n-5; i < AP->n; i++) printf("\t%lu %lu %f\n",(unsigned long)C_row[i],(unsigned long)C_col[i],C_coef[i]);
        
        free(C_coef);
        free(C_col);
        free(C_row);
      }else{
          if(ISMASTER){
            // double l2norm = 0.0;
            for (long int i=0;i<AP->nnz;i++) l2norm += AP->val[i]*AP->val[i];
            l2norm = sqrt(l2norm);
            printf("\n\n\tl2norm of C = %f, nterms = %lu\n\n",l2norm,AP->nnz); 

            printf("print output matrix:\n");
            for (int i = 0; i < 5; i++)                     printf("\t%lu %lu %f\n",(unsigned long)AP->row[i],(unsigned long)AP->col[i],AP->val[i]);
            printf("\n");
            for (unsigned long i = AP->n-5; i < AP->n; i++) printf("\t%lu %lu %f\n",(unsigned long)AP->row[i],(unsigned long)AP->col[i],AP->val[i]);
          }      
      }

      char str[100]="";
      if(ISMASTER && printout){
         int k = strlen(argv[1]);
         while (*(argv[1]+k) != '/') k--;
         for (int i = 0; i < k+1; i++){
            str[i]= *(argv[1]+i);
         }
         char A_name[100] = "";
         char B_name[100] = "";
         int l = k+1; int p = k+1; k = k+1;
         char MATA_file[100]="";
         char MATB_file[100]="";
         strcpy(MATA_file, argv[1]); strcpy(MATB_file, argv[2]);
         while (MATA_file[l] != '.') {A_name[l-k] = MATA_file[l];l++;}
         while (MATB_file[p] != '.') {B_name[p-k] = MATB_file[p];p++;}
         strcat(str,A_name);
         strcat(str,B_name);
      }

  //art dump info
      if(ISMASTER){
        FILE *fid = fopen("outtime.txt", "a");
        fprintf(fid,"%15lu %15lu %15lu ",(gstype)AP->n,(gstype)AP->m,(gstype)AP->nnz);
        fprintf(fid,"%50f ",l2norm);
        fprintf(fid,"%f\n",TOT_TIMEM);
        fclose(fid);
      }

  if(ISMASTER){
    
   //  if (printout) {
   //    strcat(str,".bin");
   //    CSR *AP_h = NULL;
   //    if(AP->on_the_device)
   //      AP_h = CSRm::copyToHost(AP);
   //    else
   //      AP_h = AP;
   //    write_binary_matrix<itype,itype>(str,
   //                                     AP_h->row,AP_h->col,AP_h->val,
   //                                     AP_h->n,  AP_h->m,  AP_h->nnz);
   //     printf("output matrix name:\n %s\n",str);
   //  }
    // if(printout) CSRMatrixPrintMM(AP, "AP.mtx");
    if(printout) {
      strcat(str,".mtx");
      CSRMatrixPrintMM(AP, str);
      printf("output matrix name:\n %s\n",str);
    }
    CSRm::free(AP);
  }
  
  MPI_Finalize();
  return 0;
}




template <typename mtype,typename otype>
void write_binary_matrix(const char *filename, mtype *iat, mtype *ja, double *coef,
                         mtype nr, mtype nc, mtype nt){
  
  struct hdr { otype nr,nc; long int nt; } hdr_var;
  struct row { otype ii,jj; double   aa; } *row_var = (struct row*) malloc(nt * sizeof(struct row));
  if (row_var==NULL) printf("ERROR malloc row_var \n");

  FILE *fid = fopen(filename, "wb");
  if (!fid) {
    printf("FILE NOT FOUND!\n");
    exit(1);
  }

  hdr_var.nr = nr;
  hdr_var.nc = nc;
  hdr_var.nt = nt;

  printf("WRITING rows %d columns %d nterm %ld\n",hdr_var.nr,hdr_var.nc,hdr_var.nt);

  if (!fwrite(&hdr_var,sizeof(struct hdr),1,fid)){
     fclose(fid);
     exit(2);
  }

  printf("WRITING header is done\n");

  for (mtype i = 0; i < nr; i++) { 
    for (mtype j = iat[i]; j < iat[i+1]; j++) {
      row_var[j].ii = (otype)i + 1;
      row_var[j].jj = (otype)  ja[j]+1;
      row_var[j].aa =        coef[j];
    }
  }

  if (!fwrite(row_var,sizeof(struct row),nt,fid)){
     fclose(fid);
     exit(3);
  }

  fclose(fid);
  free(row_var);

  printf("WRITING is done to file %s\n",filename);
}