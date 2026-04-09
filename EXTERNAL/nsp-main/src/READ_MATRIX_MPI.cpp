#define MAX_LINE 1024
#define white_space(c) ((c) == ' ' || (c) == '\t')
#define valid_digit(c) ((c) >= '0' && (c) <= '9')

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

int scan_string(char *line,int *nr,int *nc, int *nt) {
    return sscanf(line, "%d %d %ld", nr, nc, nt);
}
int scan_string(char *line,int *nr,int *nt) {
    return sscanf(line, "%d %ld", nr, nt);
}
int scan_string(char *line,long int *nr,long int *nc,long int *nt) {
    return sscanf(line, "%ld %ld %ld", nr, nc, nt);
}
int scan_string(char *line,long int *nr,long int *nt) {
    return sscanf(line, "%ld %ld", nr, nt);
}

double fast_atof (const char *p){
   int frac;
   double sign, value, scale;

   // Skip leading white space, if any.
   while (white_space(*p) ) p++;

   // Get sign, if any.
   sign = 1.0;
   if (*p == '-') {
      sign = -1.0;
      p++;
   } else if (*p == '+') p++;

   // Get digits before decimal point or exponent, if any.
   for (value = 0.0; valid_digit(*p); p++) {
      value = value * 10.0 + (*p - '0');
   }

   // Get digits after decimal point, if any.
   if (*p == '.') {
      double pow10 = 10.0;
      p++;
      while (valid_digit(*p)) {
         value += (*p - '0') / pow10;
         pow10 *= 10.0;
         p++;
      }
   }

   // Handle exponent, if any.
   frac = 0;
   scale = 1.0;
   if ((*p == 'e') || (*p == 'E')) {
      unsigned int expon;

      // Get sign of exponent, if any.
      p++;
      if (*p == '-') {
         frac = 1;
         p++;
      } else if (*p == '+') {
         p++;
      }

      // Get digits of exponent, if any.
      for (expon = 0; valid_digit(*p); p++) {
         expon = expon * 10 + (*p - '0');
      }
      if (expon > 308) expon = 308;

      // Calculate scaling factor.
      while (expon >= 50) { scale *= 1E50; expon -= 50; }
      while (expon >=  8) { scale *= 1E8;  expon -=  8; }
      while (expon >   0) { scale *= 10.0; expon -=  1; }
   }

   // Return signed and scaled floating point result.
   return sign * (frac ? (value / scale) : (value * scale));
}

template <typename Ti, typename Tie>
Tie binsearch_iat_str(MPI_File fh, int myrank, int nprocs, Tie nt, Tie x, int header_size, int row_size){
  char *val = (char*)malloc( (row_size+1)*sizeof(char));
  MPI_Offset mid;
  MPI_Offset low  = 0;
  MPI_Offset high = nt;
  Tie vv;
  while(low<high) {
    mid=(high+low)/2;
    CHECK_MPI(MPI_File_read_at(fh, header_size+(MPI_Offset)mid*row_size, val, row_size, MPI_CHAR, MPI_STATUS_IGNORE));
    if (typeid(Tie) == typeid(int))
      vv = atoi(val);
    else
      vv = atol(val);
    if(x < vv) {
      high=mid;
    } else {
      low=mid+1;
    }
  }
  free(val);
  return (Tie)low;
}

template <typename Ti, typename Tie>
Tie binsearch_iat(MPI_File fh, int myrank, int nprocs, Tie nt, Tie x, int header_size, int row_size){
  Ti val[1];
  MPI_Offset mid;
  MPI_Offset low  = 0;
  MPI_Offset high = nt;
  while(low<high) {
    mid=(high+low)/2;
    if (typeid(Ti) == typeid(int)){
      CHECK_MPI(MPI_File_read_at(fh, header_size+(MPI_Offset)mid*row_size, val, 1, MPI_INT, MPI_STATUS_IGNORE));
      // CHECK_MPI(MPI_File_read_at_all_begin(fh, header_size+(MPI_Offset)mid*row_size, val, 1, MPI_INT));
      // CHECK_MPI(MPI_File_read_at_all_end(fh, val, MPI_STATUS_IGNORE));  
    } else {
      CHECK_MPI(MPI_File_read_at(fh, header_size+(MPI_Offset)mid*row_size, val, 1, MPI_LONG, MPI_STATUS_IGNORE));
      // CHECK_MPI(MPI_File_read_at_all_begin(fh, header_size+(MPI_Offset)mid*row_size, val, 1, MPI_LONG));
      // CHECK_MPI(MPI_File_read_at_all_end(fh, val, MPI_STATUS_IGNORE));
    }
    if(x < val[0]) {
      high=mid;
    } else {
      low=mid+1;
    }
  }
  return (Tie)low;
}




/*            NOTE
  partition = 1  --  using partition among nterms of the matrix
  partition = 0  --  using partition among nrows  of the matrix
  (when nprocs = 1 => both partitions are equivalent)

  header_type = 1 -- if the matrix header is [nrows ncols nterms]
  header_type = 0 -- if the matrix header is [nrows nterms]

  IF YOU HAVE EMPTY ROWS (rows without any entries) if they fall on the end of partitions
            - then you might have problems when partitioning among the rows
*/
template < typename Ti = int, typename Tv = double, typename Tie = long int>
int READ_BINARY_MATRIX_MPI(const char *MAT_file, Tie *nr, Tie *nc, Tie *nt, Tie &ntrank, 
                            Ti **irow, Ti **ja, Tv **coef, 
                            int myrank, int nprocs, 
                            bool header_type = 1, bool partition = 0){

  MPI_File fh;
  MPI_Datatype header_mpi;
  MPI_Offset my_offset;
   
  // Define structs for binary writing
  struct header { Tie nr,nc,nt; } header_var;
  struct row { Ti ii,jj; Tv aa; };
  int header_size, row_size;
  int long long bufsize;
  int long long MAX_SIZE_MPI = 1073741824; // 2^30 = INT_MAX / 2  (this value is chosen to be large but no so close to INT_MAX since)
                                           //                     (for some runs using INT_MAX was giving problems in reading ascii)
  Tie tot = 0;
  ////////////////////////////////////////////////////////////////////////////////////
                                //  H E A D E R rectangular
  if (header_type){
    int l_h[3] = { 1, 1, 1};
    MPI_Aint displ_h[3];  
    MPI_Aint address_h;
    MPI_Get_address(&header_var, &address_h);
    MPI_Get_address(&header_var.nr, &displ_h[0]);
    MPI_Get_address(&header_var.nc, &displ_h[1]);
    MPI_Get_address(&header_var.nt, &displ_h[2]);
    displ_h[0] = MPI_Aint_diff(displ_h[0], address_h);
    displ_h[1] = MPI_Aint_diff(displ_h[1], address_h);
    displ_h[2] = MPI_Aint_diff(displ_h[2], address_h);  
    MPI_Datatype types_h[3] = { MPI_LONG, MPI_LONG, MPI_LONG };
    MPI_Type_create_struct(3, l_h, displ_h, types_h, &header_mpi);
    MPI_Type_commit(&header_mpi);
    MPI_Type_size(header_mpi, &header_size);
  } else{                       //  H E A D E R square
    int l_h[2] = { 1, 1};
    MPI_Aint displ_h[2];
    MPI_Aint address_h;
    MPI_Get_address(&header_var, &address_h);
    MPI_Get_address(&header_var.nr, &displ_h[0]);
    MPI_Get_address(&header_var.nt, &displ_h[1]);
    displ_h[0] = MPI_Aint_diff(displ_h[0], address_h);
    displ_h[1] = MPI_Aint_diff(displ_h[1], address_h);
    MPI_Datatype types_h[2] = { MPI_LONG, MPI_LONG };
    MPI_Type_create_struct(2, l_h, displ_h, types_h, &header_mpi);
    MPI_Type_commit(&header_mpi);
    MPI_Type_size(header_mpi, &header_size);
  }
  //////////////////////////////////////////////////////////////////////////////////////
  //                                  R O W
  //////////////////////////////////////////////////////////////////////////////////////
  MPI_Datatype row_mpi;
  int l_r[3] = { 1, 1, 1 };
  MPI_Aint displ_r[3];
  struct row row_var;
  MPI_Aint address_r;
  MPI_Get_address(&row_var, &address_r);
  MPI_Get_address(&row_var.ii, &displ_r[0]);
  MPI_Get_address(&row_var.jj, &displ_r[1]);
  MPI_Get_address(&row_var.aa, &displ_r[2]);
  displ_r[0] = MPI_Aint_diff(displ_r[0], address_r);
  displ_r[1] = MPI_Aint_diff(displ_r[1], address_r);
  displ_r[2] = MPI_Aint_diff(displ_r[2], address_r);
  MPI_Datatype types_r[3];
  if (typeid(Ti) == typeid(int)) {
    types_r[0] = MPI_INT;
    types_r[1] = MPI_INT;
    types_r[2] = MPI_DOUBLE;
  }
  else {
    types_r[0] = MPI_LONG;
    types_r[1] = MPI_LONG;
    types_r[2] = MPI_DOUBLE;
  }
  MPI_Type_create_struct(3, l_r, displ_r, types_r, &row_mpi);
  MPI_Type_commit(&row_mpi);
  MPI_Type_size(row_mpi, &row_size);
  //////////////////////////////////////////////////////////////////////////////////////
  //                                  OPEN FILE
  //////////////////////////////////////////////////////////////////////////////////////
  // works with WORLD and SELF
  CHECK_MPI(MPI_File_open(MPI_COMM_SELF, MAT_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh));
  //                                READ HEADER
  CHECK_MPI(MPI_File_read_at_all_begin(fh, 0, &header_var, 1, header_mpi));
  CHECK_MPI(MPI_File_read_at_all_end(fh, &header_var, MPI_STATUS_IGNORE));

  *nr = header_var.nr;
  *nt = header_var.nt;
  if (header_type) *nc = header_var.nc;
  else             *nc = header_var.nr;

  if (myrank == 0) std::cout << "read mpi, matrix has nr = " << *nr << ", nc = " << *nc << ", nt = " << *nt << std::endl;

  //////////////////////////////////////////////////////////////////////////////////////
  long int psta = 0;
  long int pend;
  if ( partition ){     // subdivision among the nonzero terms
    ntrank = *nt / nprocs;
    if (myrank < *nt % nprocs) ntrank++;

    pend = ntrank;

    if (typeid(Tie) == typeid(int)) {
      CHECK_MPI(MPI_Exscan(&ntrank, &tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    }else{
      CHECK_MPI(MPI_Exscan(&ntrank, &tot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD));
    }

    if (myrank < nprocs-1) ntrank += 16384;  // this part must be parametrized
  }else{                // subdivision among the rows
    if (myrank == nprocs - 1) ntrank = *nr - ( (*nr / nprocs) * (nprocs - 1) );
    else ntrank = *nr / nprocs;

    if (typeid(Tie) == typeid(int)) {
      CHECK_MPI(MPI_Exscan(&ntrank, &tot, 1,  MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    }else{
      CHECK_MPI(MPI_Exscan(&ntrank, &tot, 1,  MPI_LONG, MPI_SUM, MPI_COMM_WORLD));
    }
    
    // find the ending of the chunk of rows
    ntrank = binsearch_iat<Ti>(fh, myrank, nprocs, *nt, tot+ntrank, header_size, row_size);
    // find the beginning of the chunk of rows
    tot    = binsearch_iat<Ti>(fh, myrank, nprocs, *nt, tot       , header_size, row_size);
    ntrank = ntrank - tot;
  }

  my_offset = header_size+(MPI_Offset)tot*row_size;
  bufsize = ntrank*row_size;


  struct row *row_arr = (struct row*) malloc(bufsize); if (row_arr==NULL) printf("ERROR malloc row_arr\n");

  //////////////////////////////////////////////////////////////////////////////////////
  //                                  READ ROWS
  //    *  on bender, the constraint is on the bufsize
  //    *  on marconi, the constaint can be on nt (eliminating also the while loop)
  // since the time difference is not significant in using the chunks of smaller size, 
  // let's go with the safer version -- of smaller chunks.
  if (bufsize > MAX_SIZE_MPI){
      int long chunk_size = MAX_SIZE_MPI*row_size;
      while (chunk_size > 1073741824) {
          MAX_SIZE_MPI /= 2;
          chunk_size = (int long)MAX_SIZE_MPI*row_size;
      }
      int long rest = ntrank;
      int cycle = ntrank / MAX_SIZE_MPI;
      int i;
      struct row *my_buffer = row_arr;
      for(i=0;i<cycle;i++){
          CHECK_MPI(MPI_File_read_at_all_begin(fh, my_offset, my_buffer, MAX_SIZE_MPI, row_mpi));
          CHECK_MPI(MPI_File_read_at_all_end(fh, my_buffer, MPI_STATUS_IGNORE));
          my_buffer += (MPI_Offset)MAX_SIZE_MPI;
          my_offset += (MPI_Offset)MAX_SIZE_MPI*row_size;
          rest      -=   (int long)MAX_SIZE_MPI;
      }
      CHECK_MPI(MPI_File_read_at_all_begin(fh, my_offset , my_buffer, rest, row_mpi));
      CHECK_MPI(MPI_File_read_at_all_end(fh, my_buffer, MPI_STATUS_IGNORE));
  }else{ 
      CHECK_MPI(MPI_File_read_at_all_begin(fh, my_offset, row_arr, ntrank, row_mpi));
      CHECK_MPI(MPI_File_read_at_all_end(fh, row_arr, MPI_STATUS_IGNORE));
  }

  if ( partition ){ 
    Ti pa = row_arr[psta].ii;
    Ti pb = row_arr[pend].ii;
    if (myrank > 0       ) while (row_arr[psta].ii == pa) psta++;
    if (myrank < nprocs-1) while (row_arr[pend].ii == pb && pend < ntrank) pend++;

    ntrank = pend-psta;
  }

  if(MPI_File_close(&fh) != MPI_SUCCESS){
      printf("[MPI process %d] Failure in closing the file.\n", myrank);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  *irow = (Ti *) malloc(ntrank * sizeof(Ti)); if (*irow  ==NULL) printf("ERROR malloc irow   \n");
  *ja   = (Ti *) malloc(ntrank * sizeof(Ti)); if (*ja    ==NULL) printf("ERROR malloc ja     \n");
  *coef = (Tv *) malloc(ntrank * sizeof(Tv)); if (*coef  ==NULL) printf("ERROR malloc coef   \n");

  Ti *irow_ptr  = *irow;
  Ti *ja_ptr    = *ja;
  Tv *coef_ptr  = *coef;

  if (partition){
    for (Ti i = psta; i < pend; i++){
      irow_ptr[i-psta] = row_arr[i].ii;
      ja_ptr  [i-psta] = row_arr[i].jj-1;
      coef_ptr[i-psta] = row_arr[i].aa;
   }
  }else{
    for (Ti i = 0; i < ntrank; i++){
      irow_ptr[i] = row_arr[i].ii;
      ja_ptr[i]   = row_arr[i].jj - 1;
      coef_ptr[i] = row_arr[i].aa;
    }
  }

  free(row_arr);
  MPI_Type_free(&header_mpi);
  MPI_Type_free(&row_mpi);

  // check correctness
  // printf("\n\n%d %ld      %ld %ld %f\n",myrank,(long int)(tot+ntrank+1),\
  //                                          (long int)irow_ptr[ntrank-1],\
  //                                          (long int)ja_ptr  [ntrank-1]+1,\
  //                                                  coef_ptr  [ntrank-1]);   

   return 0;
}







template <typename Ti, typename Tie>
int READ_ASCII_MATRIX_MPI(const char *MAT_file, Tie *nr, Tie *nc, Tie *nt, Tie &ntrank, Ti **irow,
                    Ti **ja, double **coef, int myrank, int nprocs){
   
   MPI_File fh;
   MPI_Offset my_offset;
   MPI_Offset bufsize;

   long int tot = 0;
   char line[MAX_LINE+1];
   FILE *fid;
   int off_1,off_2,header_size = 0,row_size;
   
   fid = fopen(MAT_file, "r");
   if (!fid) exit(1);
   
   // read header
   while(*(fgets(line,MAX_LINE,fid)) == '%');
    if (scan_string(line,nr,nc,nt) != 3){
      if (scan_string(line,nr,nt) == 2) *nc = *nr;
      else {
        printf("wrong matrix header\n");
        fclose(fid);
        exit(1);
      }
    }
  //  if (typeid(Tie) == typeid(int)) {
  //   while(*(fgets(line,MAX_LINE,fid)) == '%');
  //   if (scan_string(line,nr,nc,nt) != 3){
  //     if (scan_string(line,nr,nt) == 2) *nc = *nr;
  //     else {
  //       printf("wrong matrix header\n");
  //       fclose(fid);
  //       exit(1);
  //     }
  //   }
  //  }else{
  //   while(*(fgets(line,MAX_LINE,fid)) == '%');
  //   if (sscanf(line, "%ld %ld %ld", nr, nc, nt) != 3){
  //       if (sscanf(line, "%ld %ld", nr, nt) != 2) *nc = *nr;
  //       else {
  //         printf("wrong matrix header\n");
  //         fclose(fid);
  //         exit(1);
  //       }
  //   }
  //  }

   while(line[header_size] != '\n' && header_size <= MAX_LINE) header_size++;
   header_size++;
   
   {// find body offsets
      FILE in = *fid;
      int i = 0;
      if (fgets(line,MAX_LINE,fid) == NULL) printf("error reading body offsets\n");
      while (line[i] == ' '  && i <= MAX_LINE) i++;
      while (line[i] != ' '  && i <= MAX_LINE) i++;
      off_1 = i;
      while (line[i] == ' '  && i <= MAX_LINE) i++;
      while (line[i] != ' '  && i <= MAX_LINE) i++;
      off_2 = i;
      while (line[i] != '\n' && i <= MAX_LINE) i++;
      row_size = i+1;
      *fid = in;
   }
   if (myrank == 0) std::cout << "Matrix nrows " << *nr << ", ncols " << *nc << " and nterm " << *nt << "\n";
   
    //////////////////////////////////////////////////////////////////////////////////////
      CHECK_MPI(MPI_File_open(MPI_COMM_SELF, MAT_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh));
   if ( 0 ){     // subdivision among the nonzero terms (to be completed)
     ntrank = *nt / nprocs;
     if (myrank < *nt % nprocs) ntrank++;
 
     if (typeid(Tie) == typeid(int)) {
       CHECK_MPI(MPI_Exscan(&ntrank, &tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
     }else{
       CHECK_MPI(MPI_Exscan(&ntrank, &tot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD));
     }
   }else{       // subdivision among the rows
     // ntrank = *nr / nprocs;
     // if (myrank < *nr % nprocs) ntrank++;
     if (myrank == nprocs - 1) ntrank = *nr - ( (*nr / nprocs) * (nprocs - 1) );
     else ntrank = *nr / nprocs;
 
     if (typeid(Tie) == typeid(int)) {
       CHECK_MPI(MPI_Exscan(&ntrank, &tot, 1,  MPI_INT, MPI_SUM, MPI_COMM_WORLD));
     }else{
       CHECK_MPI(MPI_Exscan(&ntrank, &tot, 1,  MPI_LONG, MPI_SUM, MPI_COMM_WORLD));
     }
     
     // find the ending of the chunk of rows
     ntrank = binsearch_iat_str<Ti>(fh, myrank, nprocs, (Tie)(*nt), (Tie)(tot+ntrank), header_size, row_size);
     // find the beginning of the chunk of rows
     tot    = binsearch_iat_str<Ti>(fh, myrank, nprocs, (Tie)(*nt), (Tie)tot       , header_size, row_size);
     ntrank = ntrank - tot;
   }
   ////////////////////////////////////////////////////////////////////////////////////// 
   
   // allocate data
   *irow = (Ti*) malloc(ntrank * sizeof(Ti));
   *ja = (Ti*) malloc(ntrank * sizeof(Ti));
   *coef = (double*) malloc(ntrank * sizeof(double));
   if (irow==NULL) printf("ERROR malloc irow\n");
     if (*ja==NULL) printf("ERROR malloc ja\n");
   if (*coef==NULL) printf("ERROR malloc coef\n");
   char *buffer = (char*)malloc( (ntrank*row_size+1)*sizeof(char));

   Ti *irow_ptr = *irow;
   Ti *ja_ptr = *ja;
   double *coef_ptr = *coef;

   fclose(fid);

   ////////////////////////////////////////// M P I //////////////////////////////////////////

   MPI_Barrier(MPI_COMM_WORLD);

  //  if(MPI_File_open(MPI_COMM_WORLD, MAT_file, MPI_MODE_RDONLY | MPI_MODE_UNIQUE_OPEN, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
  //  {
  //     printf("[MPI process %d] Failure in opening the file.\n", myrank);
  //     MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  //  }

   /*
      Collective I/O combined with noncontiguous accesses yields the highest performance
      MPI_File_read_at_all_begin is collective, nonblocking, with explicit offset routine
   */
   my_offset = header_size+tot*row_size;
   bufsize = ntrank*row_size;
   int MAX_SIZE_MPI = 1073741824;
   if (bufsize > MAX_SIZE_MPI){
      int long long rest = bufsize;
      int cycle = bufsize / MAX_SIZE_MPI;
      int i;
      char *my_buffer = buffer;
      for(i=0;i<cycle;i++){
         if (MPI_File_read_at_all_begin(fh, my_offset, my_buffer, MAX_SIZE_MPI, MPI_CHAR) != MPI_SUCCESS){
            printf("[MPI process %d] Failure in MPI_File_read_at_all_begin 0.\n", myrank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
         }
         if (MPI_File_read_at_all_end(fh,my_buffer,MPI_STATUS_IGNORE) !=  MPI_SUCCESS){
            printf("[MPI process %d] Failure in MPI_File_read_at_all_end 0.\n", myrank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
         }
         my_buffer += MAX_SIZE_MPI;
         my_offset += MAX_SIZE_MPI;
         rest      -= MAX_SIZE_MPI;
      }
      if (MPI_File_read_at_all_begin(fh, my_offset , my_buffer, rest, MPI_CHAR) != MPI_SUCCESS){
         printf("[MPI process %d] Failure in MPI_File_read_at_all_begin 1 .\n", myrank);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      if (MPI_File_read_at_all_end(fh,my_buffer,MPI_STATUS_IGNORE) != MPI_SUCCESS){
         printf("[MPI process %d] Failure in MPI_File_read_at_all 1.\n", myrank);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
   }
   else{ 
      if (MPI_File_read_at_all_begin(fh, my_offset , buffer, bufsize, MPI_CHAR) != MPI_SUCCESS){
         printf("[MPI process %d] Failure in MPI_File_read_at_all_begin 2.\n", myrank);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      if (MPI_File_read_at_all_end(fh,buffer,MPI_STATUS_IGNORE) !=  MPI_SUCCESS){
         printf("[MPI process %d] Failure in MPI_File_read_at_all_end 2.\n", myrank);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
   }
   
   if(MPI_File_close(&fh) != MPI_SUCCESS)
   {
      printf("[MPI process %d] Failure in closing the file.\n", myrank);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }

   char *curLine = buffer;
   if (typeid(Ti) == typeid(int)){
      for (Ti i = 0; i < ntrank; i++){
         irow_ptr[i] = atoi(curLine);
         ja_ptr[i] = atoi(curLine+off_1) - 1;
        //  coef_ptr[i] = atof(curLine+off_2);
         coef_ptr[i] = fast_atof(curLine+off_2);

         curLine += row_size; // curLine = strchr(curLine, '\n') + 1;
      }
   }else{
      for (Ti i = 0; i < ntrank; i++){
         irow_ptr[i] = atol(curLine);
         ja_ptr[i] = atol(curLine+off_1) - 1;
        //  coef_ptr[i] = atof(curLine+off_2);
         coef_ptr[i] = fast_atof(curLine+off_2);

         curLine += row_size; // curLine = strchr(curLine, '\n') + 1;
      }
   }	

   free(buffer);

   // check correctness
   // printf("\n\n%d %ld      %ld %ld %f\n",myrank,(long int)(tot+ntrank+1),\
   //                                         (long int)irow_ptr[ntrank-1],\
   //                                         (long int)ja_ptr  [ntrank-1] + 1,\
   //                                                 coef_ptr  [ntrank-1]);

  // printf("head\n");
  // for (long int i = 0; i < 12; i++){
  //   printf("%ld %ld %lf\n",(long)irow_ptr[i], (long)ja_ptr[i], coef_ptr[i]);
  // }

  // printf("\ntail\n");
  // for (long int i = ntrank-12; i < ntrank; i++){
  //   printf("%ld %ld %lf\n",(long)irow_ptr[i], (long)ja_ptr[i], coef_ptr[i]);
  // }

   return 0;
}
