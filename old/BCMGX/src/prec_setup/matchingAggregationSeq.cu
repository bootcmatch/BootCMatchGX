
/* input:
  unshifted matching
  local w
*/
CSR* matchingPairAggregationSeq(itype shift, vector<vtype> *w_, vector<itype> *M_, bool symmetric){

  assert(w_->on_the_device);
  assert(M_->on_the_device);

  vector<vtype> *w = Vector::copyToHost(w_);
  vector<itype> *M = Vector::copyToHost(M_);

  itype nrows_A = M->n;
  itype i, j;
  vtype wagg0, wagg1, normwagg;
  itype ncolc = 0, npairs = 0, nsingle = 0;

  itype *p = M->val;
  vtype *w_data = w->val;

  itype *markc = (itype *) calloc(nrows_A, sizeof(itype));

  for(i=0; i<nrows_A; ++i)
    markc[i] = -1;

  vtype *wtempc = (vtype *) calloc(nrows_A, sizeof(vtype));

  for(i=0; i<nrows_A; ++i){
      j = p[i];
      if((j>=0) && (i != j)){
	       if(markc[i] == -1 && markc[j]== -1){
	          wagg0=w_data[i+shift];
	          wagg1=w_data[j+shift];
	          normwagg=sqrt(pow(wagg0,2)+pow(wagg1,2));

	          if(normwagg > DBL_EPSILON){
        		  markc[i]=ncolc;
        		  markc[j]=ncolc;
        		  wtempc[i]=w_data[i+shift]/normwagg;
        		  wtempc[j]=w_data[j+shift]/normwagg;
        		  ncolc++;
        		  npairs++;
            }
          }
        }
    }

  for(i=0; i<nrows_A; ++i){
    if(markc[i]==-1){
	     if(fabs(w_data[i+shift]) <= DBL_EPSILON){
        printf("BAD singleton: %d \n", i);
	      /* only-fine grid node: corresponding null row in the prolongator */
	      markc[i]=ncolc-1;
	      wtempc[i]=0.0;
	    }else{
        printf("GOOD singleton: %d \n", i);
	      markc[i]=ncolc;
	      ncolc++;
	      wtempc[i]=w_data[i+shift]/fabs(w_data[i+shift]);
	      nsingle++;
	    }
    }
  }

  itype ncoarse=npairs+nsingle;

  assert(ncolc == ncoarse);

  /* Each row of P has only 1 element. It can be zero in case
     of only-fine grid variable or singleton */

  CSR *P = CSRm::init(nrows_A, ncolc, nrows_A, true, false, false, -1, shift);

  if (ncoarse > 0){
    for(i=0; i<nrows_A; i++){
  	  P->row[i]=i;
  	  P->col[i]=markc[i];
  	  P->val[i]=wtempc[i];
	   }
    P->row[nrows_A]=nrows_A;
  }

  free(markc);
  free(wtempc);
  Vector::free(w);
  Vector::free(M);

  return P;
}
