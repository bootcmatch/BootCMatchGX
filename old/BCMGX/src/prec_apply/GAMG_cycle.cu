#include "prec_apply/GAMG_cycle.h"
#include "basic_kernel/halo_communication/local_permutation.h"
#include "basic_kernel/smoother/relax.h"
#include "utility/distribuite.h"
#include <stdlib.h>

#include "utility/function_cnt.h"

float TOTAL_SOLRELAX_TIME=0.;
float TOTAL_RESTGAMG_TIME=0.;

vector<vtype> *GAMGcycle::Res_buffer;

void GAMGcycle::initContext(int n){
    Vectorinit_CNT
    GAMGcycle::Res_buffer = Vector::init<vtype>(n ,true, true);
}

__inline__
void GAMGcycle::setBufferSize(itype n){
    GAMGcycle::Res_buffer->n = n;
}

void GAMGcycle::freeContext(){
    Vector::free(GAMGcycle::Res_buffer);
}

int cntrelax=0;
extern char idstring[];

void GAMG_cycle(handles *h, int k, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, vectorCollection<vtype> *Rhs, vectorCollection<vtype> *Xtent, vectorCollection<vtype> *Xtent_2, int l, int coarsesolver_type ){
    
  _MPI_ENV;
  static int flow=1;
  static int first=0;

  hierarchy *hrrc = boot_amg->H_array[k];
  int relax_type = amg_cycle->relax_type;

  buildData *amg_data = bootamg_data->amg_data;
  int coarse_solver = amg_data->coarse_solver;

  if(VERBOSE > 0)
    std::cout << "GAMGCycle: start of level " << l << " Max level " << hrrc->num_levels << "\n";
  
  char filename[256];
  FILE *fp;

  if(l == hrrc->num_levels){
      
    if(DETAILED_TIMING && ISMASTER){
        TIME::start();
    }
//#if LOCAL_COARSEST == 1
    if (coarsesolver_type == 1){
    relaxCoarsest(
        h,                                 
        amg_cycle->relaxnumber_coarse,      
        hrrc->A_array[l-1], hrrc->D_array[l-1], hrrc->M_array[l-1],
        Rhs->val[l-1],                      
        coarse_solver,                      
        amg_cycle->relax_weight,            
        Xtent->val[l-1],                    
        &Xtent_2->val[l-1],
        hrrc->A_array[l-1]->n);
     }else{
//#else
     if(first==1 || flow==0) {
     	snprintf(filename,sizeof(filename),"Rhs_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Rhs->val[l-1],-1,fp); 
	fclose(fp);
     	snprintf(filename,sizeof(filename),"Xtent_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Xtent->val[l-1],-1,fp); 
	fclose(fp);
     }
    relax(
        h,
        amg_cycle->relaxnumber_coarse,
        l-1,
        hrrc->A_array[l-1], hrrc->D_array[l-1], hrrc->M_array[l-1],
        Rhs->val[l-1], 
        coarse_solver,
        amg_cycle->relax_weight,
        Xtent->val[l-1], 
        &Xtent_2->val[l-1] );

     }
    	if(first==1 || flow==0) {
     	snprintf(filename,sizeof(filename),"Xtent_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Xtent->val[l-1],-1,fp); 
	fclose(fp);
        snprintf(filename,sizeof(filename),"Xtent2_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
        FILE *fp=fopen(filename,"w");
        if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
     	}
     	Vector::print(Xtent_2->val[l-1],-1,fp); 
     	fclose(fp);
	flow++;
    	 }

//#endif
   
    if(DETAILED_TIMING && ISMASTER){
        TOTAL_SOLRELAX_TIME += TIME::stop();
    }
    
  }else{
      
    // presmoothing steps
    if(DETAILED_TIMING && ISMASTER){
        TIME::start();
    }
     if(first==1 || flow==0) {
     	snprintf(filename,sizeof(filename),"Rhs_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Rhs->val[l-1],-1,fp); 
	fclose(fp);
     	snprintf(filename,sizeof(filename),"Xtent_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Xtent->val[l-1],-1,fp); 
	fclose(fp);
     }
    relax(
      h,
      amg_cycle->prerelax_number*((amg_cycle->cycle_type!=4)?1:(1<<(l-1))),
      l-1,
      hrrc->A_array[l-1], hrrc->D_array[l-1], hrrc->M_array[l-1],
      Rhs->val[l-1],
      relax_type,
      amg_cycle->relax_weight,
      Xtent->val[l-1],
      &Xtent_2->val[l-1]
    );
    	if(first==1 || flow==0) {
        snprintf(filename,sizeof(filename),"Xtent2_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
        FILE *fp=fopen(filename,"w");
        if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
     	}
     	Vector::print(Xtent_2->val[l-1],-1,fp); 
     	fclose(fp);
	flow++;
    	 }
    

    if(DETAILED_TIMING && ISMASTER){
        TOTAL_SOLRELAX_TIME += TIME::stop();
    }

    if(VERBOSE > 1){
        vtype tnrm = Vector::norm(h->cublas_h, Rhs->val[l-1]);
        std::cout << "RHS at level " << l << " " << tnrm << "\n";
        tnrm = Vector::norm(h->cublas_h, Xtent->val[l-1]);
        std::cout << "After first smoother at level " << l << " XTent " << tnrm << "\n";
    }

    // compute residual
    if(DETAILED_TIMING && ISMASTER){
            TIME::start();
    }
    GAMGcycle::setBufferSize(Rhs->val[l-1]->n);
    vector<vtype> *Res = GAMGcycle::Res_buffer;
    Vector::copyTo(Res, Rhs->val[l-1]);

    // sync
    if(nprocs > 1) {
        halo_sync(hrrc->A_array[l-1]->halo, hrrc->A_array[l-1], Xtent->val[l-1], true);
    }
    
    CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, hrrc->A_array[l-1], Xtent->val[l-1], Res, -1., 1.);
    
    if(VERBOSE > 1){
        vtype tnrm = Vector::norm_MPI(h->cublas_h, Res);
        std::cout << "Residual at level " << l << " " << tnrm << "\n";
    }
    

    if(nprocs == 1){
    	if(first==1||flow==0) {
        snprintf(filename,sizeof(filename),"Rlocal_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	CSRm::printMM(hrrc->R_array[l-1],filename);
        snprintf(filename,sizeof(filename),"Res_%d_full_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
        FILE *fp=fopen(filename,"w");
        if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
     	}
     	Vector::print(Res,-1,fp); 
     	fclose(fp);
    	 }
        CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, hrrc->R_array[l-1], Res, Rhs->val[l], 1., 0.);
     if(first==1 || flow==0) {
     	snprintf(filename,sizeof(filename),"Rhs2_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Rhs->val[l],-1,fp); 
	fclose(fp);
	flow++;
     }
    }else{
//#if LOCAL_COARSEST == 1
      if (coarsesolver_type == 1){
    // before coarsets
        if(l == hrrc->num_levels-1){
            CSR *R = hrrc->R_array[l-1];
            vector<vtype> *cust_null = NULL;    // BUG
            vector<vtype> *Res_full = aggregateVector(Res, hrrc->A_array[l-1]->full_n, cust_null );
            assert( hrrc->R_array[l-1]->n == hrrc->R_array[l-1]->full_n );
            CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, R, Res_full, Rhs->val[l], 1., 0.);
                
            Vector::free(Res_full);
        }else{

            CSR *R_local = hrrc->R_local_array[l-1];
            assert(hrrc->A_array[l-1]->full_n == R_local->m);

            vector<vtype> *Res_full = Xtent_2->val[l-1];
            cudaMemcpy(Res_full->val, Res->val, hrrc->A_array[l-1]->n*sizeof(vtype), cudaMemcpyDeviceToDevice);
            
            CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, R_local, Res_full, Rhs->val[l], 1., 0.);
        }
      }else{
//#else
        CSR *R_local = hrrc->R_local_array[l-1];
        assert(hrrc->A_array[l-1]->full_n == R_local->m);

        vector<vtype> *Res_full = Xtent_2->val[l-1];
        cudaMemcpy(Res_full->val, Res->val, hrrc->A_array[l-1]->n*sizeof(vtype), cudaMemcpyDeviceToDevice);
    	if(first==1 || flow==0) {
        snprintf(filename,sizeof(filename),"Rlocal_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	CSRm::printMM(R_local,filename);
        snprintf(filename,sizeof(filename),"Res_%d_full_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
        FILE *fp=fopen(filename,"w");
        if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
     	}
     	Vector::print(Res_full,-1,fp); 
     	fclose(fp);
    	 }
        
        CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, R_local, Res_full, Rhs->val[l], 1., 0.);
     if(first==1 || flow==0) {
     	snprintf(filename,sizeof(filename),"Rhs2_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Rhs->val[l],-1,fp); 
	fclose(fp);
	flow++;
      }
	  }
//#endif
    }
    
    
    if(nprocs>1) {
        Vector::fillWithValue(Xtent->val[l], 0.);
    } else {
        Vector::fillWithValue(Xtent->val[l], 0.);
    }
    if(DETAILED_TIMING && ISMASTER){
            TOTAL_RESTGAMG_TIME += TIME::stop();
    }
    if(hrrc->num_levels>2 || amg_cycle->relaxnumber_coarse>0) {
//	    if(myid==0) { printf("Task 0 calling GAMG_cycle recursively for l=%d\n",l); }
	    for(int i=1; i<=amg_cycle->num_grid_sweeps[l-1]; i++){
	      GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, Rhs, Xtent, Xtent_2, l+1, coarsesolver_type);
	      if(l == hrrc->num_levels-1)
	        break;
    	   }
    }

    if(DETAILED_TIMING && ISMASTER){
            TIME::start();
    }
    if(nprocs == 1) {
    	if(first==1||flow==0) {
        snprintf(filename,sizeof(filename),"Plocal_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	CSRm::printMM(hrrc->P_array[l-1],filename);
        snprintf(filename,sizeof(filename),"Xtent_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
        FILE *fp=fopen(filename,"w");
        if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
     	}
     	Vector::print(Xtent->val[l],-1,fp); 
     	fclose(fp);
    	 }
        CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, hrrc->P_array[l-1], Xtent->val[l], Xtent->val[l-1], 1., 1.);
     if(first==1 || flow==0) {
     	snprintf(filename,sizeof(filename),"Xtent_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Xtent->val[l-1],-1,fp); 
	fclose(fp);
	flow++;
     }

    } else {
        // before coarsets
        assert( hrrc->P_local_array[l-1]->halo.init == 1 );
        halo_sync(hrrc->P_local_array[l-1]->halo, hrrc->P_local_array[l-1], Xtent->val[l], false);
        assert (hrrc->P_local_array[l-1]->shrinked_flag == 1);
//#if LOCAL_COARSEST==1
     if (coarsesolver_type == 1){
        if (l == hrrc->num_levels-1) {
            assert( (hrrc->A_array[l]->n == hrrc->A_array[l]->full_n) && (Xtent->val[l]->n == hrrc->A_array[l]->full_n) );  
            CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, hrrc->P_local_array[l-1], Xtent->val[l], Xtent->val[l-1], 1., 1.); 
        } else {
//#endif
            CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, hrrc->P_local_array[l-1], Xtent->val[l], Xtent->val[l-1], 1., 1.);
//#if LOCAL_COARSEST==1
        }
      }
//#endif
      if ( coarsesolver_type == 0 ){
    	if(first==1 || flow==0) {
        snprintf(filename,sizeof(filename),"Plocal_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	CSRm::printMM(hrrc->P_local_array[l-1],filename);
        snprintf(filename,sizeof(filename),"Xtent_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
        FILE *fp=fopen(filename,"w");
        if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
     	}
     	Vector::print(Xtent->val[l],-1,fp); 
     	fclose(fp);
    	 }
            CSRm::CSRVector_product_adaptive_miniwarp_new(h->cusparse_h0, hrrc->P_local_array[l-1], Xtent->val[l], Xtent->val[l-1], 1., 1.);
     if(first==1 || flow==0) {
     	snprintf(filename,sizeof(filename),"Xtent_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Xtent->val[l-1],-1,fp); 
	fclose(fp);
	flow++;
     }
      }
   
    }
    
    if(DETAILED_TIMING && ISMASTER){
            TOTAL_RESTGAMG_TIME += TIME::stop();
    }


    if(VERBOSE > 1){
      vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l-1]);
      std::cout << "After recursion at level " << l << " XTent " << tnrm << "\n";
    }

    if(DETAILED_TIMING && ISMASTER){
        TIME::start();
    }
    

    // postsmoothing steps
     if(first==1 || flow==0) {
     	snprintf(filename,sizeof(filename),"Rhs_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Rhs->val[l-1],-1,fp); 
	fclose(fp);
     	snprintf(filename,sizeof(filename),"Xtent_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
	fp=fopen(filename,"w");
	if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
	}	 
     	Vector::print(Xtent->val[l-1],-1,fp); 
	fclose(fp);
     }
    relax(
      h,
      amg_cycle->postrelax_number*((amg_cycle->cycle_type!=4)?1:(1<<(l-1))),
      l-1,
      hrrc->A_array[l-1], hrrc->D_array[l-1], hrrc->M_array[l-1],
      Rhs->val[l-1],
      relax_type,
      amg_cycle->relax_weight,
      Xtent->val[l-1],
      &Xtent_2->val[l-1]
    );
    	if(first==1 || flow==0) {
        snprintf(filename,sizeof(filename),"Xtent2_%d_%s_%d_%d_%d",__LINE__,idstring,amg_cycle->relaxnumber_coarse,flow,myid);
        FILE *fp=fopen(filename,"w");
        if(fp==NULL) {
	     printf("Could not open %s\n",filename);
	     exit(1);
     	}
     	Vector::print(Xtent_2->val[l-1],-1,fp); 
     	fclose(fp);
     	flow++;
    	 }
    
    if(DETAILED_TIMING && ISMASTER){
      	TOTAL_SOLRELAX_TIME += TIME::stop();
    }


    if(VERBOSE > 1){
      vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l-1]);
      std::cout << "After second smoother at level " << l << " XTent " << tnrm << "\n";
    }

  }
  first=0;

  if(VERBOSE > 0)
      std::cout << "GAMGCycle: end of level " << l << "\n";
   cntrelax=0;  
}

