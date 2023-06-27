#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <unistd.h>
#include <getopt.h>

#define DIE CHECK_DEVICE(cudaDeviceSynchronize());MPI_Finalize();exit(0);

#include "basic_kernel/matrix/scalar.h"
#include "basic_kernel/matrix/vector.h"
#include "basic_kernel/matrix/matrixIO.h"
#include "utility/myMPI.h"
#include "utility/handles.h"
#include "prec_setup/AMG.h"
#include <string>
#include "utility/distribuite.h"
#include "basic_kernel/halo_communication/local_permutation.h"
#include "basic_kernel/custom_cudamalloc/custom_cudamalloc.h"
#include "utility/utils.h"

#include "utility/function_cnt.h"

#define NUM_THR 1024

// --------------- only for KDevelop ----------------------
#include <curand_mtgp32_kernel.h>
// --------------------------------------------------------

#define MAX_NNZ_PER_ROW_LAP 5
#define MPI 1
#ifndef __GNUC__
typedef int (*__compar_fn_t)(const void *, const void *);
#endif


// ----------------------------------------- New Version ------------------------------------------

// vector<itype>* compute_mask_permut (CSR *Alocal, CSR *Plocal, vector<int> *bitcol, FILE* fp) {
//     itype mypfirstrow, myplastrow;
//     mypfirstrow = Plocal->row_shift;
//     myplastrow  = Plocal->n + Plocal->row_shift-1;
//     fprintf(fp, "first=%d, last=%d\n", mypfirstrow, myplastrow);
//     
//     int i, j;
//     unsigned int k;
// //     print_bitabit(bitcol, fp); fprintf(fp, "\n");
//     for (j= mypfirstrow; j<= myplastrow; j++) {
//         i = j / (sizeof(int) * 8);
//         bitcol->val[i] |= (1U << (j % (sizeof(int) * 8)) );
//         
//     }
// //     print_bitabit(bitcol, fp); fprintf(fp, "\n");
//     
//     vector<itype> *y = NULL;
//     int y_done = 0, y_size = 0;
//     for (i=0; i<bitcol->n; i++)
//         y_size += __builtin_popcount(bitcol->val[i]);
//     y = Vector::init<itype>(2*y_size, true, false);     // Il 2*... è solo per inserire anche i nuovi indici di colonna oltre che quelli originali
//     Vector::fillWithValue(y, 99999);
//     fprintf(fp, "The y_size is %d (y->n=%d)\n", y_size, y->n);
//     
//     unsigned int mask;
//     for (i= 0; i<bitcol->n; i++) {
//         mask = (unsigned int) bitcol->val[i]; // the right-shift on a negative signed integer is implementation-depending
//         
//         
//         
//         int y_id;
//         unsigned int h_mask = __builtin_popcount(mask);
//         for (j= 0; j < 8*sizeof(int); j++) {
//             k = 1U << j;
//             if (mask & k) {
//     //             printf("yid_mask: "); print_bit(mask << (8*sizeof(int) - j), 1, 0); printf("\n");
//                 int original_column =  j + i*(8*sizeof(int));
//                 y_id = y_done + h_mask -  __builtin_popcount(mask >> j);
//     //             fprintf(fp, "The y_id is %d, the col is %d\n", y_id, original_column);
//                 y->val[y_id] = original_column;
//                 y->val[y_id + (y->n/2)] = y_id;
//             }
//             k = k>>1;
//         }
//         y_done += __builtin_popcount(bitcol->val[i]);
//         
//         
//         
//     }
//     
//     return(y);
// }
// 
// __global__
// void compute_mask_permut_GPU_glob (itype m, const itype *permut_mask, itype *shrinking_permut) {
//     int original_column = blockIdx.x * blockDim.x + threadIdx.x;
//     
//     if (original_column < m) {
//         int permut_mask_id = original_column / (8*sizeof(int));
//         int permut_mask_bit = original_column % (8*sizeof(int));
//         
//         unsigned int mask = (unsigned int) permut_mask[permut_mask_id];
//         unsigned int h_mask = __popc(mask);
//         unsigned int k = 1U << permut_mask_bit;
//         
//         if (mask & k) {
//             int y_done = 0, y_id, i;
//             for(i=0; i<permut_mask_id; i++)
//                 y_done += __popc(permut_mask[i]);
//             
//             y_id = y_done + h_mask - __popc(mask >> permut_mask_bit);
//             shrinking_permut[y_id] = original_column;
//         }
//     }
// }
// 
// vector<itype>* compute_mask_permut_GPU (const CSR *Alocal, const CSR *Plocal, vector<int> *bitcol) {
//     assert(Alocal->on_the_device);
//     assert(Plocal->on_the_device);
//     
//     itype mypfirstrow, myplastrow;
//     mypfirstrow = Plocal->row_shift;
//     myplastrow  = Plocal->n + Plocal->row_shift-1;
//     
//     int i, j, y_size = 0;
//     for (j= mypfirstrow; j<= myplastrow; j++) {
//         i = j / (sizeof(int) * 8);
//         bitcol->val[i] |= (1U << (j % (sizeof(int) * 8)) );
//         
//     }
//     VectorcopyToDevice_CNT
//     vector<int> *dev_bitcol = Vector::copyToDevice(bitcol);
//     assert(dev_bitcol->on_the_device);
//     
//     for (i=0; i<bitcol->n; i++)
//         y_size += __builtin_popcount(bitcol->val[i]);
//     
//     Vectorinit_CNT
//     vector<itype> *shrinking_permut = Vector::init<itype>(y_size, true, true);
//     
//     gridblock gb;
//     gb = gb1d((bitcol->n)*32, NUM_THR);
//     compute_mask_permut_GPU_glob<<<gb.g, gb.b>>>(Alocal->m, dev_bitcol->val, shrinking_permut->val);
//     
//     
//     return(shrinking_permut);
// }
// 
// void apply_mask_permut (CSR **Alocal_dev, vector<itype>* mask_permut, FILE* fp) {
//     CSR* Alocal = NULL; bool dev_flag;
//     if((*Alocal_dev)->on_the_device){
//       Alocal = CSRm::copyToHost(*Alocal_dev);
//       dev_flag = true;
//     }else{
//       Alocal = *Alocal_dev;
//       dev_flag = false;
//     }
//     
//     int j, number_of_permutations = (mask_permut->n)/2, start, med, end, flag;
//     fprintf(fp, "Applying %d permutations...", number_of_permutations);
//     fflush(fp);
//     
//     for (j=0; j < Alocal->nnz; j++) {
//         flag = 1;
//         start = 0;
//         end = number_of_permutations;
//         while (flag) {
//             med = start + (end - start)/2;
//             if (Alocal->col[j] == mask_permut->val[med]) {
//                 Alocal->col[j] = mask_permut->val[med + number_of_permutations];
//                 flag = 0;
//             }else{
//                 if (Alocal->col[j] < mask_permut->val[med])
//                     end = med;
//                 else
//                     start = med;
//                 flag = (start != end);
//             }
//         }
//     }
//     Alocal->m = number_of_permutations;
//     
//     fprintf(fp, "\tDone!\n");
//     
//     if (dev_flag){
//         CSRm::free(*Alocal_dev);
//         *Alocal_dev = CSRm::copyToDevice(Alocal);
//         CSRm::free(Alocal);
//     }else{
//         CSRm::free(*Alocal_dev);
//         *Alocal_dev = Alocal;
//         CSRm::free(Alocal);
//     }
//     return;
// }
// 
// __global__
// void apply_mask_permut_GPU_glob (itype nnz, itype *col, int shrinking_permut_len, itype* shrinking_permut) {
//     
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     
//     int number_of_permutations = shrinking_permut_len, start, med, end, flag;
//     
//     if (id < nnz) {
//         flag = 1;
//         start = 0;
//         end = number_of_permutations;
//         while (flag) {
//             med = start + (end - start)/2;
//             if (col[id] == shrinking_permut[med]) {
//                 col[id] = med;
//                 flag = 0;
//             }else{
//                 if (col[id] < shrinking_permut[med])
//                     end = med;
//                 else
//                     start = med;
//                 flag = (start != end);
//             }
//         }
//     }
//     
//     return;
// }
// 
// void apply_mask_permut_GPU (CSR *Alocal, vector<itype>* shrinking_permut, FILE* fp) {
//     assert(Alocal->on_the_device);
//     assert(shrinking_permut->on_the_device);
//     
//     gridblock gb;
//     gb = gb1d(Alocal->nnz, NUM_THR);
//     apply_mask_permut_GPU_glob<<<gb.g, gb.b>>>(Alocal->nnz, Alocal->col, shrinking_permut->n, shrinking_permut->val);
//     
//     Alocal->m = shrinking_permut->n;
//     
//     return;
// }
#if 0
binsearch(const itype array[], itype size, itype value) {
 29   itype low, high, medium;
 30   low=0;
 31   high=size;
 32   while(low<high) {
 33       medium=(high+low)/2;
 34       if(value > array[medium]) {
 35         low=medium+1;
 36       } else {
 37         high=medium;
 38       }
 39   }
 40   return low;
 41 }
#endif

__global__
void apply_mask_permut_GPU_noSideEffects_glob (itype nnz, const itype *col, int shrinking_permut_len, const itype* shrinking_permut, itype* comp_col) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    //int number_of_permutations = shrinking_permut_len, start, med, end, flag;
    int number_of_permutations = shrinking_permut_len, start, med, end;
    
    if (id < nnz) {
        start = 0;
        end = number_of_permutations;
#if 0
        while (flag && (end >= start) ) {
            med = start + (end - start)/2;
            if (col[id] == shrinking_permut[med]) {
                comp_col[id] = med;
                flag = 0;
            }else{
                if (col[id] < shrinking_permut[med])
                    end = med -1;
                else
                    start = med +1;
            }
        }
#else
        while (start<end) {
            med = (end + start)/2;
            if (col[id] > shrinking_permut[med]) {
                    start = med +1;
            } else {
                    end = med;
            }
        }
        comp_col[id] = start;
#endif
    }
    
    return;
}

vector<itype>* apply_mask_permut_GPU_noSideEffects (const CSR *Alocal, const vector<itype>* shrinking_permut) {
    assert(Alocal->on_the_device);
    assert(shrinking_permut->on_the_device);
    
    // ------------- custom cudaMalloc -------------
//     Vectorinit_CNT
//     vector<itype>* comp_col = Vector::init<itype>(Alocal->nnz, true, true);
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    vector<itype>* comp_col;
    if (Alocal->custom_alloced) {
        comp_col = Vector::init<itype>(Alocal->nnz, false, true);
        comp_col->val = CustomCudaMalloc::alloc_itype(Alocal->nnz);
    } else {
        Vectorinit_CNT
        comp_col = Vector::init<itype>(Alocal->nnz, true, true);
    }
    // ---------------------------------------------
    
    
    gridblock gb;
    gb = gb1d(Alocal->nnz, NUM_THR);
    apply_mask_permut_GPU_noSideEffects_glob<<<gb.g, gb.b>>>(Alocal->nnz, Alocal->col, shrinking_permut->n, shrinking_permut->val, comp_col->val);
    
    return(comp_col);
}

// void reverse_mask_permut (CSR **Alocal_dev, vector<itype>* mask_permut, FILE* fp) {
//     CSR* Alocal = NULL; bool dev_flag;
//     if((*Alocal_dev)->on_the_device){
//       Alocal = CSRm::copyToHost(*Alocal_dev);
//       dev_flag = true;
//     }else{
//       Alocal = *Alocal_dev;
//       dev_flag = false;
//     }
//     
//     int j, number_of_permutations = (mask_permut->n)/2, start, med, end, flag;
//     fprintf(fp, "Applying %d permutations...", number_of_permutations);
//     fflush(fp);
//     
//     for (j=0; j < Alocal->nnz; j++) {
//         flag = 1;
//         start = number_of_permutations;
//         end = mask_permut->n;
//         while (flag) {
//             med = start + (end - start)/2;
//             if (Alocal->col[j] == mask_permut->val[med]) {
//                 Alocal->col[j] = mask_permut->val[med - number_of_permutations];
//                 flag = 0;
//             }else{
//                 if (Alocal->col[j] < mask_permut->val[med])
//                     end = med;
//                 else
//                     start = med;
//                 flag = (start != end);
//             }
//         }
//     }
//     Alocal->m = Alocal->full_m;
//     
//     fprintf(fp, "\tDone!\n");
//     
//     if (dev_flag){
//         CSRm::free(*Alocal_dev);
//         *Alocal_dev = CSRm::copyToDevice(Alocal);
//         CSRm::free(Alocal);
//     }else{
//         CSRm::free(*Alocal_dev);
//         *Alocal_dev = Alocal;
//         CSRm::free(Alocal);
//     }
//     return;
// }
// 
// __global__
// void reverse_mask_permut_GPU_glob (itype nnz, itype *col, int shrinking_permut_len, itype* shrinking_permut) {
//     
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     
//     if (id < nnz)
//         col[id] = shrinking_permut[col[id]];
//     
//     return;
// }
// 
// void reverse_mask_permut_GPU (CSR *Alocal, vector<itype>* shrinking_permut, FILE* fp) {
//     assert(Alocal->on_the_device);
//     assert(shrinking_permut->on_the_device);
//     
//     gridblock gb;
//     gb = gb1d(Alocal->nnz, NUM_THR);
//     reverse_mask_permut_GPU_glob<<<gb.g, gb.b>>>(Alocal->nnz, Alocal->col, shrinking_permut->n, shrinking_permut->val);
//     
//     Alocal->m = Alocal->full_m;
//     
//     return;
// }

extern int srmfb;

bool shrink_col(CSR* A, CSR* P) {
    vector<int> *get_shrinked_col( CSR*, CSR* );
    if (!(A->shrinked_flag)) {
        if ( P != NULL ) {    // product compatibility check
	        if(A->m!=P->full_n) {
		       fprintf(stderr,"A->m=%lu, P->full_n=%lu\n",A->m,P->full_n);
	        }
            assert( A->m == P->full_n );
        } else {
            assert( A->m == A->full_n );
        }
  
        vector<itype>* shrinking_permut = get_shrinked_col( A, P );
        if(0) {
  char filename[256];
  snprintf(filename,sizeof(filename),"shrinking_permut_%x_%d",A,srmfb);
  FILE *fp=fopen(filename,"w");
  if(fp==NULL) {
       fprintf(stderr,"Could not open X\n");
  }
  Vector::print(shrinking_permut,-1,fp);
  fclose(fp);
        }
        assert ( shrinking_permut->n >= (P!=NULL ? P->n : A->n) );
        vector<itype>* shrinkedA_col = apply_mask_permut_GPU_noSideEffects (A, shrinking_permut);
        
        A->shrinked_flag = true;
        A->shrinked_m = shrinking_permut->n;
        A->shrinked_col = shrinkedA_col->val;

        Vector::free(shrinking_permut);
        std::free(shrinkedA_col);
        return (true);
    }else{
        return (false);
    }
}

bool shrink_col(CSR* A, stype firstlocal, stype lastlocal, itype global_len) {
    vector<int> *get_shrinked_col( CSR*, stype, stype);
    if (!(A->shrinked_flag)) {
        
        assert( A->m == global_len );
  
        vector<itype>* shrinking_permut = get_shrinked_col( A, firstlocal, lastlocal );
        assert ( shrinking_permut->n >= (lastlocal - firstlocal +1) );
        vector<itype>* shrinkedA_col = apply_mask_permut_GPU_noSideEffects (A, shrinking_permut);
        
        A->shrinked_flag = true;
        A->shrinked_m = shrinking_permut->n;
        A->shrinked_col = shrinkedA_col->val;

        Vector::free(shrinking_permut);
        std::free(shrinkedA_col);
        return (true);
    }else{
        return (false);
    }
}

CSR* get_shrinked_matrix(CSR* A, CSR* P) {
    
    if (!(A->shrinked_flag)) {
        assert( shrink_col(A, P) );
    } else {
        bool test = (P!=NULL) ? (P->row_shift == A->shrinked_firstrow) : (A->row_shift == A->shrinked_firstrow);
        test = test && ((P!=NULL) ? (P->row_shift+P->n == A->shrinked_lastrow) : (A->row_shift+A->n == A->shrinked_lastrow));
        assert ( test ); // NOTE: check Pfirstrow, Plastrow
    }
    
    CSR* A_ = CSRm::init(A->n, A->shrinked_m, A->nnz, false, A->on_the_device, A->is_symmetric, A->full_n, A->row_shift);
    A_->row = A->row;
    A_->val = A->val;
    A_->col = A->shrinked_col;
    
    return(A_);
}
