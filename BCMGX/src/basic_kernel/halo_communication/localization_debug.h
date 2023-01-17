#ifndef LOCALIZATION_DEBUG

#include <unistd.h>
#include <stdio.h>
#include "prec_setup/AMG.h"

#define LOCALIZATION_DEBUG

#define PICO_PRINT(X) \
  {\
    FILE *fp;\
    char s[50], s1[50];\
    sprintf(s, "temp_%d.txt", myid);\
    fp = fopen ( s, "w" );\
    fclose(fp);\
    fp = fopen ( s, "a+" );\
    X;\
    fclose(fp);\
    for (int i=0; i<nprocs; i++) {\
        if (myid == i) {\
            printf("\t------------------------- Proc %d File %s Line %d -------------------------\n\n", i, __FILE__, __LINE__);\
            sprintf(s1, "cat temp_%d.txt", myid);\
            system(s1);\
            sprintf(s1, "rm temp_%d.txt", myid);\
            system(s1);\
        }\
        MPI_Barrier(MPI_COMM_WORLD);\
    }\
  }
 
// ---------------------------------------------------------------------------------
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define PRINT_DEVICE_POINTER_INFO(X,Y) \
    { \
        cudaPointerAttributes cu_temp_att;\
        cudaPointerGetAttributes ( &cu_temp_att, X );\
        const char *type;\
        if (cu_temp_att.type == 0)\
            type = "Unregistered memory";\
        if (cu_temp_att.type == 1)\
            type = "Host memory";\
        if (cu_temp_att.type == 2)\
            type = "Device memory";\
        if (cu_temp_att.type == 3)\
            type = "Managed memory";\
        char s0[500];\
        sprintf(s0, "\
        struct cudaPointerAttributes ( %s ) {\n\
                enum cudaMemoryType type = %s\n\
                int device = %d;\n\
                void *devicePointer = %p;\n\
                void *hostPointer = %p;\n\
            }\n",\
        #X, type, cu_temp_att.device, cu_temp_att.devicePointer, cu_temp_att.hostPointer);\
        if ( Y ) {\
            PICO_PRINT( fprintf(fp, "%s", s0); fprintf(fp, "\tmyid = %d\ti = %d\n", myid, i) )\
        } else {\
            if ( cu_temp_att.type == 0 ) {\
                printf("%s", s0);\
                printf("\tmyid = %d\ti = %d\n", myid, i);\
            }  \
        }\
    }
    
typedef struct pico_error_info{
  const char* file;
  int line;
  int level;
  
  void update (const char * nwf, int nwl, int nwlev=0) {
      file = nwf;
      line = nwl;
      level = nwlev;
  }
} pico_error_info;

typedef struct pico_compare_cnt{
  const char** files;
  int* lines;
  int* total_tests;
  int* positive_tests;
  int len;
  int used_elements;
  
  void init (int number_of_elements) {
      len = number_of_elements;
      files = (const char**) malloc(sizeof(const char*) * (number_of_elements+1) );
      lines = (int*) malloc(sizeof(int) * (number_of_elements+1));
      total_tests = (int*) malloc(sizeof(int) * (number_of_elements+1));
      positive_tests = (int*) malloc(sizeof(int) * (number_of_elements+1));
      
      for (int i=0; i<number_of_elements; i++) {
            lines[i] = 0;
            files[i] = NULL;
            total_tests[i] = 0;
            positive_tests[i] = 0;
      }
      used_elements = 0;
      lines[number_of_elements] = -1;
      files[number_of_elements] = "ERROR elements";
      total_tests[number_of_elements] = 0;
      positive_tests[number_of_elements] = 0;
  }
  
  int get_element (const char* file, int line) {
      for(int i=0; i<used_elements; i++) {
        if ( (files[i] == file) && (lines[i] == line) )
            return(i);
      }
      
      if (used_elements >= len) {
          printf("ERROR in pico_compare_cnt\n");
          return(len);
      } else {
          files[used_elements] = file;
          lines[used_elements] = line;
          used_elements ++;
          return (used_elements - 1);
      }
  }
  
  void update (int element, bool positivity) {
      total_tests[element]++;
      if (positivity)
          positive_tests[element]++;
  }
  
  void print (void) {
      _MPI_ENV;
      for (int j=0; j<nprocs; j++) {
        if (j == myid) {
            printf("                \x1b[36mProces %2d\x1b[0m                   | total tests | positive tests | negative tests |\n", myid);
            for (int i=0; i<=len; i++) {
                if (files[i] != NULL)
                    printf("%30s at line %4d:      %5d        \x1b[32m%5d\x1b[0m         \x1b[31m%5d\x1b[0m\n", files[i], lines[i], total_tests[i], positive_tests[i], total_tests[i] - positive_tests[i]);
                else
                    printf("NULL\n");
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
  }
} pico_compare_cnt;

#define PICO_COMPARE_CNT_EL( name, file, line, x ) { \
    pico_info.update(__FILE__, __LINE__); \
    int element = name.get_element( file, line); \
    name.update(element, x); \
}

template <typename T>
struct pico_vector_compare {
    vector<T>* vec1;
    vector<T>* vec2;
    int len;
    
    void init (int compare_len) {
        vec1 = NULL;
        vec2 = NULL;
        len = compare_len;
    }
    
    void reinit (int compare_len) {
        assert( vec1 == NULL && vec2 == NULL );
        len = compare_len;
    }
    
    void add_vec (vector<T>* v, int v_shift = 0) {
        if (vec1 == NULL || vec2 == NULL) {
            vector<T>* temp = Vector::init<T>(len, false, v->on_the_device);
            temp->val = v->val + v_shift;
            
            if(vec1 == NULL)
                vec1 = Vector::clone(temp);
            else
                vec2 = Vector::clone(temp);
            std::free(temp);
        }else{
            printf("ERROR: %s %d\n", __FILE__, __LINE__);
        }
        return;
    }
    
    void add_vec (T* v, int v_shift = 0) {
        if (vec1 == NULL || vec2 == NULL) {
            vector<T>* temp = Vector::init<T>(len, false, false); // NOTE: v must be on the host
            temp->val = v + v_shift;
            
            if(vec1 == NULL)
                vec1 = Vector::clone(temp);
            else
                vec2 = Vector::clone(temp);
            std::free(temp);
        }else{
            printf("ERROR: %s %d\n", __FILE__, __LINE__);
        }
        return;
    }
    
    bool result (void) {
        bool out = false;
        extern struct pico_error_info pico_info;
        if (vec1 != NULL && vec2 != NULL) {
            out = Vector::equals(vec1, vec2);
//             if (out == false) {
//                 printf("pico_vector_compare FALSE resulti in %s at line %d\n", pico_info.file, pico_info.line);
// //                 printf("Vec1: "); Vector::print(vec1);
// //                 printf("Vec2: "); Vector::print(vec2);
//             }
            Vector::free(vec1);
            Vector::free(vec2);
            vec1 = NULL;
            vec2 = NULL;
        } else {
            printf("ERROR: %s %d\n", __FILE__, __LINE__);
        }
        return (out);
    }
    
};

template <typename T>
bool compare_vector_collection(int k, boot *boot_amg, vectorCollection<T>* V_global, vectorCollection<T>* W_local) {
    extern struct pico_error_info pico_info;
//     printf("In compare_vector_collection from %s %d\n", pico_info.file, pico_info.line);
    hierarchy *hrrc = boot_amg->H_array[k];
    assert( V_global->n == W_local->n && V_global->n > 0 );
    CSR* A;
    for(int i=0; i<V_global->n; i++) {
        A = hrrc->A_array[i];
//         printf("[level %d] V_global->val[i]->n = %d, A->full_n = %d\n", i, V_global->val[i]->n, A->full_n);
        assert( V_global->val[i]->n == A->full_n );
//         printf("[level %d] W_local->val[i]->n = %d, A->n = %d\n", i, W_local->val[i]->n, A->n);
        assert( W_local->val[i]->n == A->n );
    }
    
    _MPI_ENV; // for PICO_PRINT
    bool flag = true;
    int n = V_global->n;
    struct pico_vector_compare<T> compare;
    
    for(int i=0; i<n; i++) {
        A = hrrc->A_array[i];
        (i==0) ? compare.init(A->n) : compare.reinit(A->n);
        compare.add_vec(V_global->val[i], A->row_shift);
        compare.add_vec(W_local->val[i]);
        bool compare_result = compare.result();
        flag = flag && compare_result;
        if (compare_result != true) {
//             PICO_PRINT(
//                 fprintf(fp, "compare_vector_collection FALSE: test in %s at line %d at level %d\n", pico_info.file, pico_info.line, i);
//                 fprintf(fp, "A->full_n = %d, A->n = %d, A->row_shift = %d\n", A->full_n, A->n, A->row_shift);
// //                 Vector::print(V_global->val[i], -1, fp);
// //                 Vector::print(W_local->val[i], -1, fp);
                
//                 fprintf(fp, "Reupdate W_local with V_global... \t");
//                 fflush(fp);
                printf("Reupdate W_local with V_global... \t");
                vector<T>* temp = W_local->val[i];
                W_local->val[i] = Vector::localize_global_vector(V_global->val[i], A->n, A->row_shift);
                Vector::free(temp);
                compare.reinit(A->n);
                compare.add_vec(V_global->val[i], A->row_shift);
                compare.add_vec(W_local->val[i]);
                if (compare.result() != true) {
//                     fprintf(fp, "ERROR in reupdate");
                    printf("ERROR in reupdate (%s %d)\n", pico_info.file, pico_info.line);
                    exit(0);
                } else {
//                     fprintf(fp, "DONE!");
                    printf("DONE!\n");
                }
//                 fprintf(fp, "\n")
//             )
//             exit(0);
        }
    }
        
//     if (flag == true) {
//         PICO_PRINT( fprintf(fp, "Ok in %s at line %d\n", pico_info.file, pico_info.line) )
//     }
    return (flag);
}

// --------------------------------------------------------------------------------------------------------------------------------------

void print_bit (int b, int mask, int index, FILE* fp = stdout);
  
void print_bitabit(vector<int> *b, FILE* fp = stdout);

void print_row_to_get_info (CSR* Alocal, int nprocs, FILE* fp = stdout);

void print_mask_permut_result(vector<itype> *v, int n_=-1, FILE* fp = stdout);

CSR* complete_Plocal(CSR *Alocal, CSR *Plocal, FILE* fp = stdout);

bool test_mask_permut(CSR *A, FILE* fp = stdout);

bool test_mask_permut_and_shrink(CSR *A, FILE* fp = stdout);

#endif
