#pragma once

#define CUDAMALLOCCNTON 0

#ifndef NOMPI
#include "utility/myMPI.h"
#endif

typedef struct functionCall_cnt{
  const char** callers;
  const char** calleds;
  int* number_of_calls;
  int len;
  int used_elements;
  
  void init (int number_of_elements) {
      len = number_of_elements;
      callers = (const char**) malloc(sizeof(const char*) * (number_of_elements+1) );
      calleds = (const char**) malloc(sizeof(const char*) * (number_of_elements+1) );
      number_of_calls = (int*) malloc(sizeof(int) * (number_of_elements+1));
      
      for (int i=0; i<number_of_elements; i++) {
            number_of_calls[i] = 0;
            callers[i] = NULL;
            calleds[i] = NULL;
      }
      used_elements = 0;
      number_of_calls[number_of_elements] = 0;
      callers[number_of_elements] = "ERROR elements";
  }
  
  int get_element (const char* caller, const char* called) {
      for(int i=0; i<used_elements; i++) {
        if ( callers[i] == caller && calleds[i] == called )
            return(i);
      }
      
      if (used_elements >= len) {
          printf("ERROR in functionCall_cnt\n");
          return(len);
      } else {
          callers[used_elements] = caller;
          calleds[used_elements] = called;
          used_elements ++;
          return (used_elements - 1);
      }
  }
  
  void update (int element) {
      number_of_calls[element]++;
  }
  
#ifndef NOMPI
  void print (void) {
      _MPI_ENV;
//       for (int j=0; j<nprocs; j++) {
//         if (j == myid) {
        if (ISMASTER) {
            printf("\x1b[36mProces %2d\x1b[0m\n", myid);
            for (int i=0; i<=len; i++) {
                if (callers[i] != NULL)
                    printf("%30s has made %d calls of %s\n", callers[i], number_of_calls[i], calleds[i]);
                else
                    printf("NULL\n");
            }
            printf("\n");
            for (int i=0; i<=len; i++) {
                bool flag = true;
                const char *s = calleds[i];
                
                for (int j=0; j<i && flag; j++)
                    if (calleds[j] == s)
                        flag = false;
                    
                if (flag) {
                    int n = number_of_calls[i];
                    for (int j=(i+1); j<=len; j++)
                        if (calleds[j] == s)
                            n += number_of_calls[j];
                        
                    printf("The function %30s has been called %d times\n", s, n);
                }
            }
            printf("\n");
        }
//         MPI_Barrier(MPI_COMM_WORLD);
//       }
  }
#endif
  
} functionCall_cnt;

#define FUNCTIONCALL_CNT( name, caller_function, called_function ) { \
    int element = name.get_element( caller_function, called_function ); \
    name.update(element); \
}

extern functionCall_cnt cudamalloc_cnt;

#define cudaMalloc_CNT { \
  if (CUDAMALLOCCNTON){ \
    int element = cudamalloc_cnt.get_element( __func__, "cudaMalloc" ); \
    cudamalloc_cnt.update(element); \
  }  \
}

#define Vectorinit_CNT { \
  if (CUDAMALLOCCNTON) { \
    int element = cudamalloc_cnt.get_element( __func__, "Vector::init" ); \
    cudamalloc_cnt.update(element); \
  }  \
}

#define VectorcopyToDevice_CNT { \
  if (CUDAMALLOCCNTON) { \
    int element = cudamalloc_cnt.get_element( __func__, "Vector::copyToDevice" ); \
    cudamalloc_cnt.update(element); \
  }  \
}
