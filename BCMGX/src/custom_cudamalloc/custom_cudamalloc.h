#ifndef CUSTOM_CUDAMALLOC
#define CUSTOM_CUDAMALLOC

#include "utility/setting.h"
#include <assert.h>
#include <stdlib.h>

typedef struct custom_cudamalloc {
    itype* itype_ptr;
    int itype_allocated;
    int itype_occupied;

    vtype* vtype_ptr;
    int vtype_allocated;
    int vtype_occupied;

} customCudaMalloc;

namespace CustomCudaMalloc {
void init(int n_itypes, int n_vtypes, int buff_num = 0);

itype* alloc_itype(int n_elements, int buff_num = 0);

vtype* alloc_vtype(int n_elements, int buff_num = 0);

void free(int buff_num = 0);
}
#endif
