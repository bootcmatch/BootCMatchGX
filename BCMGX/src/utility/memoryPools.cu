#define global_pools 0
#define local_pools 3

namespace MemoryPool{

    void **local;
    void **global;

    void initContext(itype full_n, itype n){
      MemoryPool::local = (void**) malloc(sizeof(void*)*local_pools);
      CHECK_HOST(MemoryPool::local);

      for(int i=0; i<local_pools; i++){
        CHECK_DEVICE( cudaMalloc(&MemoryPool::local[i], sizeof(vtype) * n) );
      }

      MemoryPool::global = (void**) malloc(sizeof(void*)*global_pools);
      CHECK_HOST(MemoryPool::local);
      for(int i=0; i<global_pools; i++){
        CHECK_DEVICE( cudaMalloc(&MemoryPool::global[i], sizeof(vtype) * full_n) );
      }

    }

    void freeContext(){
      for(int i=0; i<local_pools; i++){
        cudaFree(&MemoryPool::local[i]);
      }

      for(int i=0; i<global_pools; i++){
        cudaFree(&MemoryPool::global[i]);
      }
    }
}
