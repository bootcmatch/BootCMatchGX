#include "basic_kernel/halo_communication/newoverlap.h"
#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "utility/function_cnt.h"
#include "utility/utils.h"
#include <cub/cub.cuh>

struct Matched {
    int compare;
    __host__ __device__ __forceinline__
    Matched(int compare)
        : compare(compare)
    {
    }
    __host__ __device__ __forceinline__ bool operator()(const int& a) const
    {
        return (a != compare);
    }
};

overlappedList* init_overlappedList()
{

    overlappedList* osl = (overlappedList*)Malloc(sizeof(overlappedList));
    CHECK_HOST(osl);
#if 0
  osl->nm = nm;

  osl->oss = (overlapped*) Malloc( sizeof(overlapped) * nm );
  CHECK_HOST(osl->oss);
#endif
    osl->local_stream = (cudaStream_t*)Malloc(sizeof(cudaStream_t));
    osl->comm_stream = (cudaStream_t*)Malloc(sizeof(cudaStream_t));

    CHECK_DEVICE(cudaStreamCreate(osl->local_stream));
    CHECK_DEVICE(cudaStreamCreate(osl->comm_stream));

    return osl;
}

void free_overlappedList(overlappedList* osl)
{

#if 0
  for(int i=0; i<osl->nm; i++){
    if(osl->oss[i].loc_n)
      Vector::free(osl->oss[i].loc_rows);

    if(osl->oss[i].needy_n)
      Vector::free(osl->oss[i].needy_rows);
  }

  free(osl->oss);
#endif

    CHECK_DEVICE(cudaStreamDestroy(*osl->local_stream));
    CHECK_DEVICE(cudaStreamDestroy(*osl->comm_stream));

    free(osl);
}

overlappedList* getGlobalOverlappedList()
{
    static overlappedList* osl = init_overlappedList();
    return osl;
}

__global__ void _findNeedyRows(itype n, int MINI_WARP_SIZE, itype* A_row, itype* A_col, itype* needy_rows, itype* loc_rows, itype* needy_n, itype* loc_n, gstype start, gstype end)
{
    itype tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp = tid / MINI_WARP_SIZE;
    if (warp >= n) {
        return;
    }

    int lane = tid % MINI_WARP_SIZE;
    int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
    int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

    itype rows, rowe;
    rows = A_row[warp];
    rowe = A_row[warp + 1];

    int flag = 0;
    for (int j = rows + lane; j < rowe; j += MINI_WARP_SIZE) {
        itype c = A_col[j];
        if (c < start || c >= end) {
            flag = 1;
        }
    }

    unsigned needy = __any_sync(warp_mask, flag);

    if (lane == 0) {
        if (needy) {
            atomicAdd(needy_n, 1);
            needy_rows[warp] = warp;
        } else {
            atomicAdd(loc_n, 1);
            loc_rows[warp] = warp;
        }
    }
}

void setupOverlapped(CSR* A)
{
    overlappedList* streams = getGlobalOverlappedList();

    _MPI_ENV;

    struct overlapped* os = &(A->os);
    os->streams = (struct overlappedList*)Malloc(sizeof(struct overlappedList));
    CHECK_HOST(os->streams);
    os->streams->local_stream = streams->local_stream;
    os->streams->comm_stream = streams->comm_stream;

    Vectorinit_CNT
        vector<itype>* loc_rows
        = Vector::init<itype>(A->n, true, true);
    Vector::fillWithValue(loc_rows, -1);

    Vectorinit_CNT
        vector<itype>* needy_rows
        = Vector::init<itype>(A->n, true, true);
    Vector::fillWithValue(needy_rows, -1);

    scalar<itype>* loc_n = Scalar::init(0, true);
    scalar<itype>* needy_n = Scalar::init(0, true);

    int warpsize = CSRm::choose_mini_warp_size(A);

    GridBlock gb = gb1d(A->n, BLOCKSIZE, true, warpsize);
    if (A->shrinked_flag == false) {
        fprintf(stderr, "A should be shrinked in setupOverlapper\n");
        exit(1);
    }
    _findNeedyRows<<<gb.g, gb.b>>>(A->n, warpsize, A->row, A->shrinked_col, needy_rows->val, loc_rows->val, needy_n->val, loc_n->val, A->post_local, A->post_local + A->n);

    itype* _loc_n = Scalar::getvalueFromDevice(loc_n);
    itype* _needy_n = Scalar::getvalueFromDevice(needy_n);

    os->loc_n = *_loc_n;
    os->needy_n = *_needy_n;

    if (os->loc_n) {
        Vectorinit_CNT
            os->loc_rows
            = Vector::init<itype>(os->loc_n, true, true);
    }
    if (os->needy_n) {
        Vectorinit_CNT
            os->needy_rows
            = Vector::init<itype>(os->needy_n, true, true);
    }

    Matched m(-1);
    scalar<itype>* d_num_selected_out = Scalar::init<itype>(0, true);
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    if (os->loc_n) {
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, loc_rows->val, os->loc_rows->val, d_num_selected_out->val, A->n, m);
        // Allocate temporary storage
        cudaMalloc_CNT
            CHECK_DEVICE(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, loc_rows->val, os->loc_rows->val, d_num_selected_out->val, A->n, m);
    }

    if (os->needy_n) {
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, needy_rows->val, os->needy_rows->val, d_num_selected_out->val, A->n, m);
        // Allocate temporary storage
        if (d_temp_storage == NULL) {
            cudaMalloc_CNT
                CHECK_DEVICE(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        }
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, needy_rows->val, os->needy_rows->val, d_num_selected_out->val, A->n, m);
    }

    cudaFree(d_temp_storage);
    Scalar::free(d_num_selected_out);
    Scalar::free(loc_n);
    Scalar::free(needy_n);
    free(_loc_n);
    free(_needy_n);
    Vector::free(loc_rows);
    Vector::free(needy_rows);
}
