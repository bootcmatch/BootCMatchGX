/**
 * @file newoverlap.cu
 * @brief Implements the detection and handling of overlapping rows in a sparse matrix (CSR format).
 * 
 * This module identifies rows in a sparse matrix that reference elements outside 
 * a given local domain, distinguishing between "local" and "needy" rows. CUDA 
 * parallelism is utilized for efficient row classification.
 */
   
#include "datastruct/CSR.h"
#include "datastruct/scalar.h"
#include "datastruct/vector.h"
#include "halo_communication/newoverlap.h"
#include "utility/memory.h"
#include "utility/utils.h"
#include <cub/cub.cuh>

/**
 * @struct Matched
 * @brief Comparator for filtering values in an array.
 * 
 * This struct provides a device-compatible comparison operation 
 * used in filtering operations with the CUB library.
 */
struct Matched {
    int compare; ///< Value used for comparison.

    /**
     * @brief Constructor for Matched struct.
     * @param compare Value to compare against.
     */   
    __host__ __device__ __forceinline__
    Matched(int compare)
        : compare(compare)
    {
    }

    /**
     * @brief Overloaded operator() for filtering.
     * @param a Input value to compare.
     * @return true if a does not match the stored comparison value.
     */
    __host__ __device__ __forceinline__ bool operator()(const int& a) const
    {
        return (a != compare);
    }
};

/**
 * @brief Initializes an overlappedList structure.
 * @return Pointer to the newly allocated overlappedList structure.
 */
overlappedList* init_overlappedList()
{
    overlappedList* osl = MALLOC(overlappedList, 1);

    osl->local_stream = MALLOC(cudaStream_t, 1);
    osl->comm_stream = MALLOC(cudaStream_t, 1);

    CHECK_DEVICE(cudaStreamCreate(osl->local_stream));
    CHECK_DEVICE(cudaStreamCreate(osl->comm_stream));

    return osl;
}

/**
 * @brief Frees an overlappedList structure.
 * @param osl Pointer to the overlappedList structure to be freed.
 */
   
void free_overlappedList(overlappedList* osl)
{
    CHECK_DEVICE(cudaStreamDestroy(*osl->local_stream));
    CHECK_DEVICE(cudaStreamDestroy(*osl->comm_stream));

    FREE(osl);
}

/**
 * @brief Retrieves a globally shared overlappedList instance.
 * @return Pointer to the global overlappedList instance.
 */
overlappedList* getGlobalOverlappedList()
{
    static overlappedList* osl = init_overlappedList();
    return osl;
}

/**
 * @brief CUDA kernel to find "needy" rows in a sparse matrix.
 * 
 * Identifies rows that contain column indices outside the local domain.
 * 
 * @param n Number of rows in the matrix.
 * @param MINI_WARP_SIZE Size of the mini warp for processing.
 * @param A_row Row pointer array of the CSR matrix.
 * @param A_col Column indices array of the CSR matrix.
 * @param needy_rows Output array storing indices of needy rows.
 * @param loc_rows Output array storing indices of local rows.
 * @param needy_n Output counter for needy rows.
 * @param loc_n Output counter for local rows.
 * @param start Start index of the local domain.
 * @param end End index of the local domain.
 */  
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

/**
 * @brief Sets up overlapped regions in a CSR matrix.
 * 
 * This function categorizes matrix rows into "local" and "needy" categories
 * based on whether they reference external indices.
 * 
 * @param A Pointer to the CSR matrix.
 */  
void setupOverlapped(CSR* A)
{
    overlappedList* streams = getGlobalOverlappedList();

    _MPI_ENV;

    struct overlapped* os = &(A->os);
    os->streams = MALLOC(struct overlappedList, 1);
    os->streams->local_stream = streams->local_stream;
    os->streams->comm_stream = streams->comm_stream;

    vector<itype>* loc_rows = Vector::init<itype>(A->n, true, true);
    Vector::fillWithValue(loc_rows, -1);

    vector<itype>* needy_rows = Vector::init<itype>(A->n, true, true);
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
        os->loc_rows = Vector::init<itype>(os->loc_n, true, true);
    }
    if (os->needy_n) {
        os->needy_rows = Vector::init<itype>(os->needy_n, true, true);
    }

    Matched m(-1);
    scalar<itype>* d_num_selected_out = Scalar::init<itype>(0, true);
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    if (os->loc_n) {
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, loc_rows->val, os->loc_rows->val, d_num_selected_out->val, A->n, m);
        // Allocate temporary storage
        d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes, false);
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, loc_rows->val, os->loc_rows->val, d_num_selected_out->val, A->n, m);
    }

    if (os->needy_n) {
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, needy_rows->val, os->needy_rows->val, d_num_selected_out->val, A->n, m);
        // Allocate temporary storage
        if (d_temp_storage == NULL) {
            d_temp_storage = CUDA_MALLOC_BYTES(void, temp_storage_bytes, false);
        }
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, needy_rows->val, os->needy_rows->val, d_num_selected_out->val, A->n, m);
    }

    CUDA_FREE(d_temp_storage);
    Scalar::free(d_num_selected_out);
    Scalar::free(loc_n);
    Scalar::free(needy_n);
    FREE(_loc_n);
    FREE(_needy_n);
    Vector::free(loc_rows);
    Vector::free(needy_rows);
}

