#include "suitor.h"

#include "op/spspmpi.h"
#include "utility/function_cnt.h" // PICO
#include "utility/utils.h"

#include <cuda.h>
#include <cuda/semaphore>
#include <stdio.h>
#include <stdlib.h>

#define SUITOR_GT(a, b) (a - b) > SUITOR_EPS
#define _TRUE 1
#define _FALSE 0

// number of thread per block, the number of WARP per block is NTHREAD_PER_BLK / 32
#define NTHREAD_PER_BLK 128
// number of nodes assigned to each WARP
#define CHUNK_PER_WARP 16
#define WARP_SIZE 32
#define _mask 0xFFFFFFFF
#if 0
__device__ cuda::binary_semaphore<cuda::thread_scope_device> s(1);
#endif
/*
 *  Method to lock a memory location
 */
__device__ int lock_vertex(volatile int* mutex, int id)
{
    while (atomicCAS((int*)(mutex + id), 0, 1) != 0)
        ;
    __threadfence();
    return 1;
}

/*
 * Method to Unlock a Memory Location
 */
__device__ void unlock_vertex(volatile int* mutex, int id)
{
    __threadfence();
    atomicExch((int*)(mutex + id), 0);
}

__device__ void memcpySIMD_int(int W_OFF, int segment_len, volatile int* dest, int* src)
{

    for (int i = W_OFF; i < segment_len; i += WARP_SIZE) {
        dest[i] = src[i];
    }
}

__device__ void set_suitor(int candidate, double heaviest, int current_vid, int* finished, int* new_vertex_to_care, volatile double* ws_, volatile int* s_, volatile int* d_locks)
{

    *new_vertex_to_care = -696; // a negative integer
    *finished = _TRUE;

    if (heaviest > 0 && candidate >= 0) {
        int next_vertex = -1;
        *finished = _FALSE; /* case when you don't get lock */
        *new_vertex_to_care = current_vid;
        
        lock_vertex(d_locks, candidate);
        if (heaviest >= ws_[candidate]) { // test whether "heaviest" is still larger than previous offer to "candidate"
            next_vertex = s_[candidate]; // save previous suitor of "candidate" as it is un-lodged by "current_vid"
            s_[candidate] = current_vid; // "curernt_vid" become suitor of "candidate"
            ws_[candidate] = heaviest; // set "heaviest" as corresponding weight
            *finished = _TRUE;
            unlock_vertex(d_locks, candidate); // Unlock "candidate"
            if (next_vertex >= 0) { // take responsibility of un-lodged vertex. Find suitor for it.
                current_vid = next_vertex; // now, save un-lodged vertex into "current_vid"
                *finished = _FALSE;
            }
        } else { // can't use the result, try again
            unlock_vertex(d_locks, candidate); // unlock "candidate"
            *finished = _FALSE;
        }
        *new_vertex_to_care = current_vid; // 'current_vid' either contains original "current_vid" or previous suitor of "candidate"
    }
}

__device__ void find_Candidate(int W_OFF, int len_neighborlist, int* edge_list, volatile double* ws_, volatile int* s_, double* weight_list, volatile int* d_locks, int current_vid, int* potential_candidate, double* by_weight, int shift, int n)
{

    int y;
    double weight_value;
    double ws_value;
    int candidate = -1; // best candidate to be matched with
    double heaviest = -1.0; // best  weight that can be achieved

    // Each member thread of the warp read the neighbors of  "current_vid" determined by be the offset of the thread in the warp
    for (int j = W_OFF; j < len_neighborlist; j += WARP_SIZE) {

        weight_value = weight_list[j]; // weight of edge to y
        if (weight_value <= 0) {
            continue;
        }
        y = edge_list[j]; // y is current neighbor
        y = y - shift;

        if (y < 0 || y >= n) {
            continue;
        }

        ws_value = ws_[y]; // current offer to y

        if ((weight_value < heaviest) || (weight_value < ws_value)) {
            continue;
        }

        if ((weight_value == heaviest) && (y < candidate)) { /* candidate with higher id is consider to be best if there is tie on weight*/
            continue;
        }

        if ((weight_value == ws_[y])) {
            int successful = 0; // you can use successful for the rule of skip as well
            lock_vertex(d_locks, y);
            if ((weight_value == ws_[y]) && (current_vid < s_[y])) { /* y always prefers candidate with higher id*/
                successful = 1;
            }
            unlock_vertex(d_locks, y);
            successful += 1;
            //                }
            //            }
            if (successful == 2) {
                continue;
            }
        }
        heaviest = weight_value;
        candidate = y;
    }

    // A reduction across the warp to find best candidate
    // Use __shfl_xor to perform butterfly reduction
    for (int i = WARP_SIZE / 2; i >= 1; i /= 2) {
        double rec_value = __shfl_xor_sync(_mask, heaviest, i, WARP_SIZE); // double
        int rec_vertex = __shfl_xor_sync(_mask, candidate, i, WARP_SIZE); // integer
        if (rec_value > heaviest) {
            heaviest = rec_value;
            candidate = rec_vertex;
        } else if (rec_value == heaviest) { // give priority to higher indexed candidate
            if (rec_vertex > candidate) {
                candidate = rec_vertex;
                heaviest = rec_value;
            }
        }
    }
    // At this point every thread of the warp knows the  best "candidate" to match with "current_vid"
    *potential_candidate = candidate;
    *by_weight = heaviest;
    return;
}

__device__ void collectUnsucc(int laneId, volatile int* warpmem, int* nr_of_unsuccess, int isfinished, int new_vertex_to_care)
{

    int value = 1 - isfinished;

    for (int i = 1; i <= WARP_SIZE / 2; i *= 2) {
        int lowerLaneValue = __shfl_up_sync(_mask, value, i, WARP_SIZE);
        if (laneId >= i) {
            value += lowerLaneValue;
        }
    }
    // write in shared memory
    if (isfinished == 0) {
        warpmem[CHUNK_PER_WARP - (*nr_of_unsuccess) - (value - 1)] = new_vertex_to_care;
    }
    *nr_of_unsuccess += __shfl_sync(_mask, value, WARP_SIZE - 1, WARP_SIZE);
}

__global__ void kernel_for_matching(int n, int* d_indices, volatile int* suitors, volatile double* ws, int* d_edges, double* d_weights, volatile int* d_locks, int shift)
{

    extern __shared__ int blockmem[];
    int laneId = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5; // local warp id
    int* myMemory = blockmem + (CHUNK_PER_WARP + 1) * wid; // each warp get memory of size (CHUNK_PER_WARP+1)
    volatile int* warpmem = myMemory;

    int* neighbor_list = NULL;
    double* related_weights = NULL;
    int num_neighbors = 0;

    int saved_vertex = -1;
    int saved_candidate = -1;
    double saved_weight = -5.0;

    int candidate = -1;
    double by_weight = -5.0;

    int isfinished = _TRUE;

    int acc = 0;
    int nr_of_unsuccess = 0;

    wid = blockIdx.x * (NTHREAD_PER_BLK / WARP_SIZE) + wid; // global warp id

    num_neighbors = n - wid * CHUNK_PER_WARP; // counts #vertex to work with : reduce register usage

    if ((wid + 1) * CHUNK_PER_WARP <= n) {
        num_neighbors = CHUNK_PER_WARP;
    }

    if (num_neighbors > 0) {
        memcpySIMD_int(laneId, num_neighbors + 1, warpmem, &d_indices[wid * CHUNK_PER_WARP]);
    }

    int nr_element_to_read = num_neighbors;

    for (int vid = nr_element_to_read - 1; vid >= 0; vid--) {
        num_neighbors = warpmem[vid + 1] - warpmem[vid]; // d_indices[vid+1]-d_indices[vid];
        neighbor_list = &d_edges[warpmem[vid]]; //&d_edges[d_indices[vid]];
        related_weights = &d_weights[warpmem[vid]]; //&d_weights[d_indices[vid]];

        find_Candidate(laneId, num_neighbors, neighbor_list, ws, suitors, related_weights, d_locks, vid + wid * CHUNK_PER_WARP, &candidate, &by_weight, shift, n);

        if ((acc % WARP_SIZE) == laneId) {
            saved_vertex = vid + wid * CHUNK_PER_WARP; // save vertex
            saved_candidate = candidate;
            saved_weight = by_weight;
        }

        acc++;

        if ((acc % WARP_SIZE) == 0) {
            set_suitor(saved_candidate, saved_weight, saved_vertex, &isfinished, &saved_candidate, ws, suitors, d_locks);
            collectUnsucc(laneId, warpmem, &nr_of_unsuccess, isfinished, saved_candidate);
        }
    }

    isfinished = _TRUE;
    if (laneId < (acc % WARP_SIZE)) {
        set_suitor(saved_candidate, saved_weight, saved_vertex, &isfinished, &saved_candidate, ws, suitors, d_locks);
    }

    collectUnsucc(laneId, warpmem, &nr_of_unsuccess, isfinished, saved_candidate);

    do {

        int prev_nr_of_unsuccess = nr_of_unsuccess;
        nr_of_unsuccess = 0;
        for (int k = 0; k < prev_nr_of_unsuccess; ++k) {

            int new_vertex = warpmem[CHUNK_PER_WARP - k]; // fetch the vertex from shared memory
            neighbor_list = &d_edges[d_indices[new_vertex]];
            related_weights = &d_weights[d_indices[new_vertex]];
            num_neighbors = d_indices[new_vertex + 1] - d_indices[new_vertex];

            candidate = -1;
            by_weight = -5.0;
            find_Candidate(laneId, num_neighbors, neighbor_list, ws, suitors, related_weights, d_locks, new_vertex, &candidate, &by_weight, shift, n);

            if ((k % WARP_SIZE) == laneId) {
                saved_candidate = candidate;
                saved_vertex = new_vertex;
                saved_weight = by_weight;
            }
            if (((k + 1) % WARP_SIZE) == 0) {
                set_suitor(saved_candidate, saved_weight, saved_vertex, &isfinished, &saved_candidate, ws, suitors, d_locks);
                collectUnsucc(laneId, warpmem, &nr_of_unsuccess, isfinished, saved_candidate);
            }
        }

        isfinished = _TRUE;
        if (laneId < (prev_nr_of_unsuccess % WARP_SIZE)) {
            set_suitor(saved_candidate, saved_weight, saved_vertex, &isfinished, &saved_candidate, ws, suitors, d_locks);
        }
        collectUnsucc(laneId, warpmem, &nr_of_unsuccess, isfinished, saved_candidate);

    } while (nr_of_unsuccess != 0);
}

double cost_matching(int tot_vertices, int64_t tot_edges, int n, int* ver, int* edges, double* weight, int* match)
{
    double glob_sum = 0.0;
    int i, k;
    int64_t tot_card = 0;
    for (i = 0; i < n; i++) {
        if ((match[i] >= 0)) {
            for (k = ver[i]; k < ver[i + 1]; k++) { // Loop through neighbors of vertex i
                if (edges[k] == match[i]) {
                    tot_card++;
                    glob_sum += (double)weight[k];
                    if (match[edges[k]] != i) {
                        printf("\n*******Error : %d is offering %d with weight %f but %d is offering %d  \n", match[i], i, weight[k], match[edges[k]], edges[k]);
                    }
                } // if
            } // for k
        } // if
    } // for i

    printf("Total Match=%ld where #Vertex: %d #Edge: %ld  \n", tot_card, tot_vertices, tot_edges);
    return (glob_sum);
}

__global__ void _fix_shift_matching(int n, int* M, int shift)
{
    itype i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (M[i] != -1) {
        M[i] += shift;
    }
}

vector<int>* approx_match_gpu_suitor(handles* h, CSR* A, CSR* W, vector<itype>* M, vector<double>* ws, vector<int>* mutex)
{

    _MPI_ENV;
    assert(W->on_the_device);
    int n = W->n;

    vector<int>* _bitcol = NULL;
    _bitcol = get_missing_col(A, NULL);
    Vector::fillWithValue(M, -1);
    Vector::fillWithValue(ws, 0.0);
    Vector::fillWithValue(mutex, 0);

    //
    int load_per_blk = CHUNK_PER_WARP * (NTHREAD_PER_BLK / WARP_SIZE); // NTHREAD_PER_BLK is multiple of 32

    int nr_of_block = (n + load_per_blk - 1) / load_per_blk;

    int shared_memory_size_per_block = (CHUNK_PER_WARP + 1) * (NTHREAD_PER_BLK / WARP_SIZE) * sizeof(int);
    // get dev_bit col to pass to compute_rows. This is sync
    if (0) {
        fprintf(stderr, "Task %d reached line %d in suitor (%s)\n", myid, __LINE__, __FILE__);
    }

    kernel_for_matching<<<nr_of_block, NTHREAD_PER_BLK, shared_memory_size_per_block>>>(n, W->row, M->val, ws->val, W->col, W->val, mutex->val, 0 /* W->row_shift */);

    compute_rows_to_rcv_CPU(A, NULL, _bitcol);
    if (0) {
        fprintf(stderr, "Task %d reached line %d in suitor (%s)\n", myid, __LINE__, __FILE__);
    }
    Vector::free(_bitcol);

    cudaDeviceSynchronize();
    if (0) {
        fprintf(stderr, "Task %d reached line %d in suitor (%s)\n", myid, __LINE__, __FILE__);
    }
    return M;
}

template <typename T>
T* makeArray(int size)
{
    T* a = (T*)Malloc(sizeof(T) * size);
    assert(a != NULL);
    return a;
}

vector<int>* approx_match_gpu_suitor_v0(CSR* W, vector<itype>* M, vector<double>* ws, vector<int>* mutex)
{

    assert(W->on_the_device);
    int n = W->n;

    Vector::fillWithValue(M, -1);
    Vector::fillWithValue(ws, 0.0);
    Vector::fillWithValue(mutex, 0);

    //
    int load_per_blk = CHUNK_PER_WARP * (NTHREAD_PER_BLK / WARP_SIZE); // NTHREAD_PER_BLK is multiple of 32

    int nr_of_block = (n + load_per_blk - 1) / load_per_blk;

    int shared_memory_size_per_block = (CHUNK_PER_WARP + 1) * (NTHREAD_PER_BLK / WARP_SIZE) * sizeof(int);
    // get dev_bit col to pass to compute_rows. This is sync

    if (1) {
        fprintf(stderr, "Task 0 reached line %d in suitor (%s)\n", __LINE__, __FILE__);
    }
    kernel_for_matching<<<nr_of_block, NTHREAD_PER_BLK, shared_memory_size_per_block>>>(n, W->row, M->val, ws->val, W->col, W->val, mutex->val, 0 /* W->row_shift */);

    cudaDeviceSynchronize();
    if (1) {
        fprintf(stderr, "Task 0 reached line %d in suitor (%s)\n", __LINE__, __FILE__);
    }
    return M;
}

template <typename T>
vector<int>* approx_match_cpu_suitor(CSR* W_)
{

    CSR* W = CSRm::copyToHost(W_);
    int* row = W->row;
    int* col = W->col;
    T* val = W->val;
    int n = W->n;

    // prepare
    vector<int>* suitor = Vector::init<int>(n, true, false);

    T* ws = makeArray<T>(n);

    std::cout << "its me\n";

    for (int i = 0; i < n; i++) {
        suitor->val[i] = -1;
        ws[i] = -1;
    }

    // algorithm
    for (int i = 0; i < n; i++) {
        int u = i;
        int current = u;
        bool done = false;

        while (!done) {
            int partner = suitor->val[current];
            T heaviest = ws[current];
            for (int j = row[current]; j < row[current + 1]; j++) {
                int v = col[j];
                if (SUITOR_GT(val[j], heaviest) && SUITOR_GT(val[j], ws[v])) {
                    partner = v;
                    heaviest = val[j];
                }
            }

            done = true;

            if (heaviest != -1) {
                int y = suitor->val[partner];
                suitor->val[partner] = current;
                ws[partner] = heaviest;
                if (y != -1) {
                    current = y;
                    done = false;
                }
            }
        }
    }
    free(ws);

    CSRm::free(W);

    VectorcopyToDevice_CNT return Vector::copyToDevice(suitor);
}

template <typename T>
vector<int>* approx_match_cpu_suitor_LOCAL(CSR* W_)
{
    _MPI_ENV;

    CSR* W = CSRm::copyToHost(W_);
    int* row = W->row;
    int* col = W->col;
    T* val = W->val;
    int n = W->n;

    // prepare
    vector<int>* suitor = Vector::init<int>(n, true, false);

    T* ws = makeArray<T>(n);

    for (int i = 0; i < n; i++) {
        suitor->val[i] = -1;
        ws[i] = -1.;
    }

    int W_start = W->row_shift;
    int W_stop = W->row_shift + W->n;

    // algorithm
    for (int i = 0; i < n; i++) {
        int u = i;
        int current = u;
        bool done = false;

        while (!done) {
            int partner = suitor->val[current];
            T heaviest = ws[current];
            for (int j = row[current]; j < row[current + 1]; j++) {
                int v = col[j];
                v = v - W->row_shift;

                if (v < 0 || v >= n) {
                    continue;
                }

                if (val[j] > heaviest && val[j] > ws[v]) {
                    partner = v;
                    heaviest = val[j];
                }
            }

            done = true;

            if (heaviest != -1) {
                int y = suitor->val[partner];
                suitor->val[partner] = current;
                ws[partner] = heaviest;
                if (y != -1) {
                    current = y;
                    done = false;
                }
            }
        }
    }
    free(ws);
    CSRm::free(W);
    VectorcopyToDevice_CNT return Vector::copyToDevice(suitor);
}
