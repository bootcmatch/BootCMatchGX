#include "utility/setting.h"
#include "utility/utils.h"

void* Realloc(void* pptr, size_t sz)
{
    if (!sz) {
        printf("Allocating zero bytes...\n");
        exit(EXIT_FAILURE);
    }
    void* ptr = (void*)realloc(pptr, sz);
    if (!ptr) {
        fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void* Malloc(size_t sz)
{
    if (!sz) {
        printf("Allocating zero bytes...\n");
        exit(EXIT_FAILURE);
    }
    void* ptr = (void*)malloc(sz);
    if (!ptr) {
        fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
        exit(EXIT_FAILURE);
    }
    memset(ptr, 0, sz);
    return ptr;
}

namespace Eval {
void printMetaData(const char* name, double value, int type)
{
    printf("#META %s ", name);
    if (type == 0) {
        int value_int = (int)value;
        printf("int %d", value_int);
    } else if (type == 1) {
        printf("float %le", value);
    }
    printf("\n");
}
}

GridBlock gb1d(const unsigned n, const unsigned block_size, const bool is_warp_agg, int MINI_WARP_SIZE)
{
    GridBlock gb;

    int n_ = n;
    if (n == 0) {
        gb.b = 0;
        gb.g = 0;
        return gb;
    }
    if (is_warp_agg) {
        n_ *= MINI_WARP_SIZE;
    }

    dim3 block(block_size);
    dim3 grid((n_ + (block.x - 1)) / block.x);

    gb.b = block;
    gb.g = grid;
    return gb;
}

// =============================================================================

GridBlock _getKernelParams(int desiredThreads, const char* file, int line)
{
    GridBlock gb;

    int nb = 1;
    int nt = desiredThreads;
    if (nt > MAX_THREADS) {
        nb = nt / MAX_THREADS;
        if (nt % MAX_THREADS) {
            nb++;
        }
        nt = MAX_THREADS;
    }

    if (nt == 0) {
        fprintf(stderr, "Error in file %s at line %d: block size cannot be 0\n", file, line);
        exit(1);
    }

    if (nb == 0) {
        fprintf(stderr, "Error in file %s at line %d: grid size cannot be 0\n", file, line);
        exit(1);
    }

    gb.g = nb;
    gb.b = nt;

    return gb;
}

// =============================================================================

cudaMemcpyKind getMemcpyKind(bool dstOnDevice, bool srcOnDevice)
{
    if (dstOnDevice) {
        return srcOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    } else {
        return srcOnDevice ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
    }
}

// =============================================================================

const char* cublasGetStatusString(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "CUBLAS_STATUS_UNKNOWN_ERROR";
}

void CHECK_CUBLAS(cublasStatus_t err)
{
    const char* err_str = cublasGetStatusString(err);
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf("[ERROR CUBLAS] :\n\t%s\n", err_str);
        exit(1);
    }
}
