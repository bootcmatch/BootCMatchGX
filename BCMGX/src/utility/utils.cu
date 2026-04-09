#include "utility/setting.h"
#include "utility/utils.h"

bool trace_enabled = 0;

// void check_free_memory(int myid){
//     size_t free_mem, total_mem;
//     CHECK_DEVICE( cudaMemGetInfo( &free_mem, &total_mem ))
//     printf("[MEMORY proc %d] Free: %zu - Total: %zu - Allocated: %zu (%zu MByte)\n", myid, free_mem, total_mem, total_mem-free_mem, (total_mem-free_mem)/(1024*1024));
// }

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
        DIE("Error in file %s at line %d: block size cannot be 0\n", file, line);
    }

    if (nb == 0) {
        DIE("Error in file %s at line %d: grid size cannot be 0\n", file, line);
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
        DIE("[ERROR CUBLAS] :\n\t%s\n", err_str);
    }
}

PointerType getPointerType(void* ptr)
{
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

#if CUDART_VERSION >= 10000
    if (err == cudaSuccess) {
        if (attributes.type == cudaMemoryTypeDevice) {
            return PointerType::DEVICE;
        } else if (attributes.type == cudaMemoryTypeHost) {
            return PointerType::HOST;
        } else if (attributes.type == cudaMemoryTypeManaged) {
            return PointerType::MANAGED;
        } else {
            return PointerType::UNKNOWN;
        }
    } else {
        DIE("%s\n", cudaGetErrorString(err));
    }
#else
    if (err == cudaSuccess) {
        if (attributes.memoryType == cudaMemoryTypeDevice) {
            return PointerType::DEVICE;
        } else if (attributes.memoryType == cudaMemoryTypeHost) {
            return PointerType::HOST;
        } else {
            return PointerType::UNKNOWN;
        }
    } else {
        DIE("%s\n", cudaGetErrorString(err));
    }
#endif
}

void printPointerInfo(FILE* out, void* ptr, const char* ptrName, const char* file, int line)
{
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err == cudaSuccess) {
#if CUDART_VERSION >= 10000
        if (attributes.type == cudaMemoryTypeDevice) {
            fprintf(out, "Device pointer found at %s:%d - %s.\n", file, line, ptrName);
        } else if (attributes.type == cudaMemoryTypeHost) {
            fprintf(out, "Host pointer found at %s:%d - %s.\n", file, line, ptrName);
        } else if (attributes.type == cudaMemoryTypeManaged) {
            fprintf(out, "Managed (shared) pointer found at %s:%d - %s.\n", file, line, ptrName);
        } else {
            fprintf(out, "Unknown pointer type at %s:%d - %s\n", file, line, ptrName);
        }
#else
        if (attributes.memoryType == cudaMemoryTypeDevice) {
            fprintf(out, "Device pointer found at %s:%d - %s.\n", file, line, ptrName);
        } else if (attributes.memoryType == cudaMemoryTypeHost) {
            fprintf(out, "Host pointer found at %s:%d - %s.\n", file, line, ptrName);
        } else {
            fprintf(out, "Unknown pointer type at %s:%d - %s\n", file, line, ptrName);
        }
#endif
    } else {
        DIE("%s\n", cudaGetErrorString(err));
    }
}

bool vtypeEq(const vtype &a, const vtype &b) {
    const double tol = 1e-12;

    return fabs(a - b) <= tol;
}
