#include "utility/assignDeviceToProcess.h"
#include "utility/memory.h"
#include "utility/mpi.h"
#include "utility/utils.h"

#define MPI 1

static int stringCmp(const void* a, const void* b)
{
    return strcmp((const char*)a, (const char*)b);
}

int assignDeviceToProcess()
{
#ifdef MPI
    char host_name[MPI_MAX_PROCESSOR_NAME];
    char(*host_names)[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm nodeComm;
#else
    char host_name[20];
#endif
    int myrank;
    int gpu_per_node;
    int n, namelen, color, rank, nprocs;
    size_t bytes;
#ifdef MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Get_processor_name(host_name, &namelen);
    bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    host_names = (char(*)[MPI_MAX_PROCESSOR_NAME])malloc(bytes);
    CHECK_HOST(host_names);
    strcpy(host_names[rank], host_name);
    for (n = 0; n < nprocs; n++) {
        MPI_Bcast(&(host_names[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n,
            MPI_COMM_WORLD);
    }
    qsort(host_names, nprocs, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);
    color = 0;
    for (n = 0; n < nprocs; n++) {
        if (n > 0 && strcmp(host_names[n - 1], host_names[n])) {
            color++;
        }
        if (strcmp(host_name, host_names[n]) == 0) {
            break;
        }
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
    MPI_Comm_rank(nodeComm, &myrank);
    MPI_Comm_size(nodeComm, &gpu_per_node);
#else
    return 0;
#endif
    return myrank;
}
