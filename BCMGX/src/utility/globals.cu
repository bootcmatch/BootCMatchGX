#include "op/spspmpi.h"
#include "utility/assignDeviceToProcess.h"
#include "utility/die.h"
#include "utility/mpi.h"
#include "utility/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
// #include <nsparse.h>
#include <unistd.h>

#define DEFAULTSCALENNZMISS 64
int xsize = 0;
double* xvalstat = NULL;
int* taskmap = NULL;
int* itaskmap = NULL;
int scalennzmiss = DEFAULTSCALENNZMISS;
char idstring[128];
FILE* log_file = NULL;
std::string output_dir = "./";
std::string output_prefix = "";
std::string output_suffix = "";

// void release_bin(sfBIN bin);
// extern sfBIN global_bin;

extern double CUSPARSE_CHUNK_FRACTION;

void spgemmcusparseFree();

int get_mapped_task(int i)
{
    return taskmap ? taskmap[i] : i;
}

void close_log_file()
{
    if (log_file) {
        fflush(log_file);
        fclose(log_file);
    }
}

void open_log_file(int myid, const char* log_filename)
{
    if (log_filename) {
        char filename[255] = { 0 };
        sprintf(filename, "%s_%d", log_filename, myid);
        log_file = fopen(filename, "w");
        if (log_file == NULL) {
            DIE("Error opening file <%s>\n", filename);
        }
        if (setvbuf(log_file, NULL, _IOLBF, 0)) { // _IONBF unbuffered, _IOLBF line buffered, _IOFBF fully buffered
            DIE("Cannot change buffer mode on file <%s>\n", filename);
        }
        if (atexit(close_log_file)) {
            DIE("Error registering atexit\n");
        }
        fprintf(log_file, "MPI rank: %d\n", myid);
        fprintf(log_file, "PID: %d\n", getpid());
    }
}

void create_pid_file(int myid)
{
    char filename[255] = { 0 };
    sprintf(filename, "MPI_%d.pid", myid);
    
    FILE *pid_file = fopen(filename, "w");
    if (pid_file == NULL) {
        DIE("Error opening file <%s>\n", filename);
    }

    pid_t pid = getpid();
    fprintf(pid_file, "%d\n", pid);

    fflush(pid_file);
    fclose(pid_file);
}

namespace BCM {
void init(int* argc, char*** argv, const char* log_file_name)
{
    // -------------------------------------------------------------------------
    // Initialize MPI
    // -------------------------------------------------------------------------

    // Start MPI
    int myid, nprocs, device_id;
    StartMpi(&myid, &nprocs, argc, argv);

    // -------------------------------------------------------------------------
    // Initialize logging
    // -------------------------------------------------------------------------

    if (log_file_name) {
        open_log_file(myid, log_file_name);
    } else {
        log_file = stderr;
    }

    // -------------------------------------------------------------------------
    // Assign GPU
    // -------------------------------------------------------------------------

    int deviceCount = 0;
    CHECK_DEVICE(cudaGetDeviceCount(&deviceCount));
    ASSERT(deviceCount);
    device_id = assignDeviceToProcess();
    int assigned_device_id = device_id % deviceCount;
    fprintf(stderr, "Trying to set device %d. Total devices: %d. Assigned device: %d\n", device_id, deviceCount, assigned_device_id);
    CHECK_DEVICE(cudaSetDevice(assigned_device_id));

    // -------------------------------------------------------------------------
    // Environment variables
    // -------------------------------------------------------------------------

    if (getenv("SCALENNZMISSING")) {
        scalennzmiss = atoi(getenv("SCALENNZMISSING"));
    }

    if (getenv("SPSP_LIB")) {
        SPSP_LIB = string_to_spsplib(getenv("SPSP_LIB"));
    }

    if (getenv("CUSPARSE_CHUNK_FRACTION")) {
        CUSPARSE_CHUNK_FRACTION = atof(getenv("CUSPARSE_CHUNK_FRACTION"));
    }

    if (ISMASTER) {
        fprintf(stderr, "SPSP_LIB: %s\n", spsplib_to_string(SPSP_LIB).c_str());
        if (SPSP_LIB == SpSpLib::CUSPARSE) {
            fprintf(stderr, "CUSPARSE_CHUNK_FRACTION: %lf\n", CUSPARSE_CHUNK_FRACTION);
        }
    }

    // -------------------------------------------------------------------------

    xsize = 0;
    xvalstat = NULL;
}

void shutdown()
{
    if (SPSP_LIB == SpSpLib::CUSPARSE) {
        ::spgemmcusparseFree();
    }
    FREE(taskmap);
    FREE(itaskmap);

    CUDA_FREE(xvalstat);
    // if (global_bin.stream) {
    //     release_bin(global_bin);
    // }

    MPI_Finalize();
}
};
