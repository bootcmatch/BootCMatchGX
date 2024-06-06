// 2023 INFN APE Lab - Sezione di Roma
// cristian.rossi@roma1.infn.it

#ifndef _GPOWERU_H_
#define _GPOWERU_H_

#include <nvml.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>

#define SAMPLE_MAX_SIZE_DEFAULT 1000000
#define MAX_CHECKPOINTS 64
#define MAX_DEVICES 64

#define ROOT_ENABLED 0
#define MULTIGPU_DISABLED 0
#define TIME_STEP 0.00001 // Interval for sampling (in s)
#define POWER_THRESHOLD 0

#if ROOT_ENABLED
#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TGraphErrors.h"
#include "TLine.h"
#endif

/*+++++++++++++++++++++++++++++++++++++++++++++++++++
 *              POWER_MEASURE FUNCTIONALITY         +
 *++++++++++++++++++++++++++++++++++++++++++++++++++*/

void* threadWork(void*); // CPU thread managing the parallel power data taking during the kernel execution

float DataOutput(); // Generate the output samples files
int GPowerU_init(); // Initializations ==> enable the NVML library, starts CPU thread for the power monitoring

#if ROOT_ENABLED
void grapher(); // ROOT graph making function
#endif

__device__ void take_GPU_time(bool last); // Checkpoint power measure __device__ function
void GPowerU_checkpoints(); // Checkpoint power measure CPU function

int GPowerU_end(); // Ends power monitoring, returns data output files

/*+++++++++++++++++++++++++++++++++++++++++++++++++++
 *              POWER_MEASURE GLOBAL VARIABLES       +
 *++++++++++++++++++++++++++++++++++++++++++++++++++*/

int terminate_thread = 0; // END PROGRAM

nvmlDevice_t nvDevice[MAX_DEVICES];
//__managed__ nvmlReturn_t nvResult;
nvmlReturn_t nvResult;

// Time sampling arrays for the power monitoring curve (thread_times) and kernel checkpoints (device_times)
double thread_times[MAX_DEVICES][SAMPLE_MAX_SIZE_DEFAULT];
double device_times[SAMPLE_MAX_SIZE_DEFAULT];

// Power sampling arrays for the power monitoring curve (powers) and kernel checkpoints (powerz)
double thread_powers[MAX_DEVICES][SAMPLE_MAX_SIZE_DEFAULT];
double device_powers[SAMPLE_MAX_SIZE_DEFAULT];

int n_values; // Total number of data taken
int deviceID; // Device id
int threshold; // Threshold value in W indicating the power ranges when GPU is active (above threshold)

struct timeval start_time; // Time for synchronizing threadWork() and checkpoint()

pthread_t thread_sampler; // Thread managing the continuos power data taking

unsigned int core_clock, mem_clock;

float power_peak; // Maximum power value measured

#if MULTIGPU_DISABLED
// Variable used in the kernel checkpoint power data taking
__managed__ int kernel_checkpoints[MAX_CHECKPOINTS];
__managed__ int max_points;
__managed__ bool finish;
#endif
unsigned int device_count;

#endif
