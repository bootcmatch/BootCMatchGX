/**
 * @file profiling.cpp
 * @brief Profiling utilities for performance measurement and aggregation.
 * 
 * This file includes functions for profiling the execution time of specific code regions,
 * supporting both serial and MPI-based parallel execution. It provides detailed profiling
 * data such as minimum, maximum, and average execution times across different processes.
 * 
 * Profiling data is gathered using either `std::chrono` or `gettimeofday`, based on the 
 * compilation flags, and is stored in a global structure. The profiling information is 
 * aggregated and saved to a file.
 * 
 * @defgroup profiling Profiling utilities
 * @{
 */
#pragma once

#include <stdio.h>

/**
 * @brief Begin profiling a specific code region.
 * 
 * This function initializes the profiling data, recording the start time and associating
 * it with the specified function, file, and label. Profiling data is stored in the global
 * `prof_info` array.
 * 
 * @param file The source file where profiling starts.
 * @param function The function name where profiling starts.
 * @param label A label that identifies the region being profiled.
 */
void beginProfiling(const char* file, const char* function, const char* label);

/**
 * @brief End profiling a specific code region.
 * 
 * This function finalizes the profiling of a given region, recording the stop time.
 * It then returns to the parent profiling context.
 * 
 * @param file The source file where profiling ends.
 * @param function The function name where profiling ends.
 * @param label The label of the region being profiled.
 */
void endProfiling(const char* file, const char* function, const char* label);

/** Flag to enable or disable detailed profiling */
extern bool detailed_prof;

/**
 * @brief Macro to begin profiling a specific code region.
 * 
 * This macro starts profiling for the given region (identified by the label) only if
 * `detailed_prof` is enabled. It captures the start time of the region.
 * 
 * @param LABEL A label identifying the profiling region.
 */
#define BEGIN_PROF(LABEL) \
    if (detailed_prof)    \
    beginProfiling(__FILE__, __FUNCTION__, LABEL)

/**
 * @brief Macro to end profiling a specific code region.
 * 
 * This macro stops profiling for the given region (identified by the label) only if
 * `detailed_prof` is enabled. It captures the stop time and records the elapsed time
 * for the region.
 * 
 * @param LABEL A label identifying the profiling region.
 */
#define END_PROF(LABEL) \
    if (detailed_prof)  \
    endProfiling(__FILE__, __FUNCTION__, LABEL)

/**
 * @brief Maximum length of the profiling path string.
 */
#define MAX_PATH_LEN 1024

/**
 * @struct ProfInfoSummary
 * @brief Structure to store the aggregated summary of profiling data.
 * 
 * This structure holds the profiling summary data for a specific profiling region,
 * including the total count of executions and the sum of execution times.
 * 
 * @var ProfInfoSummary::path The full profiling path for the region.
 * @var ProfInfoSummary::count The number of times the profiling region was executed.
 * @var ProfInfoSummary::sum The sum of all execution times for the profiling region.
 */
struct ProfInfoSummary {
    char path[MAX_PATH_LEN] = { 0 }; /**< The profiling path for the region */
    unsigned long count = 0; /**< The number of times the region was executed */
    unsigned long sum = 0; /**< The sum of execution times for the region */
};

/**
 * @brief Compute a summary of local profiling information.
 * 
 * This function aggregates the profiling information for each unique profiling path,
 * computing the total elapsed time and the number of calls for each path.
 * 
 * @param len Output parameter to store the number of unique profiling paths.
 * @return A dynamically allocated array of `ProfInfoSummary` with aggregated data.
 */
ProfInfoSummary* computeLocalProfilingInfoSummary(size_t* len);

/**
 * @brief Dump a summary of local profiling information to a file.
 * 
 * This function writes the aggregated profiling information (sum and count) for each 
 * unique path to the specified output file.
 * 
 * @param out File pointer for output.
 * @param summary Array of `ProfInfoSummary` containing aggregated profiling data.
 * @param len Length of the `summary` array.
 */
void dumpLocalProfilingInfoSummary(FILE* out, ProfInfoSummary* summary, size_t len);

/**
 * @brief Dump detailed local profiling information to a file.
 * 
 * This function writes detailed profiling information, including the elapsed time for 
 * each profiling path, to a specified output file.
 * 
 * @param filename The file to which profiling information is written.
 * @param summary Array of `ProfInfoSummary` with aggregated profiling data.
 * @param summaryLen Length of the `summary` array.
 */
void dumpLocalProfilingInfo(const char* filename, ProfInfoSummary* summary, size_t summaryLen);

/**
 * @brief Dumps profiling information across multiple processes.
 * 
 * This function gathers profiling information from multiple MPI processes, aggregates it,
 * and writes the results to the specified output file. The data is gathered and aggregated
 * at the root process, and the profiling statistics (such as min, max, avg) for each
 * path are written into the file.
 * 
 * @param filename The name of the file where profiling information will be saved.
 * @param summary A pointer to an array of `ProfInfoSummary` structures containing the profiling data.
 * @param len The length of the `summary` array (number of profiling entries to gather).
 */
void dumpProfilingInfo(const char* filename, ProfInfoSummary* summary, size_t len);

/**
 * @}
 */