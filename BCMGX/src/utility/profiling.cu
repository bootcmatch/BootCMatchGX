#include "profiling.h"
#include "utility/logf.h"
#include "utility/mpi.h"
#include "utility/utils.h"

#include <assert.h>
#include <map>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define PROFILING_CHRONO 1

#if PROFILING_CHRONO
#include <chrono>
#else
#include <sys/time.h>
#endif

#define MAX_PROF_INFO 90000

/** 
 * @struct ProfInfo
 * @brief Profiling data structure holding timing and hierarchical information.
 * 
 * This structure stores detailed timing information for a specific code region.
 * It includes the start and stop times, the hierarchical level of the profile, 
 * the parent profile information, and metadata such as the file, function, and label.
 */
struct ProfInfo {
#if PROFILING_CHRONO
    /** Start time using high-resolution clock */
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    /** Stop time using high-resolution clock */
    std::chrono::time_point<std::chrono::high_resolution_clock> stop;
#else
    /** Start time using timeval (microseconds) */
    struct timeval start = { 0 };
    /** Stop time using timeval (microseconds) */
    struct timeval stop = { 0 };
#endif

    const char* file = 0; /**< The file where profiling was started */
    const char* function = 0; /**< The function where profiling was started */
    const char* label = 0; /**< The label for the profiling region */
    std::string path; /**< Full profiling path (used for hierarchical profiling) */
    size_t level = 0; /**< Profiling depth level */
    size_t index = 0; /**< Index of the profiling info in the global array */
    struct ProfInfo* parent = 0; /**< Parent profile, NULL if no parent */

#if PROFILING_CHRONO
    /** 
     * @brief Calculate the elapsed time in nanoseconds.
     * 
     * @return The elapsed time in nanoseconds.
     */
    unsigned long elapsed_time()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    }
#else
    /** 
     * @brief Calculate the elapsed time in microseconds.
     * 
     * @return The elapsed time in microseconds.
     */
    unsigned long elapsed_time()
    {
        return (stop.tv_sec * 1000000 + stop.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    }
#endif
};

bool detailed_prof = false;
ProfInfo prof_info[MAX_PROF_INFO];
size_t prof_info_size = 0;
ProfInfo* current_prof_info = NULL;

#if PROFILING_CHRONO
#define GET_PROF_TIME(VAR)                               \
    do {                                                 \
        VAR = std::chrono::high_resolution_clock::now(); \
    } while (0)
#else
#define GET_PROF_TIME(VAR) gettimeofday(&VAR, (struct timezone*)0)
#endif

void beginProfiling(const char* file, const char* function, const char* label)
{
    std::string path = current_prof_info
        ? (current_prof_info->path + "::" + label)
        : label;
    size_t level = current_prof_info
        ? (current_prof_info->level + 1)
        : 0;
    GET_PROF_TIME(prof_info[prof_info_size].start);
    prof_info[prof_info_size].file = file;
    prof_info[prof_info_size].function = function;
    prof_info[prof_info_size].label = label;
    prof_info[prof_info_size].index = prof_info_size;
    prof_info[prof_info_size].parent = current_prof_info;
    prof_info[prof_info_size].path = path;
    prof_info[prof_info_size].level = level;
    current_prof_info = &prof_info[prof_info_size];
    assert(prof_info_size < MAX_PROF_INFO - 1);
    prof_info_size++;
}

void endProfiling(const char* file, const char* function, const char* label)
{
    if (file != current_prof_info->file) {
        printf("__FILE__: %s, current_prof_info->file: %s\n", file, current_prof_info->file);
        printf("__FUNCTION__: %s, current_prof_info->function: %s\n", function, current_prof_info->function);
    }
    assert(file == current_prof_info->file);
    assert(function == current_prof_info->function);
    assert(label == current_prof_info->label);
    // cudaDeviceSynchronize();
    GET_PROF_TIME(current_prof_info->stop);
    current_prof_info = current_prof_info->parent;
}

ProfInfoSummary* computeLocalProfilingInfoSummary(size_t* len)
{
    std::map<std::string, unsigned long> sum;
    std::map<std::string, unsigned long> count;
    for (int i = 0; i < prof_info_size; i++) {
        ProfInfo* current = &prof_info[i];
        {
            std::map<std::string, unsigned long>::iterator it = count.find(current->path);
            if (it == count.end()) {
                count.insert({ current->path, 1 });
            } else {
                it->second += 1;
            }
        }
        {
            std::map<std::string, unsigned long>::iterator it = sum.find(current->path);
            if (it == sum.end()) {
                sum.insert({ current->path, current->elapsed_time() });
            } else {
                it->second += current->elapsed_time();
            }
        }
    }

    assert(count.size() == sum.size());

    *len = count.size();
    ProfInfoSummary* ret = new ProfInfoSummary[*len];

    size_t index = 0;
    for (auto it = sum.begin(); it != sum.end(); it++) {
        strncpy(ret[index].path, it->first.c_str(), MAX_PATH_LEN - 1);
        ret[index].sum = it->second;
        ret[index].count = count[it->first];
        index++;
    }

    return ret;
}

void dumpLocalProfilingInfoSummary(FILE* out, ProfInfoSummary* summary, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        fprintf(out, "%15lu ", summary[i].sum);
        fprintf(out, "(count %10lu) ", summary[i].count);
        fprintf(out, "%s", summary[i].path);
        fprintf(out, "\n");
    }
}

void dumpLocalProfilingInfo(const char* filename, ProfInfoSummary* summary, size_t summaryLen)
{
    FILE* out = fopen(filename, "a");
    if (out == NULL) {
        printf("File %s opening failed.\n", filename);
    }

    for (int i = 0; i < prof_info_size; i++) {
        ProfInfo* current = &prof_info[i];
        fprintf(out, "%15lu ", current->elapsed_time());
        fprintf(out, "%s\n", current->path.c_str());
    }

    fprintf(out, "--------------------------------------------------------------------------\n");

    dumpLocalProfilingInfoSummary(out, summary, summaryLen);

    fclose(out);
}

void dumpProfilingInfo(const char* filename, ProfInfoSummary* summary, size_t len)
{
    _MPI_ENV;

    // printf("Proc %d summary len %lu\n", myid, len);

    size_t lenPerProc[nprocs] = { 0 };

    CHECK_MPI(MPI_Gather(
        &len, // send_data,
        sizeof(size_t), // send_count,
        MPI_BYTE, // send_datatype,
        lenPerProc,
        sizeof(size_t), // recv_count,
        MPI_BYTE, // recv_datatype,
        0, // root,
        MPI_COMM_WORLD));

    size_t totLen = 0;
    if (ISMASTER) {
        for (int i = 0; i < nprocs; i++) {
            totLen += lenPerProc[i];
        }
        // printf("Total len %d\n", totLen);
    }

    ProfInfoSummary* recv_data = ISMASTER ? new ProfInfoSummary[totLen] : NULL;
    int* recv_counts = ISMASTER ? new int[nprocs] : NULL;
    int* recv_displs = ISMASTER ? new int[nprocs] : NULL;

    if (ISMASTER) {
        for (int i = 0; i < nprocs; i++) {
            recv_counts[i] = lenPerProc[i] * sizeof(ProfInfoSummary);
            recv_displs[i] = i == 0 ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
        }
    }

    CHECK_MPI(MPI_Gatherv(
        summary, // send_data,
        sizeof(ProfInfoSummary) * len, // send_count,
        MPI_BYTE, // send_datatype,

        recv_data,
        recv_counts,
        recv_displs,

        MPI_BYTE, // recv_datatype,
        0, // root,
        MPI_COMM_WORLD));

    if (ISMASTER) {
        std::map<std::string, std::vector<ssize_t>> path2proc2index;

        printf("Total len %zu\n", totLen);
        int p = 0;
        for (int i = 0; i < totLen; i++) {
            while (lenPerProc[p] == 0) {
                p++;
            }
            if (lenPerProc[p] > 0) {
                lenPerProc[p]--;
            }
            // printf("recv_data[%d].path: %s -> p %d\n", i, recv_data[i].path, p);
            auto it = path2proc2index.find(recv_data[i].path);
            if (it == path2proc2index.end()) {
                path2proc2index.insert({ recv_data[i].path, std::vector<ssize_t>(nprocs) });
                for (int j = 0; j < nprocs; j++) {
                    path2proc2index[recv_data[i].path].push_back(-1);
                }
            }
            // printf("recv_data[%d].path: %s -> p %d\n", i, recv_data[i].path, p);
            path2proc2index[recv_data[i].path][p] = i;
        }

        FILE* fp = fopen(filename, "a");
        if (fp == NULL) {
            printf("File %s opening failed.\n", filename);
        }

        unsigned long maxSolveTime = 0UL;
        bool printHeader = true;
        for (auto it = path2proc2index.begin(); it != path2proc2index.end(); it++) {
            const std::string& path = it->first;

            // printf("Path: %s\n", path.c_str());

            unsigned long minCount = 0;
            unsigned long maxCount = 0;
            unsigned long sumCount = 0;

            unsigned long minSum = 0;
            unsigned long maxSum = 0;
            unsigned long sumSum = 0;

            bool firstProc = true;
            for (int p = 0; p < nprocs; p++) {
                // printf("\tp: %d\n", p);
                ssize_t index = it->second[p];
                if (index < 0) {
                    continue;
                }
                const ProfInfoSummary& current = recv_data[index];
                if (firstProc) {
                    minCount = current.count;
                    maxCount = current.count;
                    sumCount = current.count;

                    minSum = current.sum;
                    maxSum = current.sum;
                    sumSum = current.sum;

                    firstProc = false;
                    continue;
                }
                // assert(strcmp(current.path, recv_data[i].path) == 0);
                if (current.count < minCount) {
                    minCount = current.count;
                }
                if (current.count > maxCount) {
                    maxCount = current.count;
                }
                sumCount += current.count;

                if (current.sum < minSum) {
                    minSum = current.sum;
                }
                if (current.sum > maxSum) {
                    maxSum = current.sum;
                }
                sumSum += current.sum;
            }

            if (path == "solve") {
                maxSolveTime = maxSum;
            }

            if (printHeader) {
                logf(fp, "\n");
                logf(fp, "|%-130s", "PATH");
                logf(fp, "|%15s", "MIN [ns]");
                logf(fp, "|%15s", "MAX [ns]");
                logf(fp, "|%15s", "AVG [ns]");
                logf(fp, "|%10s", "# MIN");
                logf(fp, "|%10s", "# MAX");
                logf(fp, "|%10s", "# AVG");
                logf(fp, "|%6s", "# PROC");
                logf(fp, "|\n");
                printHeader = false;
            }
            if (!IS_ZERO(sumSum)) {
                logf(fp, "|%-130s", path.c_str());
                logf(fp, "|%15lu", minSum);
                logf(fp, "|%15lu", maxSum);
                logf(fp, "|%15.2lf", (double)sumSum / (double)nprocs);
                logf(fp, "|%10lu", minCount);
                logf(fp, "|%10lu", maxCount);
                logf(fp, "|%10.2lf", (double)sumCount / (double)nprocs);
                logf(fp, "|%6d", nprocs);
                logf(fp, "|\n");
            }
        }

        if (!IS_ZERO(maxSolveTime)) {
            logf(fp, "\nTOTAL SOLVE TIME [s]: %lf\n", (double)maxSolveTime * 1e-9);
        }

        if (fp) {
            fclose(fp);
        }
    }

    if (recv_data) {
        delete (recv_data);
    }
    if (recv_counts) {
        delete (recv_counts);
    }
    if (recv_displs) {
        delete (recv_displs);
    }
}