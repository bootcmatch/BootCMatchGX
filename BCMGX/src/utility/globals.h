#pragma once

#include <stdio.h>

// #define DEFAULTSCALENNZMISS 1024
extern int xsize;
extern double* xvalstat;
extern int* taskmap;
extern int* itaskmap;
extern int scalennzmiss;
extern char idstring[128];
extern FILE* log_file;

void close_log_file();
void open_log_file(int myid, const char* log_filename);

#if 1
#define TRACE_REACHED_LINE()                                           \
    if (log_file) {                                                    \
        fprintf(log_file, "Reached line %d in file %s, function %s\n", \
            __LINE__, __FILE__, __FUNCTION__);                         \
    }
#else
#define TRACE_REACHED_LINE()                                     \
    fprintf(stderr, "Reached line %d in file %s, function %s\n", \
        __LINE__, __FILE__, __FUNCTION__)
#endif

namespace BCM {
void init(int argc, char **argv);
void shutdown();
};