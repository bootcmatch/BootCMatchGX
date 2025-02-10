#include <stdio.h>
#include <stdlib.h>
#include <string>

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
            fprintf(stderr, "Error opening file <%s>\n", filename);
            exit(EXIT_FAILURE);
        }
        if (atexit(close_log_file)) {
            fprintf(stderr, "Error registering atexit\n");
            exit(EXIT_FAILURE);
        }
    }
}

namespace BCM {
void init(int argc, char** argv)
{
}

void shutdown()
{
}
};
