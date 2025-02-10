#include "config/Params.h"
#include "utility/memory.h"
#include "utility/mpi.h"
#include "utility/string.h"

#include <getopt.h>
#include <string.h>

#define USAGE \
    "Usage:\n\t%s --settings <FILE_NAME>\n\n"

int main(int argc, char** argv)
{
    std::string settings_file;

    signed char ch = 0;
    
    static struct option long_options[] = {
        { "settings", required_argument, NULL, 's' },
        { NULL, 0, NULL, 0 }
    };

    while ((ch = getopt_long(argc, argv, "s", long_options, NULL)) != -1) {
        switch (ch) {
        case 's':
            settings_file = optarg;
            break;
        default:
            printf(USAGE, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    int myid, nprocs, device_id;
    StartMpi(&myid, &nprocs, &argc, &argv);

    if (settings_file.empty()) {
        printf(USAGE, argv[0]);
        exit(EXIT_FAILURE);
    }

    params p = ends_with(settings_file, ".properties")
        ? Params::initFromPropertiesFile(settings_file.c_str())
        : Params::initFromFile(settings_file.c_str());
    if (p.error != 0) {
        return -1;
    }

    Params::dump(p, stdout);

    MPI_Finalize();

    return 0;
}