#include "timing.h"

#include "utility/utils.h"

#include <ctime>
#include <stdlib.h>
#include <sys/time.h>

#define GET_TIME_OF_DAY gettimeofday(&temp_1, (struct timezone*)0);
#define TIME_ELAPSED ((temp_1.tv_sec) + ((temp_1.tv_usec) / (1.e6)))

namespace TIME {

int timer_index;
int n;
double *starts, *stops;
static struct timeval temp_1;

void init()
{
    TIME::timer_index = 0;
    TIME::n = 0;
    TIME::starts = NULL;
    TIME::stops = NULL;
}

void addTimer()
{
    TIME::starts = (double*)realloc(TIME::starts, sizeof(double) * TIME::n);
    CHECK_HOST(TIME::starts);
    TIME::stops = (double*)realloc(TIME::stops, sizeof(double) * TIME::n);
    CHECK_HOST(TIME::stops);
    TIME::starts[TIME::n - 1] = 0.;
    TIME::stops[TIME::n - 1] = 0.;
}

void start()
{
    if (TIME::timer_index == TIME::n) {
        TIME::n++;
        TIME::addTimer();
    }
    GET_TIME_OF_DAY;
    TIME::starts[TIME::timer_index] = TIME_ELAPSED;
    TIME::timer_index++;
}

float stop()
{
    double milliseconds = 0.;
    double start_ = TIME::starts[TIME::timer_index - 1];
    GET_TIME_OF_DAY;
    double stop_ = TIME_ELAPSED;
    milliseconds = stop_ - start_;
    milliseconds *= 1000.0;
    TIME::timer_index--;
    return (float)milliseconds;
}

void free()
{
    std::free(TIME::starts);
    std::free(TIME::stops);
}

}
