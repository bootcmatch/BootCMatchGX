#ifndef TIMING_H
#define TIMING_H
#include <sys/time.h>

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)*1.e6+(temp_2.tv_usec-temp_1 .tv_usec))
#endif
