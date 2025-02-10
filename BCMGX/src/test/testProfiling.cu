#include "utility/profiling.h"

void spiderman() {
    BEGIN_PROF(__FUNCTION__);
    printf("Hey, I'm in %s\n", __FUNCTION__);
    END_PROF(__FUNCTION__);
}

void paperino() {
    BEGIN_PROF(__FUNCTION__);
    printf("Hey, I'm in %s\n", __FUNCTION__);
    spiderman();
    END_PROF(__FUNCTION__);
}

void pluto() {
    BEGIN_PROF(__FUNCTION__);
    printf("Hey, I'm in %s\n", __FUNCTION__);
    paperino();
    END_PROF(__FUNCTION__);
}

void pippo() {
    BEGIN_PROF(__FUNCTION__);
    printf("Hey, I'm in %s\n", __FUNCTION__);
    spiderman();
    END_PROF(__FUNCTION__);
}

int main(void) {
    BEGIN_PROF(__FUNCTION__);
    printf("Hey, I'm in %s\n", __FUNCTION__);
    pippo();
    pluto();
    pippo();
    END_PROF(__FUNCTION__);

    dumpProfilingInfo(stderr);

    return 0;
}