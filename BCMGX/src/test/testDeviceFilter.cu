#include "utility/arrays.h"
#include "utility/deviceFilter.h"
#include "utility/memory.h"

int main(int argc, char** argv)
{
    int hArr[] = { 1, -1, 2, -1, 3, -1, 3, 5, 7 };
    size_t len = sizeof(hArr) / sizeof(hArr[0]);

    int* dArr = copyArrayToDevice(hArr, len);

    debugArray("dArr[%d] = %d\n", dArr, len, true, stderr);

    size_t lenFiltered;
    int* dArrFiltered = deviceFilter(dArr, len, IsNotEqual<int>(-1), &lenFiltered);

    debugArray("dArrFiltered[%d] = %d\n", dArrFiltered, lenFiltered, true, stderr);
    fprintf(stderr, "lenFiltered: %d\n", lenFiltered);

    CUDA_FREE(dArr);
    CUDA_FREE(dArrFiltered);

    return 0;
}
