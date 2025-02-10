#include "utility/arrays.h"
#include "utility/devicePrefixSum.h"
#include "utility/memory.h"

int main(int argc, char** argv)
{
    int hArr[] = { 7, 1, 5, 3, 1, 2, 2, 3, 3, 5, 7 };
    size_t len = sizeof(hArr) / sizeof(hArr[0]);

    int* dArr = copyArrayToDevice(hArr, len);

    debugArray("dArr[%d] = %d\n", dArr, len, true, stderr);

    devicePrefixSum(dArr, len);

    debugArray("Prefix summed [%d] = %d\n", dArr, len, true, stderr);

    CUDA_FREE(dArr);

    return 0;
}
