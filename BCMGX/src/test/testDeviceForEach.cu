#include "utility/arrays.h"
#include "utility/deviceForEach.h"

int main(int argc, char** argv)
{
    int hArr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    size_t len = sizeof(hArr) / sizeof(hArr[0]);

    int* dArr = copyArrayToDevice(hArr, len);

    debugArray("dArr[%d] = %d\n", dArr, len, true, stderr);

    deviceForEach(dArr, len, FillWithIndexOperator<int>());

    debugArray("dProcessedArr[%d] = %d\n", dArr, len, true, stderr);

    cudaFree(dArr);

    return 0;
}
