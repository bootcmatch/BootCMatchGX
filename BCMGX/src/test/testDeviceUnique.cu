#include "utility/arrays.h"
#include "utility/deviceUnique.h"
#include "utility/memory.h"

int main(int argc, char** argv)
{
    int hArr[] = { 1, 1, 2, 2, 3, 3, 3, 5, 7 };
    size_t len = sizeof(hArr) / sizeof(hArr[0]);

    int* dArr = copyArrayToDevice(hArr, len);

    // debugArray("dArr[%d] = %d\n", dArr, len, true, stderr);

    size_t lenUnique;
    int* dArrUnique = deviceUnique(dArr, len, &lenUnique);

    // debugArray("dArrUnique[%d] = %d\n", dArrUnique, lenUnique, true, stderr);

    CUDA_FREE(dArr);
    CUDA_FREE(dArrUnique);

    return 0;
}
