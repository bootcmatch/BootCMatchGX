#include "utility/arrays.h"
#include "utility/deviceMap.h"

struct Mapper {
    __device__ int operator()(const int& a)
    {
        return a * 2;
    }
};

int main(int argc, char** argv)
{
    int hArr[] = { 7, 1, 5, 3, 1, 2, 2, 3, 3, 5, 7 };
    size_t len = sizeof(hArr) / sizeof(hArr[0]);

    int* dArr = copyArrayToDevice(hArr, len);

    debugArray("dArr[%d] = %d\n", dArr, len, true, stderr);

    int* dMappedArr = deviceMap<int, int, Mapper>(dArr, len, Mapper());

    debugArray("dMappedArr[%d] = %d\n", dMappedArr, len, true, stderr);

    cudaFree(dMappedArr);
    cudaFree(dArr);

    return 0;
}
