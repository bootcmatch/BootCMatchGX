#include "utility/arrays.h"
#include "utility/deviceSort.h"
#include "utility/memory.h"

struct Comparator {
    __device__ bool operator()(const int& a, const int& b) const
    {
        return a < b;
    }

    __device__ int operator()(const int& a) const
    {
        return a;
    }
};

int main(int argc, char** argv)
{
    int hArr[] = { 7, 1, 5, 3, 1, 2, 2, 3, 3, 5, 7 };
    size_t len = sizeof(hArr) / sizeof(hArr[0]);

    int* dArr = copyArrayToDevice(hArr, len);

    debugArray("dArr[%d] = %d\n", dArr, len, true, stderr);

    deviceSort<int, int, Comparator>(dArr, len, Comparator());

    debugArray("dArrSorted[%d] = %d\n", dArr, len, true, stderr);

    CUDA_FREE(dArr);

    return 0;
}
