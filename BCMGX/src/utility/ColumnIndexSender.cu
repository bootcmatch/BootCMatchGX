#include "utility/ColumnIndexSender.h"

ColumnIndexSender::ColumnIndexSender(ProcessSelector* processSelector, FILE* debug)
    : AbstractSender(debug, MPI_ITYPE)
    , processSelector(processSelector)
{
}

int ColumnIndexSender::getProcessForItem(int index, itype item)
{
    return processSelector->getProcessByRow(item);
}

itype ColumnIndexSender::mapItemForProcess(itype item, int proc)
{
    itype ret = use_row_shift
        ? item + processSelector->row_shift
        : item;
    //_MPI_ENV;
    // printf("myid: %d, item: %d, mapped item: %d\n", myid, item, ret);
    return ret;
}

std::string ColumnIndexSender::toString(itype item)
{
    char str[512] = { 0 };
    snprintf(str, sizeof(str), "%d", item);
    return str;
}

void ColumnIndexSender::debugItems(const char* title, itype* arr, size_t len, bool isOnDevice)
{
    char str[512] = { 0 };
    snprintf(str, sizeof(str), "%s[%%d] = %%d\n", title);
    debugArray(str, arr, len, isOnDevice, debug);
}
