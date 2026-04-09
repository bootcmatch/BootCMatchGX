#include "utility/ColumnIndexSender.h"

ColumnIndexSender::ColumnIndexSender(ProcessSelector* processSelector, FILE* debug)
    : AbstractSender(debug, MPI_GSSTYPE)
    // : AbstractSender(debug, MPI_gssTYPE)
    , processSelector(processSelector)
{
}

int ColumnIndexSender::getProcessForItem(int index, gsstype item)
{
    return processSelector->getProcessByRow(item);
}

gsstype ColumnIndexSender::mapItemForProcess(gsstype item, int proc)
{
    gsstype ret = use_row_shift
        ? item + processSelector->row_shift
        : item;
    //_MPI_ENV;
    // printf("myid: %d, item: %d, mapped item: %d\n", myid, item, ret);
    return ret;
}

std::string ColumnIndexSender::toString(gsstype item)
{
    char str[512] = { 0 };
    snprintf(str, sizeof(str), "%ld", item);
    return str;
}

void ColumnIndexSender::debugItems(const char* title, gsstype* arr, size_t len, bool isOnDevice)
{
    char str[512] = { 0 };
    snprintf(str, sizeof(str), "%s[%%d] = %%d\n", title);
    debugArray(str, arr, len, isOnDevice, debug);
}
