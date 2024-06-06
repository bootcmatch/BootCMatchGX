#include "utility/MatrixItemSender.h"

MatrixItemSender::MatrixItemSender(ProcessSelector* processSelector, FILE* debug)
    : AbstractSender(debug, MPI_MATRIX_ITEM_T)
    , processSelector(processSelector)
{
}

int MatrixItemSender::getProcessForItem(int index, matrixItem_t item)
{
    return processSelector->getProcessByRow(item.col);
}

matrixItem_t MatrixItemSender::mapItemForProcess(matrixItem_t item, int proc)
{
    if (use_row_shift) {
        matrixItem_t ret = item;
        ret.col += processSelector->row_shift;
        return ret;
    } else {
        return item;
    }
}

std::string MatrixItemSender::toString(matrixItem_t item)
{
    char str[1024] = { 0 };
    snprintf(str, sizeof(str), "(%d, %d)=%lf",
        item.row,
        item.col,
        item.val);
    return str;
}

void MatrixItemSender::debugItems(const char* title, matrixItem_t* arr, size_t len, bool isOnDevice)
{
    debugMatrixItems(title, arr, len, isOnDevice, debug);
}
