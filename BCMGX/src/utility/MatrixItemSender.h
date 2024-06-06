#pragma once

#include "datastruct/matrixItem.h"
#include "utility/AbstractSender.h"
#include "utility/ProcessSelector.h"
#include <string>

class MatrixItemSender : public AbstractSender<matrixItem_t> {
private:
    ProcessSelector* processSelector;

public:
    MatrixItemSender(ProcessSelector* processSelector, FILE* debug);
    int getProcessForItem(int index, matrixItem_t item);
    matrixItem_t mapItemForProcess(matrixItem_t item, int proc);
    std::string toString(matrixItem_t item);
    void debugItems(const char* title, matrixItem_t* arr, size_t len, bool isOnDevice);
};
