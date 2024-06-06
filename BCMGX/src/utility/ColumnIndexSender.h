#pragma once

#include "utility/AbstractSender.h"
#include "utility/ProcessSelector.h"
#include "utility/setting.h"
#include <string>

class ColumnIndexSender : public AbstractSender<itype> {
private:
    ProcessSelector* processSelector;

public:
    ColumnIndexSender(ProcessSelector* processSelector, FILE* debug);
    int getProcessForItem(int index, itype item);
    itype mapItemForProcess(itype item, int proc);
    std::string toString(itype item);
    void debugItems(const char* title, itype* arr, size_t len, bool isOnDevice);
};

struct ColumnIndexComparator {
    __device__ bool operator()(const itype& a, const itype& b) const
    {
        return a < b;
    }

    __device__ itype operator()(const itype& a) const
    {
        return a;
    }
};
