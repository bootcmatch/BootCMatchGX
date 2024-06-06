#pragma once

#include "setting.h"

namespace MemoryPool {

extern void** local;
extern void** global;

void initContext(itype full_n, itype n);

void freeContext();
}
