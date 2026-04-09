#pragma once

#include "CSR.h"

#define BUFSIZE 1024

CSR* readMatrixFromFile(const char *matrix_path, int m_type, bool loadOnDevice=true, int nprocs=1);

CSR* readMTXDouble(const char *file_name);

CSR* readMTX2Double(const char *file_name);

void CSRMatrixPrintMM(CSR *A_, const char *file_name);



