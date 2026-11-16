#ifndef CPU_REF_H
#define CPU_REF_H

#include "matrix.h"

// Computes y = A * x
void spmvCpu(const CSR &csr, const float *x, float *y);

// Computes Y = A * B (where B is dense with k columns)
void spmmCpu(const CSR &csr, const float *B, float *Y, int k);

#endif