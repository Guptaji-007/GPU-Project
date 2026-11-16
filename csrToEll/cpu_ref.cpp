#include <cmath>
#include <iostream>
#include <chrono>
#include "matrix.h"

// CPU SpMV (y = A * x)
void spmvCpu(const CSR &csr, const float *x, float *y) {
    for (int i = 0; i < csr.rows; i++) {
        float sum = 0.0f;
        // csr.rowPtr[i] to csr.rowPtr[i+1] gives the range of non-zeros for row i
        for (int j = csr.rowPtr[i]; j < csr.rowPtr[i+1]; j++) {
            sum += csr.val[j] * x[csr.colIdx[j]];
        }
        y[i] = sum;
    }
}

// CPU SpMM (Y = A * B)
void spmmCpu(const CSR &csr, const float *B, float *Y, int k) {
    for (int i = 0; i < csr.rows; i++) {
        for (int colB = 0; colB < k; colB++) {
            float sum = 0.0f;
            for (int j = csr.rowPtr[i]; j < csr.rowPtr[i+1]; j++) {
                // B is dense, accessed via (row * k + col)
                sum += csr.val[j] * B[csr.colIdx[j] * k + colB];
            }
            Y[i * k + colB] = sum;
        }
    }
}