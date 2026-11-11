#ifndef MATRIX_H
#define MATRIX_H

// COO: a list of (row, col, value)
struct COO {
    int rows, cols, nnz;
    int *rowIdx;
    int *colIdx;
    float *val;
};

// CSR: non-zeros of row i are at positions rowPtr[i] .. rowPtr[i+1]-1
struct CSR {
    int rows, cols, nnz;
    int *rowPtr;
    int *colIdx;
    float *val;
};

// ELL: every row padded to maxPerRow entries.
// stored column-major: entry k of row i is at index k * rows + i
// padded slots have colIdx = -1 and val = 0
struct ELL {
    int rows, cols;
    int maxPerRow;
    int *colIdx;
    float *val;
};

COO readMtx(const char *fileName);
CSR cooToCsr(const COO &coo);
ELL csrToEll(const CSR &csr);

void freeCoo(COO &coo);
void freeCsr(CSR &csr);
void freeEll(ELL &ell);

#endif
