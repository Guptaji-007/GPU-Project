#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "matrix.h"

static void addEntry(COO &coo, int row, int col, float value) {
    coo.rowIdx[coo.nnz] = row;
    coo.colIdx[coo.nnz] = col;
    coo.val[coo.nnz] = value;
    coo.nnz++;
}

// reads a Matrix Market (.mtx) file into COO
COO readMtx(const char *fileName) {
    FILE *f = fopen(fileName, "r");
    if (!f) {
        printf("could not open %s\n", fileName);
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), f);  // header, e.g. "%%MatrixMarket matrix coordinate real symmetric"
    bool isPattern = strstr(line, "pattern") != NULL;      // no values, only positions
    bool isSymmetric = strstr(line, "symmetric") != NULL;  // only lower half is stored
    bool isSkew = strstr(line, "skew-symmetric") != NULL;

    // skip comment lines
    do {
        fgets(line, sizeof(line), f);
    } while (line[0] == '%');

    COO coo;
    int fileNnz;
    sscanf(line, "%d %d %d", &coo.rows, &coo.cols, &fileNnz);

    // for symmetric matrices we add the mirrored entry too, so we may need 2x space
    int maxNnz = isSymmetric ? 2 * fileNnz : fileNnz;
    coo.rowIdx = (int *)malloc(maxNnz * sizeof(int));
    coo.colIdx = (int *)malloc(maxNnz * sizeof(int));
    coo.val = (float *)malloc(maxNnz * sizeof(float));
    coo.nnz = 0;

    for (int i = 0; i < fileNnz; i++) {
        int row, col;
        float value = 1.0f;
        if (isPattern)
            fscanf(f, "%d %d", &row, &col);
        else
            fscanf(f, "%d %d %f", &row, &col, &value);

        row--;  // mtx files are 1-based
        col--;
        addEntry(coo, row, col, value);

        if (isSymmetric && row != col)
            addEntry(coo, col, row, isSkew ? -value : value);
    }

    fclose(f);
    return coo;
}

CSR cooToCsr(const COO &coo) {
    CSR csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;
    csr.rowPtr = (int *)calloc(csr.rows + 1, sizeof(int));
    csr.colIdx = (int *)malloc(csr.nnz * sizeof(int));
    csr.val = (float *)malloc(csr.nnz * sizeof(float));

    // count non-zeros in each row
    for (int i = 0; i < coo.nnz; i++)
        csr.rowPtr[coo.rowIdx[i] + 1]++;

    // prefix sum gives the start of each row
    for (int i = 0; i < csr.rows; i++)
        csr.rowPtr[i + 1] += csr.rowPtr[i];

    // put each entry in its row, nextPos[r] = next free slot in row r
    int *nextPos = (int *)malloc(csr.rows * sizeof(int));
    memcpy(nextPos, csr.rowPtr, csr.rows * sizeof(int));
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.rowIdx[i];
        int pos = nextPos[row]++;
        csr.colIdx[pos] = coo.colIdx[i];
        csr.val[pos] = coo.val[i];
    }
    free(nextPos);

    return csr;
}

ELL csrToEll(const CSR &csr) {
    ELL ell;
    ell.rows = csr.rows;
    ell.cols = csr.cols;

    // longest row decides the width
    ell.maxPerRow = 0;
    for (int i = 0; i < csr.rows; i++) {
        int rowLen = csr.rowPtr[i + 1] - csr.rowPtr[i];
        if (rowLen > ell.maxPerRow)
            ell.maxPerRow = rowLen;
    }

    long total = (long)ell.rows * ell.maxPerRow;
    ell.colIdx = (int *)malloc(total * sizeof(int));
    ell.val = (float *)malloc(total * sizeof(float));
    for (long i = 0; i < total; i++) {
        ell.colIdx[i] = -1;
        ell.val[i] = 0.0f;
    }

    for (int row = 0; row < csr.rows; row++) {
        int k = 0;
        for (int j = csr.rowPtr[row]; j < csr.rowPtr[row + 1]; j++) {
            long idx = (long)k * ell.rows + row;  // column-major
            ell.colIdx[idx] = csr.colIdx[j];
            ell.val[idx] = csr.val[j];
            k++;
        }
    }

    return ell;
}

void freeCoo(COO &coo) {
    free(coo.rowIdx);
    free(coo.colIdx);
    free(coo.val);
}

void freeCsr(CSR &csr) {
    free(csr.rowPtr);
    free(csr.colIdx);
    free(csr.val);
}

void freeEll(ELL &ell) {
    free(ell.colIdx);
    free(ell.val);
}
