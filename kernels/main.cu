#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "matrix.h"
#include "cpu_ref.h"
#include "utils.h"

// kernel 1: CSR scalar SpMV
// one thread computes one row of y (same loop as spmvCpu)
__global__ void csrScalarSpmv(int rows, const int *rowPtr, const int *colIdx,
                              const float *val, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; j++)
            sum += val[j] * x[colIdx[j]];
        y[row] = sum;
    }
}

// kernel 2: CSR vector SpMV
// one warp (32 threads) works on one row, so long rows are shared by 32 threads
__global__ void csrVectorSpmv(int rows, const int *rowPtr, const int *colIdx,
                              const float *val, const float *x, float *y) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int row = thread / 32;   // which row this warp works on
    int lane = thread % 32;  // position of this thread inside the warp

    if (row < rows) {
        // each thread adds every 32nd non-zero of the row
        // neighbouring threads read neighbouring val/colIdx, so reads are coalesced
        float sum = 0.0f;
        for (int j = rowPtr[row] + lane; j < rowPtr[row + 1]; j += 32)
            sum += val[j] * x[colIdx[j]];

        // add the 32 partial sums inside the warp: 16, 8, 4, 2, 1
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        // lane 0 now has the full sum
        if (lane == 0)
            y[row] = sum;
    }
}

// kernel 3: ELL SpMV
// one thread per row. ELL is stored column-major (entry k of row i is at k * rows + i),
// so in every step of the loop neighbouring threads read neighbouring memory -> coalesced
__global__ void ellSpmv(int rows, int maxPerRow, const int *colIdx,
                        const float *val, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int k = 0; k < maxPerRow; k++) {
            long idx = (long)k * rows + row;
            int col = colIdx[idx];
            if (col != -1)  // -1 means padded slot, skip it
                sum += val[idx] * x[col];
        }
        y[row] = sum;
    }
}

// kernel 4: CSR naive SpMM 
// one thread computes one element: Y[row][c].
__global__ void csrNaiveSpmm(int rows, int k, const int *rowPtr, const int *colIdx,
                             const float *val, const float *B, float *Y) {
    long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < (long)rows * k) {
        int row = id % rows;  
        int c = id / rows;    
        float sum = 0.0f;
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; j++)
            sum += val[j] * B[(long)colIdx[j] * k + c];
        Y[(long)row * k + c] = sum;
    }
}

// kernel 5: coalesced CSR SpMM
// same as kernel 4 but I swapped how threads map to Y.
// now threads next to each other share a row and take neighbouring columns,
// so a warp reads 32 values of B in a row instead of jumping around memory
__global__ void csrCoalescedSpmm(int rows, int k, const int *rowPtr, const int *colIdx,
                                 const float *val, const float *B, float *Y) {
    long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < (long)rows * k) {
        int row = id / k;  // a warp shares one row
        int c = id % k;    // and takes neighbouring columns of B
        float sum = 0.0f;
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; j++)
            sum += val[j] * B[(long)colIdx[j] * k + c];
        Y[(long)row * k + c] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx>\n", argv[0]);
        return 1;
    }

    //  load Matrix Data
    COO coo = readMtx(argv[1]);
    CSR csr = cooToCsr(coo);

    //  allocate cpu memory for vectors
    float *x_cpu = (float*)malloc(csr.cols * sizeof(float));
    float *y_cpu = (float*)malloc(csr.rows * sizeof(float));
    float *y_gpu_result = (float*)malloc(csr.rows * sizeof(float));
    
    // fill input vector 'x' with dummy data 
    for(int i = 0; i < csr.cols; i++) x_cpu[i] = 1.0f;

   
    //  cpu execution and timing
    
    auto startCpu = std::chrono::high_resolution_clock::now();
    
    spmvCpu(csr, x_cpu, y_cpu);
    
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = endCpu - startCpu;
    printf("CPU SpMV Time: %f ms\n", cpu_ms.count());

    
    // gpu execution and timing
    
    // allocate gpu memory
    int *gpuRowPtr, *gpuColIdx;
    float *gpuVal, *gpuX, *gpuY;
    cudaMalloc(&gpuRowPtr, (csr.rows + 1) * sizeof(int));
    cudaMalloc(&gpuColIdx, csr.nnz * sizeof(int));
    cudaMalloc(&gpuVal, csr.nnz * sizeof(float));
    cudaMalloc(&gpuX, csr.cols * sizeof(float));
    cudaMalloc(&gpuY, csr.rows * sizeof(float));

    // copy matrix and x from cpu to gpu
    cudaMemcpy(gpuRowPtr, csr.rowPtr, (csr.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuColIdx, csr.colIdx, csr.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuVal, csr.val, csr.nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuX, x_cpu, csr.cols * sizeof(float), cudaMemcpyHostToDevice);

    // one thread per row, 256 threads in a block
    int threads = 256;
    int blocks = (csr.rows + threads - 1) / threads;

    cudaEvent_t startGpu, stopGpu;
    cudaEventCreate(&startGpu);
    cudaEventCreate(&stopGpu);

    int runs = 50;
    float total_gpu_ms = 0.0f;

    // Warmup run (gets the GPU awake and caches warmed up) as the first time GPU , it is slower to initialization overhead(loading data ,etc).
    // so we do a warmup run and ignore its time so the hw is ready

    // Kernel calling code part for warmup
    // run the kernel more than once and take the average as this smooths out randomm os background tasks and gives stable, reportable no. for benchmark
    csrScalarSpmv<<<blocks, threads>>>(csr.rows, gpuRowPtr, gpuColIdx, gpuVal, gpuX, gpuY);
    cudaDeviceSynchronize();

    // if the launch failed, this tells us why
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    // timed execution loop
    for(int i = 0; i < runs; i++) {
        cudaEventRecord(startGpu);
        
        csrScalarSpmv<<<blocks, threads>>>(csr.rows, gpuRowPtr, gpuColIdx, gpuVal, gpuX, gpuY);
        
        cudaEventRecord(stopGpu);
        cudaEventSynchronize(stopGpu);
        
        float iter_ms = 0.0f;
        cudaEventElapsedTime(&iter_ms, startGpu, stopGpu);
        total_gpu_ms += iter_ms;
    }

    float avgMs = total_gpu_ms / runs;
    printf("Average GPU CSR scalar SpMV Time: %f ms\n", avgMs);

    // each non-zero does one multiply and one add = 2 flops
    float gflops = 2.0f * csr.nnz / (avgMs * 1e6f);
    printf("GPU CSR scalar SpMV GFLOP/s: %f\n", gflops);


    // correctness check and cleanup

    // copy the result back from gpu
    cudaMemcpy(y_gpu_result, gpuY, csr.rows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Checking CSR scalar SpMV Correctness: ");
    checkResult(y_cpu, y_gpu_result, csr.rows);


    // kernel 2: CSR vector SpMV (one warp per row)

    // clear y so the check below really tests kernel 2
    cudaMemset(gpuY, 0, csr.rows * sizeof(float));

    // we need 32 threads for every row now
    int vectorBlocks = (csr.rows * 32 + threads - 1) / threads;

    // warmup
    csrVectorSpmv<<<vectorBlocks, threads>>>(csr.rows, gpuRowPtr, gpuColIdx, gpuVal, gpuX, gpuY);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    float totalVectorMs = 0.0f;
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(startGpu);

        csrVectorSpmv<<<vectorBlocks, threads>>>(csr.rows, gpuRowPtr, gpuColIdx, gpuVal, gpuX, gpuY);

        cudaEventRecord(stopGpu);
        cudaEventSynchronize(stopGpu);

        float iterMs = 0.0f;
        cudaEventElapsedTime(&iterMs, startGpu, stopGpu);
        totalVectorMs += iterMs;
    }

    float avgVectorMs = totalVectorMs / runs;
    printf("Average GPU CSR vector SpMV Time: %f ms\n", avgVectorMs);
    printf("GPU CSR vector SpMV GFLOP/s: %f\n", 2.0f * csr.nnz / (avgVectorMs * 1e6f));

    cudaMemcpy(y_gpu_result, gpuY, csr.rows * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Checking CSR vector SpMV Correctness: ");
    checkResult(y_cpu, y_gpu_result, csr.rows);



    // kernel 3: ELL SpMV (one thread per row, coalesced)

    // ELL pads every row to the longest row. if one row is very long
    // (like in web graphs) this needs too much memory, so skip ELL then
    int maxLen = 0;
    for (int i = 0; i < csr.rows; i++) {
        int rowLen = csr.rowPtr[i + 1] - csr.rowPtr[i];
        if (rowLen > maxLen)
            maxLen = rowLen;
    }
    float padRatio = (float)maxLen * csr.rows / csr.nnz;
    printf("ELL size / nnz = %.2f\n", padRatio);

    if (padRatio < 10.0f) {
        ELL ell = csrToEll(csr);
        long ellSize = (long)ell.rows * ell.maxPerRow;

        int *gpuEllColIdx;
        float *gpuEllVal;
        cudaMalloc(&gpuEllColIdx, ellSize * sizeof(int));
        cudaMalloc(&gpuEllVal, ellSize * sizeof(float));
        cudaMemcpy(gpuEllColIdx, ell.colIdx, ellSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuEllVal, ell.val, ellSize * sizeof(float), cudaMemcpyHostToDevice);

        // clear y so the check below really tests kernel 3
        cudaMemset(gpuY, 0, csr.rows * sizeof(float));

        // warmup (same blocks as kernel 1, one thread per row)
        ellSpmv<<<blocks, threads>>>(ell.rows, ell.maxPerRow, gpuEllColIdx, gpuEllVal, gpuX, gpuY);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA error: %s\n", cudaGetErrorString(err));

        float totalEllMs = 0.0f;
        for (int i = 0; i < runs; i++) {
            cudaEventRecord(startGpu);

            ellSpmv<<<blocks, threads>>>(ell.rows, ell.maxPerRow, gpuEllColIdx, gpuEllVal, gpuX, gpuY);

            cudaEventRecord(stopGpu);
            cudaEventSynchronize(stopGpu);

            float iterMs = 0.0f;
            cudaEventElapsedTime(&iterMs, startGpu, stopGpu);
            totalEllMs += iterMs;
        }

        // GFLOP/s counts only the real non-zeros, not the padding
        float avgEllMs = totalEllMs / runs;
        printf("Average GPU ELL SpMV Time: %f ms\n", avgEllMs);
        printf("GPU ELL SpMV GFLOP/s: %f\n", 2.0f * csr.nnz / (avgEllMs * 1e6f));

        cudaMemcpy(y_gpu_result, gpuY, csr.rows * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Checking ELL SpMV Correctness: ");
        checkResult(y_cpu, y_gpu_result, csr.rows);

        cudaFree(gpuEllColIdx);
        cudaFree(gpuEllVal);
        freeEll(ell);
    } else {
        printf("ELL SpMV: skipped (too much padding)\n");
    }

    // SpMM will need the matrix on the gpu again later, so it is freed later
    cudaFree(gpuX);
    cudaFree(gpuY);

    // SpMM part
  
    //SpMM setup and allocation
  
    int k = 32; // number of columns for dense matrix B (test 8, 32, 64 later)
    
    // B has 'cols' rows and 'k' columns. Output Y has 'rows' rows and 'k' columns
    float *B_cpu = (float*)malloc(csr.cols * k * sizeof(float));
    float *Y_cpu = (float*)malloc(csr.rows * k * sizeof(float));
    float *Y_gpu_result = (float*)malloc(csr.rows * k * sizeof(float));
    
    // fill dense matrix B with dummy data
    for(int i = 0; i < csr.cols * k; i++) {
        B_cpu[i] = 1.0f; 
    }

  
    // cpu SpMM execution and timing
  
    auto startCpuSpmm = std::chrono::high_resolution_clock::now();
    
    spmmCpu(csr, B_cpu, Y_cpu, k);
    
    auto endCpuSpmm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms_spmm = endCpuSpmm - startCpuSpmm;
    printf("CPU SpMM Time (k=%d): %f ms\n", k, cpu_ms_spmm.count());

  
    // gpu SpMM execution and timing
  
    // cudaMalloc and cudaMemcpy B_gpu here
    float *gpuB, *gpuY_spmm;
    cudaMalloc(&gpuB, csr.cols * k * sizeof(float));
    cudaMalloc(&gpuY_spmm, csr.rows * k * sizeof(float));
    cudaMemcpy(gpuB, B_cpu, csr.cols * k * sizeof(float), cudaMemcpyHostToDevice);

    // clear y so the check below really tests kernel 4
    cudaMemset(gpuY_spmm, 0, (size_t)csr.rows * k * sizeof(float));

    //one thread per element of y
    int spmmBlocks = (int)(((long)csr.rows * k + threads - 1) / threads);

    float total_gpu_ms_spmm = 0.0f;

    // warmup run
    csrNaiveSpmm<<<spmmBlocks, threads>>>(csr.rows, k, gpuRowPtr, gpuColIdx, gpuVal, gpuB, gpuY_spmm);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // timed execution loop
    for(int i = 0; i < runs; i++) {
        cudaEventRecord(startGpu);
        
        csrNaiveSpmm<<<spmmBlocks, threads>>>(csr.rows, k, gpuRowPtr, gpuColIdx, gpuVal, gpuB, gpuY_spmm);
        
        cudaEventRecord(stopGpu);
        cudaEventSynchronize(stopGpu);
        
        float iter_ms = 0.0f;
        cudaEventElapsedTime(&iter_ms, startGpu, stopGpu);
        total_gpu_ms_spmm += iter_ms;
    }

    printf("Average GPU SpMM Time (k=%d): %f ms\n", k, total_gpu_ms_spmm / runs);
    printf("GPU naive SpMM GFLOP/s: %f\n", 2.0 * csr.nnz * k / (1e6 * total_gpu_ms_spmm / runs));
  
    // SpMM correctness check
  
    cudaMemcpy(Y_gpu_result, gpuY_spmm, (size_t)csr.rows * k * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Checking SpMM Correctness: ");
    // total elements in Y is (csr.rows * k)
    checkResult(Y_cpu, Y_gpu_result, csr.rows * k);

    // kernel 5: CSR coalesced SpMM (a warp shares one row)
    // clear Y so the check below really tests kernel 5
    cudaMemset(gpuY_spmm, 0, (size_t)csr.rows * k * sizeof(float));

    // warmup (same number of threads as kernel 4, one per element of Y)
    csrCoalescedSpmm<<<spmmBlocks, threads>>>(csr.rows, k, gpuRowPtr, gpuColIdx, gpuVal, gpuB, gpuY_spmm);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    float totalCoalescedMs = 0.0f;
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(startGpu);

        csrCoalescedSpmm<<<spmmBlocks, threads>>>(csr.rows, k, gpuRowPtr, gpuColIdx, gpuVal, gpuB, gpuY_spmm);

        cudaEventRecord(stopGpu);
        cudaEventSynchronize(stopGpu);

        float iterMs = 0.0f;
        cudaEventElapsedTime(&iterMs, startGpu, stopGpu);
        totalCoalescedMs += iterMs;
    }

    float avgCoalescedMs = totalCoalescedMs / runs;
    printf("Average GPU CSR coalesced SpMM Time (k=%d): %f ms\n", k, avgCoalescedMs);
    printf("GPU CSR coalesced SpMM GFLOP/s: %f\n", 2.0 * csr.nnz * k / (1e6 * avgCoalescedMs));

    cudaMemcpy(Y_gpu_result, gpuY_spmm, (size_t)csr.rows * k * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Checking CSR coalesced SpMM Correctness: ");
    checkResult(Y_cpu, Y_gpu_result, csr.rows * k);

    cudaFree(gpuB);
    cudaFree(gpuY_spmm);
    // CSR arrays on GPU freed
    cudaFree(gpuRowPtr);
    cudaFree(gpuColIdx);
    cudaFree(gpuVal);

    
    free(B_cpu);
    free(Y_cpu);
    free(Y_gpu_result);


    cudaEventDestroy(startGpu);
    cudaEventDestroy(stopGpu);
    free(x_cpu); 
    free(y_cpu); 
    free(y_gpu_result);
    freeCoo(coo); 
    freeCsr(csr);

    return 0;
}
