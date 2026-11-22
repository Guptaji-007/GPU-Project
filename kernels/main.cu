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
    printf("Average GPU SpMV Time: %f ms\n", avgMs);

    // each non-zero does one multiply and one add = 2 flops
    float gflops = 2.0f * csr.nnz / (avgMs * 1e6f);
    printf("GPU SpMV GFLOP/s: %f\n", gflops);


    // correctness check and cleanup

    // copy the result back from gpu
    cudaMemcpy(y_gpu_result, gpuY, csr.rows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Checking SpMV Correctness: ");
    checkResult(y_cpu, y_gpu_result, csr.rows);

    // SpMM will need the matrix on the gpu again later, but free it for now
    cudaFree(gpuRowPtr);
    cudaFree(gpuColIdx);
    cudaFree(gpuVal);
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

    float total_gpu_ms_spmm = 0.0f;

    // warmup run
    // yourGpuSpmmKernel<<<grid, block>>>(...);
    cudaDeviceSynchronize();

    // timed execution loop
    for(int i = 0; i < runs; i++) {
        cudaEventRecord(startGpu);
        
        // GpuSpmmKernel<<<grid, block>>>(...);
        
        cudaEventRecord(stopGpu);
        cudaEventSynchronize(stopGpu);
        
        float iter_ms = 0.0f;
        cudaEventElapsedTime(&iter_ms, startGpu, stopGpu);
        total_gpu_ms_spmm += iter_ms;
    }

    printf("Average GPU SpMM Time (k=%d): %f ms\n", k, total_gpu_ms_spmm / runs);

  
    // SpMM correctness check
  
    // (cudamemcpy result from gpu
    
    printf("Checking SpMM Correctness: ");
    // total elements in Y is (csr.rows * k)
    checkResult(Y_cpu, Y_gpu_result, csr.rows * k);

    
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