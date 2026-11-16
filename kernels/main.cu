#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "matrix.h"
#include "cpu_ref.h"
#include "utils.h"


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
    
    // cudamalloc and cudaMemcpy to be here 

    cudaEvent_t startGpu, stopGpu;
    cudaEventCreate(&startGpu);
    cudaEventCreate(&stopGpu);

    int runs = 50;
    float total_gpu_ms = 0.0f;

    // Warmup run (gets the GPU awake and caches warmed up) as the first time GPU , it is slower to initialization overhead(loading data ,etc).
    // so we do a warmup run and ignore its time so the hw is ready

    // Kernel calling code part for warmup
    // run the kernel more than once and take the average as this smooths out randomm os background tasks and gives stable, reportable no. for benchmark 
    cudaDeviceSynchronize();

    // timed execution loop
    for(int i = 0; i < runs; i++) {
        cudaEventRecord(startGpu);
        
        // GpuSpmvKernel<<<grid, block>>>(...);
        
        cudaEventRecord(stopGpu);
        cudaEventSynchronize(stopGpu);
        
        float iter_ms = 0.0f;
        cudaEventElapsedTime(&iter_ms, startGpu, stopGpu);
        total_gpu_ms += iter_ms;
    }

    printf("Average GPU SpMV Time: %f ms\n", total_gpu_ms / runs);

    
    // correctness check and cleanup
    
    // cudamemcpy bakc the result from gpu
    
    printf("Checking SpMV Correctness: ");
    checkResult(y_cpu, y_gpu_result, csr.rows);

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