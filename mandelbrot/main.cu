#include "CImg.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
using namespace cimg_library;

#define MAX_ITERS 10000
#define X_PIXELS 1024
#define Y_PIXELS 1024
#define N (X_PIXELS * Y_PIXELS) // problem size

// Demonstrating the computational speed up of a simple Mandelbrot Escape Time Algorithm using GPU Parallelization.

int colors[16][3]
    = {{66, 30, 15}, {25, 7, 26}, {9, 1, 47}, {4, 4, 73}, {0, 7, 100}, {12, 44, 138}, {24, 82, 177}, {57, 125, 209},
        {134, 181, 229}, {211, 236, 248}, {241, 233, 191}, {248, 201, 95}, {255, 170, 0}, {204, 128, 0}, {153, 87, 0}, {106, 52, 3}};

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Note: Mandelbrot X Scale (-2.00, 0.47)
// Note: Mandelbrot Y Scale (-1.12, 1.12)
// Note that CImg has a draw_mandelbrot function already that we will not use here as it takes advantage of some parllelization. This will be a purely single threaded function.
CImg<unsigned char> mandelbrot_cpu() {
    CImg<unsigned char> cpu_mandelbrot(X_PIXELS, Y_PIXELS, 1, 3, 0); 
    
    for (int ix = 0; ix < X_PIXELS + 1; ix++) {
        for (int iy = 0; iy < Y_PIXELS + 1; iy++) {
            float x_scaled = -2.00 + ((0.47 + 2.00) / (X_PIXELS)) * ix;
            float y_scaled = -1.12 + ((2.24) / (Y_PIXELS)) * iy;
            int iteration = 0;
            float x = 0.0;
            float y = 0.0;

            while (x * x + y * y <= 4 && iteration < MAX_ITERS) {
                float x_temp = x * x - y * y + x_scaled;
                y = 2 * x * y + y_scaled;
                x = x_temp;
                iteration++;
            }
            cpu_mandelbrot.draw_point(ix, iy, colors[iteration % 16]);

        }
    }
    return cpu_mandelbrot;
}

__global__ void mandelbrot_gpu(int *out, int x_pixels, int y_pixels, int max_iters, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int ix = threadIdx.x; 
        int iy = blockIdx.x;
        float x_scaled = -2.00 + ((0.47 + 2.00) / (x_pixels)) * ix;
        float y_scaled = -1.12 + ((2.24) / (y_pixels)) * iy;
        int iteration = 0;
        float x = 0.0;
        float y = 0.0;

        while (x * x + y * y <= 4 && iteration < max_iters) {
            float x_temp = x * x - y * y + x_scaled;
            y = 2 * x * y + y_scaled;
            x = x_temp;
            iteration++;
            }
        out[tid] = iteration % 16;
    }
}

int main() {
    int  *out;
    int *d_out;


    // allocate host memory
    out = (int*)malloc(sizeof(int) * N);

    
    // allocate device memory
    cudaMalloc((void**)&d_out, sizeof(int) * N);

    

    // Excecute Kernel
    int block_size = 1024; // threads per block
    int grid_size = ((N + block_size) / block_size); // blocks to evenly divide the problem

    mandelbrot_gpu<<<grid_size, block_size>>>(d_out, X_PIXELS, Y_PIXELS, MAX_ITERS, N);

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Draw Pixels
    CImg<unsigned char> gpu_mandelbrot(X_PIXELS, Y_PIXELS, 1, 3, 0); 
    for (int i = 0; i < N + 1; i++) {
        gpu_mandelbrot.draw_point(i % X_PIXELS, i / X_PIXELS, colors[out[i]]);
    }
    CImgDisplay gpu_disp(gpu_mandelbrot,"GPU_MandelBrot");
    CImg<unsigned char> cpu_mandelbrot = mandelbrot_cpu();
    CImgDisplay cpu_disp(cpu_mandelbrot,"CPU_MandelBrot");

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        mandelbrot_cpu();
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

     // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        mandelbrot_gpu<<<grid_size, block_size>>>(d_out, X_PIXELS, Y_PIXELS, MAX_ITERS, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    while (!cpu_disp.is_closed() && !gpu_disp.is_closed()) {
        cpu_disp.wait();
        gpu_disp.wait();
    }

    // Free memory
    free(out);
    cudaFree(d_out);
    
    return 0;
}