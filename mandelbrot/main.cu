#include "CImg.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
using namespace cimg_library;
// Demonstrating the computational speed up of a simple Mandelbrot Escape Time Algorithm by using GPU Parallelization.

int colors[16][3]
    = {{66, 30, 15}, {25, 7, 26}, {9, 1, 47}, {4, 4, 73}, {0, 7, 100}, {12, 44, 138}, {24, 82, 177}, {57, 125, 209},
        {134, 181, 229}, {211, 236, 248}, {241, 233, 191}, {248, 201, 95}, {255, 170, 0}, {204, 128, 0}, {153, 87, 0}, {106, 52, 3}};


// Note: Mandelbrot X Scale (-2.00, 0.47)
// Note: Mandelbrot Y Scale (-1.12, 1.12)
// Note that CImg has a draw_mandelbrot function already that we will not use here as it takes advantage of some parllelization. This will be a purely single threaded function.
void mandelbrot_cpu() {
    int x_pixels = 2470; // num x-pixels
    int y_pixels = 2240; // num y-pixels
    CImg<unsigned char> mandelbrot(x_pixels, y_pixels, 1, 3, 0); 
    
    for (int ix = 0; ix < x_pixels + 1; ix++) {
        for (int iy = 0; iy < y_pixels + 1; iy++) {
            float x_scaled = -2.00 + ((0.47 + 2.00) / (x_pixels)) * ix;
            float y_scaled = -1.12 + ((2.24) / (y_pixels)) * iy;
            int iteration = 0;
            int max_iteration = 10000;
            float x = 0.0;
            float y = 0.0;

            while (x * x + y * y <= 4 && iteration < max_iteration) {
                float x_temp = x * x - y * y + x_scaled;
                y = 2 * x * y + y_scaled;
                x = x_temp;
                iteration++;
            }
            mandelbrot.draw_point(ix, iy, colors[iteration % 16]);

        }
    }
    CImgDisplay main_disp(mandelbrot,"Test");
    while (!main_disp.is_closed()) {
        main_disp.wait();
    }
}

__global__ void mandelbrot_gpu() {

}

int main() {
    mandelbrot_cpu();
    return 0;
}