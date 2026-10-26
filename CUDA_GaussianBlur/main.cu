#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h> // For basic CPU timing (optional)

// Define these before including stb_image.h and stb_image_write.h
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- CUDA Error Handling ---
// Macro to wrap CUDA calls for error checking
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error = %d (%s) at %s:%d '%s'\n",
                static_cast<int>(result), cudaGetErrorString(result), file, line, func);
        cudaDeviceReset(); // Reset device state
        exit(EXIT_FAILURE);
    }
}

// --- Gaussian Kernel Generation ---
// Generates a 1D Gaussian kernel
void generateGaussianKernel(float *kernel, int kernelRadius, float sigma) {
    float sum = 0.0f;
    int kernelSize = 2 * kernelRadius + 1;
    for (int i = -kernelRadius; i <= kernelRadius; ++i) {
        float val = expf(-(i * i) / (2.0f * sigma * sigma));
        kernel[i + kernelRadius] = val;
        sum += val;
    }
    // Normalize the kernel
    for (int i = 0; i < kernelSize; ++i) {
        kernel[i] /= sum;
        // printf("Kernel[%d] = %f\n", i, kernel[i]); // Debug print
    }
}

// --- CUDA Kernel for Gaussian Blur ---
// Applies separable Gaussian blur (horizontal then vertical)
// This version processes one pixel per thread.
// Assumes RGB image (3 channels)

// Simple clamp function for boundary handling
__device__ int clamp(int idx, int minVal, int maxVal) {
    return max(minVal, min(idx, maxVal));
}

// Kernel for horizontal blur pass
__global__ void gaussianBlurHorizontal(unsigned char *d_out, const unsigned char *d_in,
                                       int width, int height,
                                       const float *d_kernel, int kernelRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
        int kernelSize = 2 * kernelRadius + 1;

        // Apply 1D horizontal kernel
        for (int k = -kernelRadius; k <= kernelRadius; ++k) {
            int ix = clamp(x + k, 0, width - 1); // Clamp coordinates to image bounds
            int idx = (y * width + ix) * 3;       // Index for input pixel (RGB)
            float weight = d_kernel[k + kernelRadius];

            sumR += d_in[idx + 0] * weight;
            sumG += d_in[idx + 1] * weight;
            sumB += d_in[idx + 2] * weight;
        }

        int outIdx = (y * width + x) * 3;
        d_out[outIdx + 0] = (unsigned char)clamp(sumR, 0.0f, 255.0f);
        d_out[outIdx + 1] = (unsigned char)clamp(sumG, 0.0f, 255.0f);
        d_out[outIdx + 2] = (unsigned char)clamp(sumB, 0.0f, 255.0f);
    }
}

// Kernel for vertical blur pass
__global__ void gaussianBlurVertical(unsigned char *d_out, const unsigned char *d_in,
                                     int width, int height,
                                     const float *d_kernel, int kernelRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
        int kernelSize = 2 * kernelRadius + 1;

        // Apply 1D vertical kernel
        for (int k = -kernelRadius; k <= kernelRadius; ++k) {
            int iy = clamp(y + k, 0, height - 1); // Clamp coordinates to image bounds
            int idx = (iy * width + x) * 3;        // Index for input pixel (RGB)
            float weight = d_kernel[k + kernelRadius];

            sumR += d_in[idx + 0] * weight;
            sumG += d_in[idx + 1] * weight;
            sumB += d_in[idx + 2] * weight;
        }

        int outIdx = (y * width + x) * 3;
        d_out[outIdx + 0] = (unsigned char)clamp(sumR, 0.0f, 255.0f);
        d_out[outIdx + 1] = (unsigned char)clamp(sumG, 0.0f, 255.0f);
        d_out[outIdx + 2] = (unsigned char)clamp(sumB, 0.0f, 255.0f);
    }
}



// --- CPU reference implementation for comparison ---
// Uses the same normalized 1D Gaussian kernel, edge clamping, and
// horizontal-then-vertical passes as the GPU implementation.
static inline int clampHost(int idx, int minVal, int maxVal) {
    return (idx < minVal) ? minVal : ((idx > maxVal) ? maxVal : idx);
}

void gaussianBlurCPUHorizontal(unsigned char *out, const unsigned char *in,
                               int width, int height,
                               const float *kernel, int kernelRadius) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            for (int k = -kernelRadius; k <= kernelRadius; ++k) {
                int ix = clampHost(x + k, 0, width - 1);
                int idx = (y * width + ix) * 3;
                float weight = kernel[k + kernelRadius];
                sumR += in[idx + 0] * weight;
                sumG += in[idx + 1] * weight;
                sumB += in[idx + 2] * weight;
            }
            int outIdx = (y * width + x) * 3;
            out[outIdx + 0] = (unsigned char)clampHost((int)fminf(fmaxf(sumR, 0.0f), 255.0f), 0, 255);
            out[outIdx + 1] = (unsigned char)clampHost((int)fminf(fmaxf(sumG, 0.0f), 255.0f), 0, 255);
            out[outIdx + 2] = (unsigned char)clampHost((int)fminf(fmaxf(sumB, 0.0f), 255.0f), 0, 255);
        }
    }
}

void gaussianBlurCPUVertical(unsigned char *out, const unsigned char *in,
                             int width, int height,
                             const float *kernel, int kernelRadius) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            for (int k = -kernelRadius; k <= kernelRadius; ++k) {
                int iy = clampHost(y + k, 0, height - 1);
                int idx = (iy * width + x) * 3;
                float weight = kernel[k + kernelRadius];
                sumR += in[idx + 0] * weight;
                sumG += in[idx + 1] * weight;
                sumB += in[idx + 2] * weight;
            }
            int outIdx = (y * width + x) * 3;
            out[outIdx + 0] = (unsigned char)clampHost((int)fminf(fmaxf(sumR, 0.0f), 255.0f), 0, 255);
            out[outIdx + 1] = (unsigned char)clampHost((int)fminf(fmaxf(sumG, 0.0f), 255.0f), 0, 255);
            out[outIdx + 2] = (unsigned char)clampHost((int)fminf(fmaxf(sumB, 0.0f), 255.0f), 0, 255);
        }
    }
}


// --- Main Function: CPU/GPU benchmark and correctness comparison ---
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_image> <sigma>\n", argv[0]);
        fprintf(stderr, "Example: %s data/input.jpg 2.0\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *inputFile = argv[1];
    float sigma = (float)atof(argv[2]);
    if (!(sigma > 0.0f) || !isfinite(sigma)) {
        fprintf(stderr, "Error: Sigma must be a finite positive number.\n");
        return EXIT_FAILURE;
    }

    int width, height, originalChannels;
    unsigned char *h_img_in = stbi_load(inputFile, &width, &height, &originalChannels, 3);
    if (!h_img_in) {
        fprintf(stderr, "Error loading image '%s': %s\n", inputFile, stbi_failure_reason());
        return EXIT_FAILURE;
    }

    const int channels = 3;
    size_t imgSize = (size_t)width * (size_t)height * channels;
    int kernelRadius = (int)ceilf(3.0f * sigma);
    if (kernelRadius < 1) kernelRadius = 1;
    int kernelSize = 2 * kernelRadius + 1;
    size_t kernelSizeBytes = (size_t)kernelSize * sizeof(float);

    unsigned char *cpuIntermediate = (unsigned char *)malloc(imgSize);
    unsigned char *cpuOutput = (unsigned char *)malloc(imgSize);
    unsigned char *gpuOutput = (unsigned char *)malloc(imgSize);
    float *h_kernel = (float *)malloc(kernelSizeBytes);
    if (!cpuIntermediate || !cpuOutput || !gpuOutput || !h_kernel) {
        fprintf(stderr, "Error allocating host memory.\n");
        free(cpuIntermediate); free(cpuOutput); free(gpuOutput); free(h_kernel);
        stbi_image_free(h_img_in);
        return EXIT_FAILURE;
    }

    generateGaussianKernel(h_kernel, kernelRadius, sigma);

    // CPU timing: both separable passes, same kernel and border behavior as GPU.
    clock_t cpuStart = clock();
    gaussianBlurCPUHorizontal(cpuIntermediate, h_img_in, width, height, h_kernel, kernelRadius);
    gaussianBlurCPUVertical(cpuOutput, cpuIntermediate, width, height, h_kernel, kernelRadius);
    clock_t cpuEnd = clock();
    double cpuMs = 1000.0 * (double)(cpuEnd - cpuStart) / (double)CLOCKS_PER_SEC;

    unsigned char *d_img_in = NULL, *d_img_out = NULL, *d_intermediate = NULL;
    float *d_kernel = NULL;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaMalloc((void **)&d_img_in, imgSize));
    checkCudaErrors(cudaMalloc((void **)&d_img_out, imgSize));
    checkCudaErrors(cudaMalloc((void **)&d_intermediate, imgSize));
    checkCudaErrors(cudaMalloc((void **)&d_kernel, kernelSizeBytes));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // End-to-end timing includes host-to-device copies, both GPU passes,
    // and device-to-host copy. A separate event measures kernel-only time.
    float gpuKernelMs = 0.0f;
    float gpuTotalMs = 0.0f;

    checkCudaErrors(cudaEventRecord(start));
    checkCudaErrors(cudaMemcpy(d_img_in, h_img_in, imgSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, kernelSizeBytes, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Warm-up launches, excluded from the reported kernel timing.
    gaussianBlurHorizontal<<<gridSize, blockSize>>>(d_intermediate, d_img_in, width, height, d_kernel, kernelRadius);
    checkCudaErrors(cudaGetLastError());
    gaussianBlurVertical<<<gridSize, blockSize>>>(d_img_out, d_intermediate, width, height, d_kernel, kernelRadius);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t kernelStart, kernelStop;
    checkCudaErrors(cudaEventCreate(&kernelStart));
    checkCudaErrors(cudaEventCreate(&kernelStop));
    checkCudaErrors(cudaEventRecord(kernelStart));
    gaussianBlurHorizontal<<<gridSize, blockSize>>>(d_intermediate, d_img_in, width, height, d_kernel, kernelRadius);
    checkCudaErrors(cudaGetLastError());
    gaussianBlurVertical<<<gridSize, blockSize>>>(d_img_out, d_intermediate, width, height, d_kernel, kernelRadius);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(kernelStop));
    checkCudaErrors(cudaEventSynchronize(kernelStop));
    checkCudaErrors(cudaEventElapsedTime(&gpuKernelMs, kernelStart, kernelStop));

    checkCudaErrors(cudaMemcpy(gpuOutput, d_img_out, imgSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&gpuTotalMs, start, stop));

    size_t mismatchedValues = 0;
    unsigned int maxAbsDifference = 0;
    unsigned long long totalAbsDifference = 0;
    for (size_t i = 0; i < imgSize; ++i) {
        unsigned int diff = (cpuOutput[i] > gpuOutput[i])
            ? (unsigned int)(cpuOutput[i] - gpuOutput[i])
            : (unsigned int)(gpuOutput[i] - cpuOutput[i]);
        if (diff != 0) ++mismatchedValues;
        if (diff > maxAbsDifference) maxAbsDifference = diff;
        totalAbsDifference += diff;
    }

    int cpuSaved = stbi_write_png("blur_cpu.png", width, height, channels, cpuOutput, width * channels);
    int gpuSaved = stbi_write_png("blur_gpu.png", width, height, channels, gpuOutput, width * channels);

    printf("\n===== Gaussian Blur CPU vs GPU Benchmark =====\n");
    printf("Input image: %s\n", inputFile);
    printf("Image size: %d x %d pixels (RGB)\n", width, height);
    printf("Sigma: %.3f | Kernel radius: %d | Kernel size: %d\n", sigma, kernelRadius, kernelSize);
    printf("CPU time (two passes): %.3f ms\n", cpuMs);
    printf("GPU kernel time (two passes): %.3f ms\n", gpuKernelMs);
    printf("GPU end-to-end time (copies + kernels): %.3f ms\n", gpuTotalMs);
    if (gpuKernelMs > 0.0f)
        printf("CPU / GPU-kernel speedup: %.2fx\n", cpuMs / gpuKernelMs);
    if (gpuTotalMs > 0.0f)
        printf("CPU / GPU-end-to-end speedup: %.2fx\n", cpuMs / gpuTotalMs);
    printf("Different channel values: %zu / %zu\n", mismatchedValues, imgSize);
    printf("Maximum absolute channel difference: %u\n", maxAbsDifference);
    printf("Mean absolute channel difference: %.6f\n", (double)totalAbsDifference / (double)imgSize);
    printf("CPU output: %s\n", cpuSaved ? "blur_cpu.png (saved)" : "save failed");
    printf("GPU output: %s\n", gpuSaved ? "blur_gpu.png (saved)" : "save failed");

    checkCudaErrors(cudaEventDestroy(kernelStart));
    checkCudaErrors(cudaEventDestroy(kernelStop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(d_img_in));
    checkCudaErrors(cudaFree(d_img_out));
    checkCudaErrors(cudaFree(d_intermediate));
    checkCudaErrors(cudaFree(d_kernel));
    free(cpuIntermediate); free(cpuOutput); free(gpuOutput); free(h_kernel);
    stbi_image_free(h_img_in);

    return (cpuSaved && gpuSaved) ? EXIT_SUCCESS : EXIT_FAILURE;
}
