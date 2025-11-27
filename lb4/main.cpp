#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// CUDA + Thrust
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define TILE 16

// CUDA error check
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"\n";                                             \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// GPU kernel: tiled matrix multiplication (A * B = C), dimension NxN
__global__ void matMulTiledKernel(const double* __restrict__ A,
                                  const double* __restrict__ B,
                                  double* __restrict__ C, int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    double sum = 0.0;
    int numTiles = (N + TILE - 1) / TILE;
    for (int m = 0; m < numTiles; ++m) {
        int aCol = m * TILE + tx;
        int bRow = m * TILE + ty;
        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Simple QuickSort for CPU (in-place)
void quicksort_cpu(std::vector<int>& a, int lo, int hi) {
    if (lo >= hi)
        return;
    int i = lo, j = hi;
    int pivot = a[lo + (hi - lo) / 2];
    while (i <= j) {
        while (a[i] < pivot)
            ++i;
        while (a[j] > pivot)
            --j;
        if (i <= j) {
            std::swap(a[i], a[j]);
            ++i;
            --j;
        }
    }
    if (lo < j)
        quicksort_cpu(a, lo, j);
    if (i < hi)
        quicksort_cpu(a, i, hi);
}

int main(int argc, char** argv) {
    // Configurable sizes
    std::vector<int> mat_sizes = {
        1000, 1200, 1500, 1800,
        2000};  // можно расширять, следите за памятью GPU
    std::vector<int> sort_sizes = {100000, 200000, 500000, 1000000};

    int mat_iters = 3;  // повторов для усреднения
    int sort_iters = 5;

    std::cout << "Starting tests (CUDA). Make sure you run on machine with "
                 "NVIDIA GPU.\n";

    // Output CSV files
    // Now we only record GPU timings for matmul (CPU naive removed)
    std::ofstream fmat("results_matmul.csv");
    fmat << "size,iterations,gpu_time_ms\n";

    // Matmul tests (GPU only), using double precision matrices
    for (int N : mat_sizes) {
        std::cout << "Testing matmul N=" << N << "...\n";
        size_t elements = (size_t)N * N;
        size_t bytes = elements * sizeof(double);
        // Host allocate (double precision)
        std::vector<double> A(elements), B(elements), C_gpu(elements);

        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < elements; ++i) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }

        // GPU time including H2D, kernel, D2H
        double gpu_total_ms = 0.0;
        bool gpu_ok = true;
        try {
            double *dA = nullptr, *dB = nullptr, *dC = nullptr;
            CUDA_CHECK(cudaMalloc((void**)&dA, bytes));
            CUDA_CHECK(cudaMalloc((void**)&dB, bytes));
            CUDA_CHECK(cudaMalloc((void**)&dC, bytes));

            // prepare events for timing
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            dim3 block(TILE, TILE);
            dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

            for (int it = 0; it < mat_iters; ++it) {
                CUDA_CHECK(cudaEventRecord(start));
                CUDA_CHECK(
                    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(
                    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

                matMulTiledKernel<<<grid, block>>>(dA, dB, dC, N);
                CUDA_CHECK(cudaGetLastError());

                CUDA_CHECK(cudaMemcpy(C_gpu.data(), dC, bytes,
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                gpu_total_ms += ms;
            }

            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        } catch (...) {
            gpu_ok = false;
            std::cerr << "GPU run failed (out of memory or other CUDA error). "
                         "Skipping GPU for this size.\n";
        }

        if (gpu_ok) {
            double gpu_avg = gpu_total_ms / mat_iters;
            std::cout << "N=" << N << " gpu_avg_ms=" << gpu_avg << "\n";
            fmat << N << "," << mat_iters << "," << gpu_avg << "\n";
        } else {
            // write an empty gpu_time entry if GPU failed for this size
            fmat << N << "," << mat_iters << ",\n";
        }
    }
    fmat.close();

    // Sorting tests (unchanged)
    std::ofstream fsort("results_sort.csv");
    fsort << "size,iterations,cpu_time_ms,gpu_time_ms,speedup\n";

    for (int N : sort_sizes) {
        std::cout << "Testing sort N=" << N << "...\n";
        std::mt19937 rng(1234567);
        std::uniform_int_distribution<int> dist(0, 1000000000);

        double cpu_total_ms = 0.0;
        for (int it = 0; it < sort_iters; ++it) {
            std::vector<int> a(N);
            for (int i = 0; i < N; ++i)
                a[i] = dist(rng);
            auto t0 = std::chrono::high_resolution_clock::now();
            quicksort_cpu(a, 0, N - 1);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_total_ms +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        double cpu_avg = cpu_total_ms / sort_iters;

        // GPU sort with Thrust (including host->device and device->host)
        double gpu_total_ms = 0.0;
        for (int it = 0; it < sort_iters; ++it) {
            std::vector<int> a(N);
            for (int i = 0; i < N; ++i)
                a[i] = dist(rng);
            // H2D + sort + D2H
            auto t0 = std::chrono::high_resolution_clock::now();
            thrust::device_vector<int> dvec = a;  // copies host->device
            thrust::sort(dvec.begin(), dvec.end());
            thrust::copy(dvec.begin(), dvec.end(), a.begin());  // device->host
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_total_ms +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        double gpu_avg = gpu_total_ms / sort_iters;
        double speedup = cpu_avg / gpu_avg;
        std::cout << "sort N=" << N << " cpu_avg_ms=" << cpu_avg
                  << " gpu_avg_ms=" << gpu_avg << "\n";
        fsort << N << "," << sort_iters << "," << cpu_avg << "," << gpu_avg
              << "," << speedup << "\n";
    }
    fsort.close();

    std::cout
        << "Done. Results written to results_matmul.csv and results_sort.csv\n";

    return 0;
}
