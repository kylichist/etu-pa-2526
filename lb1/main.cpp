#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

// ===================== Утилиты =====================
Matrix generateMatrix(size_t n, size_t m, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 10.0);
    Matrix A(n, std::vector<double>(m));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            A[i][j] = dist(gen);
    return A;
}

Matrix multiplyClassic(const Matrix& A, const Matrix& B) {
    size_t n = A.size(), m = B[0].size(), k = B.size();
    Matrix C(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            for (size_t t = 0; t < k; ++t)
                C[i][j] += A[i][t] * B[t][j];
    return C;
}

// ===================== Автоматический выбор блока =====================
size_t optimalBlockSize() {
    // приближённый размер кэша L1 (в байтах)
    const size_t L1_CACHE_SIZE = 32 * 1024;
    // double занимает 8 байт, три матрицы в блоке (A,B,C)
    size_t blockBytes = 3 * sizeof(double);
    size_t blockSize =
        static_cast<size_t>(std::sqrt(L1_CACHE_SIZE / blockBytes));
    return std::max<size_t>(8, blockSize);  // минимум 8
}

// ===================== Блочное умножение =====================
void blockMultiply(const Matrix& A, const Matrix& B, Matrix& C,
                   size_t blockSize, size_t rowStart, size_t rowEnd) {
    size_t n = A.size(), m = B[0].size(), k = B.size();
    for (size_t ii = rowStart; ii < rowEnd; ii += blockSize)
        for (size_t jj = 0; jj < m; jj += blockSize)
            for (size_t kk = 0; kk < k; kk += blockSize)
                for (size_t i = ii; i < std::min(ii + blockSize, rowEnd); ++i)
                    for (size_t j = jj; j < std::min(jj + blockSize, m); ++j)
                        for (size_t t = kk; t < std::min(kk + blockSize, k);
                             ++t)
                            C[i][j] += A[i][t] * B[t][j];
}

// ===================== std::thread =====================
Matrix multiplyThreaded(const Matrix& A, const Matrix& B, size_t threads) {
    size_t n = A.size();
    Matrix C(n, std::vector<double>(B[0].size(), 0.0));

    size_t blockSize = optimalBlockSize();
    size_t rowsPerThread = (n + threads - 1) / threads;
    std::vector<std::thread> pool;

    for (size_t t = 0; t < threads; ++t) {
        size_t start = t * rowsPerThread;
        size_t end = std::min(start + rowsPerThread, n);
        if (start >= n)
            break;
        pool.emplace_back(blockMultiply, std::cref(A), std::cref(B),
                          std::ref(C), blockSize, start, end);
    }

    for (auto& th : pool)
        th.join();
    return C;
}

// ===================== std::async =====================
Matrix multiplyAsync(const Matrix& A, const Matrix& B, size_t threads) {
    size_t n = A.size();
    Matrix C(n, std::vector<double>(B[0].size(), 0.0));

    size_t blockSize = optimalBlockSize();
    size_t rowsPerThread = (n + threads - 1) / threads;
    std::vector<std::future<void>> futures;

    for (size_t t = 0; t < threads; ++t) {
        size_t start = t * rowsPerThread;
        size_t end = std::min(start + rowsPerThread, n);
        if (start >= n)
            break;
        futures.emplace_back(std::async(std::launch::async, blockMultiply,
                                        std::cref(A), std::cref(B), std::ref(C),
                                        blockSize, start, end));
    }

    for (auto& f : futures)
        f.get();
    return C;
}

// ===================== Проверка и тестирование =====================
bool equal(const Matrix& A, const Matrix& B, double eps = 1e-6) {
    if (A.size() != B.size() || A[0].size() != B[0].size())
        return false;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j)
            if (std::abs(A[i][j] - B[i][j]) > eps)
                return false;
    return true;
}

// ===================== main =====================
int main() {
    std::vector<size_t> sizes = {1000, 1200, 1500, 1800,
                                 2000};  // размеры матриц
    std::vector<size_t> thread_counts = {1,  2,  4,  8,
                                         10, 12, 16, 24};  // число потоков
    std::cout << "size,threads,method,time_s\n";
    for (size_t n : sizes) {
        Matrix A = generateMatrix(n, n, 1);
        Matrix B = generateMatrix(n, n, 2);

        // Classic
        auto start = std::chrono::high_resolution_clock::now();
        Matrix C_ref = multiplyClassic(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        double t_classic = std::chrono::duration<double>(end - start).count();
        std::cout << n << ",0,Classic," << t_classic
                  << "\n";  // 0 потоков для classic

        for (size_t th : thread_counts) {
            // Threads
            start = std::chrono::high_resolution_clock::now();
            Matrix C_thr = multiplyThreaded(A, B, th);
            end = std::chrono::high_resolution_clock::now();
            double t_threads =
                std::chrono::duration<double>(end - start).count();
            std::cout << n << "," << th << ",Threads," << t_threads << "\n";

            // Async
            start = std::chrono::high_resolution_clock::now();
            Matrix C_async = multiplyAsync(A, B, th);
            end = std::chrono::high_resolution_clock::now();
            double t_async = std::chrono::duration<double>(end - start).count();
            std::cout << n << "," << th << ",Async," << t_async << "\n";

            // Проверка корректности (опционально)
            if (!equal(C_ref, C_thr))
                std::cerr << "Threaded FAIL size=" << n << " th=" << th << "\n";
            if (!equal(C_ref, C_async))
                std::cerr << "Async FAIL size=" << n << " th=" << th << "\n";
        }
    }
}
