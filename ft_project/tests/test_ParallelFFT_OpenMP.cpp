#include <gtest/gtest.h>
#include "fft/ParallelFFT.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>

using namespace amsc_stft;

// ==========================================
// 1. COMPILE-TIME CONCEPT CHECK
// ==========================================
static_assert(IsFFT<ParallelFFT>, "ERROR: ParallelFFT must satisfy the IsFFT concept!");

// ==========================================
// 2. FUNCTIONAL TESTS WITH OpenMP
// ==========================================

class ParallelFFTOpenMPTest : public ::testing::Test {
protected:
    // Tolerance for floating-point comparisons
    const double TOL = 1e-8;
};

// Test 1: Verify Correctness with Multiple Thread Configurations
// Tests that the FFT produces correct results regardless of thread count
TEST_F(ParallelFFTOpenMPTest, CorrectnessDifferentThreadCounts) {
    std::vector<std::complex<double>> reference = {
        {1.0, 0.1}, {-2.0, 0.5}, {3.1, -0.2}, {0.0, 1.0},
        {-1.5, -1.5}, {2.0, 0.0}, {0.5, -0.5}, {4.0, 2.0}
    };

    // Test with different thread counts
    int thread_counts[] = {1, 2, 4, 8};

    for (int num_threads : thread_counts) {
        if (num_threads > omp_get_max_threads()) {
            continue;
        }

        ParallelFFT fft(num_threads);
        auto data = reference;

        // Forward and inverse transform
        fft.forward(data);
        fft.inverse(data);

        // Verify round-trip accuracy
        for (size_t i = 0; i < data.size(); ++i) {
            EXPECT_NEAR(data[i].real(), reference[i].real(), TOL)
                << "Failed at index " << i << " with " << num_threads << " threads";
            EXPECT_NEAR(data[i].imag(), reference[i].imag(), TOL)
                << "Failed at index " << i << " with " << num_threads << " threads";
        }
    }
}

// Test 2: Large Data Parallel Processing
// Ensures the FFT handles large arrays correctly with different thread counts
TEST_F(ParallelFFTOpenMPTest, LargeDataParallelProcessing) {
    const size_t n = 65536; // 2^16 - Very large array
    int num_threads = omp_get_max_threads();

    ParallelFFT fft(num_threads);
    std::vector<std::complex<double>> data(n, {1.0, 0.0});

    EXPECT_NO_THROW({
        fft.forward(data);
    });

    // Verify DC component
    EXPECT_NEAR(data[0].real(), static_cast<double>(n), TOL);

    // All other frequency bins should be ~0
    for (size_t i = 1; i < std::min(size_t(100), n); ++i) {
        EXPECT_NEAR(data[i].real(), 0.0, TOL * 100);
        EXPECT_NEAR(data[i].imag(), 0.0, TOL * 100);
    }
}

// Test 3: Round-trip with Multiple Threads
// Ensures Forward -> Inverse is accurate with different thread counts
TEST_F(ParallelFFTOpenMPTest, RoundTripMultipleThreads) {
    std::vector<std::complex<double>> original = {
        {1.0, 0.1}, {-2.0, 0.5}, {3.1, -0.2}, {0.0, 1.0},
        {-1.5, -1.5}, {2.0, 0.0}, {0.5, -0.5}, {4.0, 2.0}
    };

    int max_threads = omp_get_max_threads();

    // Test with different thread configurations
    #pragma omp parallel for schedule(static)
    for (int t = 1; t <= max_threads; ++t) {
        ParallelFFT fft(t);
        auto data = original;

        fft.forward(data);
        fft.inverse(data);

        // Verify correctness
        for (size_t i = 0; i < data.size(); ++i) {
            EXPECT_NEAR(data[i].real(), original[i].real(), TOL);
            EXPECT_NEAR(data[i].imag(), original[i].imag(), TOL);
        }
    }
}

// Test 4: Thread Safety Check
// Verifies that multiple ParallelFFT instances can work in parallel without issues
TEST_F(ParallelFFTOpenMPTest, ThreadSafety) {
    const size_t n = 4096;
    const int num_ffts = 8;
    std::vector<std::vector<std::complex<double>>> data_sets(num_ffts);

    // Initialize data
    for (int i = 0; i < num_ffts; ++i) {
        data_sets[i].resize(n);
        for (size_t j = 0; j < n; ++j) {
            data_sets[i][j] = std::complex<double>(i + j, i - j);
        }
    }

    bool results[num_ffts];

    // Create multiple FFT instances and process in parallel
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_ffts; ++i) {
        try {
            ParallelFFT local_fft(2); // Use 2 threads per instance
            local_fft.forward(data_sets[i]);
            local_fft.inverse(data_sets[i]);
            results[i] = true;
        } catch (...) {
            results[i] = false;
        }
    }

    // Verify all instances completed successfully
    for (int i = 0; i < num_ffts; ++i) {
        EXPECT_TRUE(results[i]) << "FFT instance " << i << " failed";
    }
}

// Test 5: Invalid Size Check with Parallel Execution
// Verifies that invalid sizes are caught even in parallel context
TEST_F(ParallelFFTOpenMPTest, InvalidSizeCheckParallel) {
    ParallelFFT fft(omp_get_max_threads());

    // Size 3 is not a power of 2
    std::vector<std::complex<double>> data(3, {1.0, 0.0});

    EXPECT_THROW({
        fft.forward(data);
    }, std::invalid_argument);

    EXPECT_THROW({
        fft.inverse(data);
    }, std::invalid_argument);
}

// Test 6: Hybrid Parallelism Simulation
// Simulates hybrid workload with multiple FFT operations in parallel
TEST_F(ParallelFFTOpenMPTest, HybridParallelismSimulation) {
    const size_t n = 8192; // 2^13
    const int num_blocks = 4;

    std::vector<std::vector<std::complex<double>>> blocks(num_blocks);

    // Initialize each block with different data
    for (int b = 0; b < num_blocks; ++b) {
        blocks[b].resize(n);
        for (size_t i = 0; i < n; ++i) {
            double phase = 2.0 * M_PI * (b + 1) * i / n;
            blocks[b][i] = std::complex<double>(std::sin(phase), std::cos(phase));
        }
    }

    ParallelFFT fft(omp_get_max_threads());

    // Process blocks in parallel using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < num_blocks; ++b) {
        fft.forward(blocks[b]);
    }

    // Verify processing was successful
    for (int b = 0; b < num_blocks; ++b) {
        EXPECT_NO_THROW({
            fft.inverse(blocks[b]);
        });
    }
}
