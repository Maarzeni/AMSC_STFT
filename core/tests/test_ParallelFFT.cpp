#include <gtest/gtest.h>
#include "fft/ParallelFFT.hpp"
#include <array>
#include <complex>
#include <vector>
#include <omp.h>

using namespace stft;

static_assert(IsFFT<ParallelFFT>, "ParallelFFT must satisfy the IsFFT concept.");

class ParallelFFTTest : public ::testing::Test {
protected:
    static constexpr double tolerance = 1e-8;
};

// Round-trip accuracy must not depend on how many threads compute it.
TEST_F(ParallelFFTTest, CorrectnessDifferentThreadCounts) {
    const std::vector<std::complex<double>> reference = {
        {1.0, 0.1}, {-2.0, 0.5}, {3.1, -0.2}, {0.0, 1.0},
        {-1.5, -1.5}, {2.0, 0.0}, {0.5, -0.5}, {4.0, 2.0}
    };
    constexpr auto threadCounts = std::to_array<int>({1, 2, 4, 8});

    for (int threads : threadCounts) {
        if (threads > omp_get_max_threads()) continue;

        ParallelFFT fft(threads);
        auto data = reference;
        fft.forward(data);
        fft.inverse(data);

        for (std::size_t i = 0; i < data.size(); ++i) {
            EXPECT_NEAR(data[i].real(), reference[i].real(), tolerance)
                << "Failed at index " << i << " with " << threads << " threads";
            EXPECT_NEAR(data[i].imag(), reference[i].imag(), tolerance)
                << "Failed at index " << i << " with " << threads << " threads";
        }
    }
}

// Correctness at a size the other engines' tests do not exercise, using
// every thread the machine has.
TEST_F(ParallelFFTTest, LargeDataParallelProcessing) {
    constexpr std::size_t n = 65536;  // 2^16
    ParallelFFT fft(static_cast<std::size_t>(omp_get_max_threads()));
    std::vector<std::complex<double>> data(n, {1.0, 0.0});

    EXPECT_NO_THROW(fft.forward(data));

    // A constant signal has all its energy at DC; every other bin ~ 0.
    EXPECT_NEAR(data[0].real(), static_cast<double>(n), tolerance);
    for (std::size_t i = 1; i < std::min<std::size_t>(100, n); ++i) {
        EXPECT_NEAR(data[i].real(), 0.0, tolerance * 100);
        EXPECT_NEAR(data[i].imag(), 0.0, tolerance * 100);
    }
}

/**
 * @brief Several independent ParallelFFT instances, genuinely running at
 *        the same time on different threads.
 *
 * Unlike the tests above, the outer `#pragma omp parallel for` here is not
 * incidental: it is what makes this test actually exercise concurrent
 * construction and use of separate engine instances, rather than one
 * instance used sequentially. Assertions are not made from inside the
 * parallel region — GTest's macros are not meant to be called concurrently
 * from multiple threads — so each iteration instead records pass/fail in
 * `results`, checked once the region has finished.
 */
TEST_F(ParallelFFTTest, ThreadSafety) {
    constexpr std::size_t n = 4096;
    constexpr int numFfts = 8;

    std::vector<std::vector<std::complex<double>>> dataSets(numFfts);
    for (int i = 0; i < numFfts; ++i) {
        dataSets[i].resize(n);
        for (std::size_t j = 0; j < n; ++j) {
            const double re = static_cast<double>(i) + static_cast<double>(j);
            const double im = static_cast<double>(i) - static_cast<double>(j);
            dataSets[i][j] = {re, im};
        }
    }

    std::array<bool, numFfts> ok{};

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numFfts; ++i) {
        try {
            ParallelFFT localFft(2);
            localFft.forward(dataSets[i]);
            localFft.inverse(dataSets[i]);
            ok[i] = true;
        } catch (...) {
            ok[i] = false;
        }
    }

    for (int i = 0; i < numFfts; ++i) {
        EXPECT_TRUE(ok[i]) << "FFT instance " << i << " failed";
    }
}

// The power-of-two check must still fire when the engine is parallel-capable.
TEST_F(ParallelFFTTest, InvalidSizeCheckParallel) {
    ParallelFFT fft(static_cast<std::size_t>(omp_get_max_threads()));
    std::vector<std::complex<double>> data(3, {1.0, 0.0});  // not a power of 2

    EXPECT_THROW(fft.forward(data), std::invalid_argument);
    EXPECT_THROW(fft.inverse(data), std::invalid_argument);
}
