#include <gtest/gtest.h>
#include "fft/ParallelFFT.hpp"
#include <vector>
#include <complex>
#include <cmath>

using namespace stft;

// ==========================================
// 1. COMPILE-TIME CONCEPT CHECK
// ==========================================
// If this fails, the code won't even compile.
static_assert(IsFFT<ParallelFFT>, "ERROR: ParallelFFT must satisfy the IsFFT concept!");


// ==========================================
// 2. RUNTIME TESTS
// ==========================================

class ParallelFFTTest : public ::testing::Test {
protected:
    // Tolerance for floating-point comparisons
    const double TOL = 1e-9;
};

// Test 1: Forward transform of an impulse (Delta function)
// A single pulse at index 0 should result in a flat frequency spectrum of 1s.
TEST_F(ParallelFFTTest, ForwardTransformDelta) {
    ParallelFFT fft; // Uses default hardware concurrency
    
    // Time domain: [1, 0, 0, 0]
    std::vector<std::complex<double>> data = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };

    fft.forward(data);

    // Frequency domain expect: [1, 1, 1, 1]
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(data[i].real(), 1.0, TOL) << "Real part failed at index " << i;
        EXPECT_NEAR(data[i].imag(), 0.0, TOL) << "Imaginary part failed at index " << i;
    }
}

// Test 2: Round-trip (Forward -> Inverse) on a random-looking signal
// This proves that your inverse logic and 1/N normalization are perfect.
TEST_F(ParallelFFTTest, RoundTripIdentity) {
    ParallelFFT fft(4); // Explicitly request 4 threads for testing
    
    std::vector<std::complex<double>> original = {
        {1.0, 0.1}, {-2.0, 0.5}, {3.1, -0.2}, {0.0, 1.0},
        {-1.5, -1.5}, {2.0, 0.0}, {0.5, -0.5}, {4.0, 2.0}
    };
    
    // Copy the original data because FFT works in-place
    auto data = original;

    // Go to frequency domain and immediately back to time domain
    fft.forward(data);
    fft.inverse(data);

    ASSERT_EQ(data.size(), original.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(data[i].real(), original[i].real(), TOL);
        EXPECT_NEAR(data[i].imag(), original[i].imag(), TOL);
    }
}

// Test 3: Large Array Execution
// Forces the algorithm to trigger the multi-threaded butterfly stages (len >= 16)
TEST_F(ParallelFFTTest, LargeArrayExecution) {
    ParallelFFT fft; 
    
    const size_t n = 1024; // 2^10
    std::vector<std::complex<double>> data(n, {1.0, 0.0}); // DC signal
    
    // Should execute safely without hanging or throwing thread errors
    EXPECT_NO_THROW({
        fft.forward(data);
    });

    // The DC component (index 0) of a signal of all 1s should be N
    EXPECT_NEAR(data[0].real(), static_cast<double>(n), TOL);
    
    // All other frequency bins should be 0
    for (size_t i = 1; i < n; ++i) {
        EXPECT_NEAR(data[i].real(), 0.0, TOL);
        EXPECT_NEAR(data[i].imag(), 0.0, TOL);
    }
}

// Test 4: BaseFFT Protection Check
// Ensure the CRTP base class throws an error if N is not a power of 2
TEST_F(ParallelFFTTest, ThrowsOnInvalidSize) {
    ParallelFFT fft;
    
    // Size 3 is not a power of 2
    std::vector<std::complex<double>> data(3, {1.0, 0.0});
    
    EXPECT_THROW({
        fft.forward(data);
    }, std::invalid_argument);

    EXPECT_THROW({
        fft.inverse(data);
    }, std::invalid_argument);
}