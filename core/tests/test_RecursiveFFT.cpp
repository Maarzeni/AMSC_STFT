#include <gtest/gtest.h>
#include "fft/RecursiveFFT.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <numbers>

namespace stft {
namespace testing {

// Fixture class for RecursiveFFT tests
class RecursiveFFTTest : public ::testing::Test {
protected:
    RecursiveFFT fft;
    const double tolerance = 1e-9; // Tolerance for floating-point comparison

    // Helper method to compare two complex vectors
    void compareVectors(const std::vector<std::complex<double>>& v1, 
                        const std::vector<std::complex<double>>& v2) {
        ASSERT_EQ(v1.size(), v2.size()) << "Vectors have different sizes.";
        
        for (size_t i = 0; i < v1.size(); ++i) {
            EXPECT_NEAR(v1[i].real(), v2[i].real(), tolerance)
                << "Real part mismatch at index " << i;

            EXPECT_NEAR(v1[i].imag(), v2[i].imag(), tolerance)
                << "Imaginary part mismatch at index " << i;
        }
    }
};

// 1. Test with empty input
TEST_F(RecursiveFFTTest, HandlesEmptyInput) {
    std::vector<std::complex<double>> data;

    EXPECT_THROW(fft.forward(data), std::invalid_argument);

    EXPECT_TRUE(data.empty());
}

// 2. Test with non-power-of-two size
// (must throw an exception from BaseFFT)
TEST_F(RecursiveFFTTest, ThrowsOnNonPowerOfTwo) {
    std::vector<std::complex<double>> data(3, {1.0, 0.0}); // N = 3
    
    EXPECT_THROW({
        fft.forward(data);
    }, std::invalid_argument);

    EXPECT_THROW({
        fft.inverse(data);
    }, std::invalid_argument);
}

// 3. Dirac impulse transform test
// An impulse [1, 0, 0, 0] in the time domain
// corresponds to [1, 1, 1, 1] in the frequency domain
TEST_F(RecursiveFFTTest, ImpulseResponse) {
    std::vector<std::complex<double>> time_data = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };
    
    std::vector<std::complex<double>> expected_freq_data = {
        {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}
    };

    fft.forward(time_data); // In-place transformation

    compareVectors(time_data, expected_freq_data);
}

// 4. Reversibility test: IFFT(FFT(x)) == x
// We generate a sinusoidal test signal, apply FFT and then IFFT.
// The final result must match the original signal.
TEST_F(RecursiveFFTTest, ForwardInverseIdentity) {
    const size_t N = 1024; // Power of 2

    std::vector<std::complex<double>> original_data(N);
    
    // Generate a signal composed of multiple frequencies
    for (size_t i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / N;

        double val =
            std::sin(2.0 * std::numbers::pi * 10.0 * t) +
            0.5 * std::cos(2.0 * std::numbers::pi * 50.0 * t);

        original_data[i] = {val, 0.0};
    }

    // Copy data to work in-place
    std::vector<std::complex<double>> processed_data = original_data;

    // Apply FFT
    fft.forward(processed_data);
    
    // Apply IFFT
    fft.inverse(processed_data);

    // Verify that the reconstructed signal
    // matches the original one
    compareVectors(processed_data, original_data);
}

// 5. DC component test
// A constant signal [2, 2, 2, 2] should produce
// a peak at frequency 0 equal to N * value -> [8, 0, 0, 0]
TEST_F(RecursiveFFTTest, ConstantSignal) {
    std::vector<std::complex<double>> time_data = {
        {2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}
    };
    
    std::vector<std::complex<double>> expected_freq_data = {
        {8.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };

    fft.forward(time_data);

    compareVectors(time_data, expected_freq_data);
}

} // namespace testing
} // namespace stft