#include <gtest/gtest.h>
#include "fft/IterativeFFT.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <numbers>

namespace stft {
namespace testing {

// Fixture class for IterativeFFT tests
class IterativeFFTTest : public ::testing::Test {
protected:
    IterativeFFT fft;
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
TEST_F(IterativeFFTTest, HandlesEmptyInput) {
    std::vector<std::complex<double>> data;

    // BaseFFT checkPowerOfTwo throws on size 0
    EXPECT_THROW(fft.forward(data), std::invalid_argument);
    EXPECT_TRUE(data.empty());
}

// 2. Test with non-power-of-two size
TEST_F(IterativeFFTTest, ThrowsOnNonPowerOfTwo) {
    std::vector<std::complex<double>> data(6, {1.0, 0.0}); // N = 6
    
    EXPECT_THROW({
        fft.forward(data);
    }, std::invalid_argument);

    EXPECT_THROW({
        fft.inverse(data);
    }, std::invalid_argument);
}

// 3. Dirac impulse transform test
// An impulse [1, 0, 0, 0] -> [1, 1, 1, 1]
TEST_F(IterativeFFTTest, ImpulseResponse) {
    std::vector<std::complex<double>> time_data = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };
    
    std::vector<std::complex<double>> expected_freq_data = {
        {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}
    };

    fft.forward(time_data); 

    compareVectors(time_data, expected_freq_data);
}

// 4. Reversibility test: IFFT(FFT(x)) == x
TEST_F(IterativeFFTTest, ForwardInverseIdentity) {
    const size_t N = 512; 

    std::vector<std::complex<double>> original_data(N);
    
    for (size_t i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / N;
        // Signal with two components
        double val = std::cos(2.0 * std::numbers::pi * 5.0 * t) +
                     0.3 * std::sin(2.0 * std::numbers::pi * 12.0 * t);
        original_data[i] = {val, 0.0};
    }

    std::vector<std::complex<double>> processed_data = original_data;

    fft.forward(processed_data);
    fft.inverse(processed_data);

    compareVectors(processed_data, original_data);
}

// 5. DC component test
// A constant signal [3, 3, 3, 3, 3, 3, 3, 3] should produce
// a peak at frequency 0 equal to N * value -> [24, 0, 0, 0, 0, 0, 0, 0]
TEST_F(IterativeFFTTest, ConstantSignal) {
    const size_t N = 8;
    std::vector<std::complex<double>> time_data(N, {3.0, 0.0});
    
    std::vector<std::complex<double>> expected_freq_data(N, {0.0, 0.0});
    expected_freq_data[0] = {3.0 * N, 0.0};

    fft.forward(time_data);

    compareVectors(time_data, expected_freq_data);
}

} // namespace testing
} // namespace stft