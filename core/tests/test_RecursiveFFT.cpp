#include <gtest/gtest.h>
#include "fft/RecursiveFFT.hpp"
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

using namespace stft;

/**
 * @brief Fixture: one RecursiveFFT engine plus a tolerant complex-vector
 *        comparison, shared by every test below.
 */
class RecursiveFFTTest : public ::testing::Test {
protected:
    RecursiveFFT fft;
    static constexpr double tolerance = 1e-9;

    static void compareVectors(const std::vector<std::complex<double>>& v1,
                               const std::vector<std::complex<double>>& v2) {
        ASSERT_EQ(v1.size(), v2.size()) << "Vectors have different sizes.";
        for (std::size_t i = 0; i < v1.size(); ++i) {
            EXPECT_NEAR(v1[i].real(), v2[i].real(), tolerance)
                << "Real part mismatch at index " << i;
            EXPECT_NEAR(v1[i].imag(), v2[i].imag(), tolerance)
                << "Imaginary part mismatch at index " << i;
        }
    }
};

TEST_F(RecursiveFFTTest, HandlesEmptyInput) {
    std::vector<std::complex<double>> data;
    EXPECT_THROW(fft.forward(data), std::invalid_argument);
    EXPECT_TRUE(data.empty());
}

TEST_F(RecursiveFFTTest, ThrowsOnNonPowerOfTwo) {
    std::vector<std::complex<double>> data(3, {1.0, 0.0});
    EXPECT_THROW(fft.forward(data), std::invalid_argument);
    EXPECT_THROW(fft.inverse(data), std::invalid_argument);
}

// A time-domain impulse [1, 0, 0, 0] is flat in frequency: [1, 1, 1, 1].
TEST_F(RecursiveFFTTest, ImpulseResponse) {
    std::vector<std::complex<double>> data = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };
    const std::vector<std::complex<double>> expected = {
        {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}
    };

    fft.forward(data);
    compareVectors(data, expected);
}

// inverse(forward(x)) must reconstruct x, up to floating-point error.
TEST_F(RecursiveFFTTest, ForwardInverseIdentity) {
    constexpr std::size_t N = 1024;
    std::vector<std::complex<double>> original(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / N;
        const double val = std::sin(2.0 * std::numbers::pi * 10.0 * t)
                          + 0.5 * std::cos(2.0 * std::numbers::pi * 50.0 * t);
        original[i] = {val, 0.0};
    }

    std::vector<std::complex<double>> roundTrip = original;
    fft.forward(roundTrip);
    fft.inverse(roundTrip);

    compareVectors(roundTrip, original);
}

// A constant signal [2, 2, 2, 2] has all its energy at DC: N * value = 8.
TEST_F(RecursiveFFTTest, ConstantSignal) {
    std::vector<std::complex<double>> data(4, {2.0, 0.0});
    const std::vector<std::complex<double>> expected = {
        {8.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };

    fft.forward(data);
    compareVectors(data, expected);
}
