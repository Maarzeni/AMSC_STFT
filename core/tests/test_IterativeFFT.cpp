#include <gtest/gtest.h>
#include "fft/IterativeFFT.hpp"
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

using namespace stft;

/**
 * @brief Fixture: one IterativeFFT engine plus a tolerant complex-vector
 *        comparison, shared by every test below.
 *
 * Same test battery as RecursiveFFTTest, deliberately: the two engines
 * implement the same transform through different algorithms, and this
 * checks that both agree on it, not just that each is internally
 * consistent.
 */
class IterativeFFTTest : public ::testing::Test {
protected:
    IterativeFFT fft;
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

TEST_F(IterativeFFTTest, HandlesEmptyInput) {
    std::vector<std::complex<double>> data;
    EXPECT_THROW(fft.forward(data), std::invalid_argument);
    EXPECT_TRUE(data.empty());
}

TEST_F(IterativeFFTTest, ThrowsOnNonPowerOfTwo) {
    std::vector<std::complex<double>> data(6, {1.0, 0.0});
    EXPECT_THROW(fft.forward(data), std::invalid_argument);
    EXPECT_THROW(fft.inverse(data), std::invalid_argument);
}

// A time-domain impulse [1, 0, 0, 0] is flat in frequency: [1, 1, 1, 1].
TEST_F(IterativeFFTTest, ImpulseResponse) {
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
TEST_F(IterativeFFTTest, ForwardInverseIdentity) {
    constexpr std::size_t N = 512;
    std::vector<std::complex<double>> original(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / N;
        const double val = std::cos(2.0 * std::numbers::pi * 5.0 * t)
                          + 0.3 * std::sin(2.0 * std::numbers::pi * 12.0 * t);
        original[i] = {val, 0.0};
    }

    std::vector<std::complex<double>> roundTrip = original;
    fft.forward(roundTrip);
    fft.inverse(roundTrip);

    compareVectors(roundTrip, original);
}

// A constant signal has all its energy at DC: N * value.
TEST_F(IterativeFFTTest, ConstantSignal) {
    constexpr std::size_t N = 8;
    std::vector<std::complex<double>> data(N, {3.0, 0.0});
    std::vector<std::complex<double>> expected(N, {0.0, 0.0});
    expected[0] = {3.0 * N, 0.0};

    fft.forward(data);
    compareVectors(data, expected);
}
