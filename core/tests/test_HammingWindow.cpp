/**
 * @file test_HammingWindow.cpp
 * @brief Tests the Hamming window's formula and its documented DSP properties.
 *
 * Value semantics, operator() delegation and other behaviour inherited
 * unchanged from BaseWindow are tested once, in test_BaseWindow.cpp, rather
 * than re-verified per concrete window.
 */

#include <gtest/gtest.h>
#include "window/HammingWindow.hpp"
#include <cmath>
#include <numeric>
#include <numbers>
#include <vector>

using namespace stft;

static_assert(WindowFunction<HammingWindow>);

class HammingWindowTest : public ::testing::Test {
protected:
    static constexpr double TOL = 1e-10;
    static constexpr double COEFF_TOL = 1e-8;
    static constexpr std::size_t SMALL_SIZE = 16;
    static constexpr std::size_t MEDIUM_SIZE = 256;

    static double hammingCoefficient(std::size_t n, std::size_t N) {
        return 0.54 - 0.46 * std::cos(2.0 * std::numbers::pi * n / (N - 1));
    }
};

TEST_F(HammingWindowTest, ConstructionAndSize) {
    HammingWindow ham(MEDIUM_SIZE);
    EXPECT_EQ(ham.size(), MEDIUM_SIZE);
}

TEST_F(HammingWindowTest, FormulaCorrectness) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();

    for (std::size_t n = 0; n < MEDIUM_SIZE; ++n) {
        EXPECT_NEAR(coeffs[n], hammingCoefficient(n, MEDIUM_SIZE), COEFF_TOL)
            << "Coefficient at index " << n << " does not match formula";
    }
}

// Minimum 0.54 - 0.46 = 0.08 at the endpoints, maximum 0.54 + 0.46 = 1.0
// at the center — unlike Hann/Blackman, Hamming never reaches zero.
TEST_F(HammingWindowTest, CoefficientRange) {
    HammingWindow ham(MEDIUM_SIZE);
    for (double c : ham.coefficients()) {
        EXPECT_GE(c, 0.08 - COEFF_TOL);
        EXPECT_LE(c, 1.0 + COEFF_TOL);
    }
}

TEST_F(HammingWindowTest, NonZeroEndpoints) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();

    EXPECT_NEAR(coeffs.front(), 0.08, COEFF_TOL);
    EXPECT_NEAR(coeffs.back(), 0.08, COEFF_TOL);
}

TEST_F(HammingWindowTest, Symmetry) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();
    for (std::size_t i = 0; i < MEDIUM_SIZE / 2; ++i) {
        EXPECT_NEAR(coeffs[i], coeffs[MEDIUM_SIZE - 1 - i], COEFF_TOL);
    }
}

TEST_F(HammingWindowTest, PeakAtCenter) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();
    const double centerValue = coeffs[MEDIUM_SIZE / 2];

    EXPECT_NEAR(centerValue, 1.0, 1e-4);
    for (double c : coeffs) {
        EXPECT_LE(c, centerValue + 1e-8);
    }
}

TEST_F(HammingWindowTest, CoherentGainTheoretical) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();
    const double manualCg = std::accumulate(coeffs.begin(), coeffs.end(), 0.0)
                           / MEDIUM_SIZE;

    EXPECT_NEAR(ham.coherentGain(), 0.54, 0.01);
    EXPECT_NEAR(ham.coherentGain(), manualCg, TOL);
}

// Documented theoretical ENBW ~1.36 bins: narrower than Hann (1.5) and
// Blackman (1.73), consistent with Hamming's better frequency resolution.
TEST_F(HammingWindowTest, PowerBandwidth) {
    EXPECT_NEAR(HammingWindow(MEDIUM_SIZE).powerBandwidth(), 1.36, 0.1);
}

TEST_F(HammingWindowTest, ApplyWindowInPlace) {
    HammingWindow ham(SMALL_SIZE);
    std::vector<double> signal(SMALL_SIZE, 1.0);
    ham.apply(signal);

    const auto& coeffs = ham.coefficients();
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal[i], coeffs[i], TOL);
    }
}

TEST_F(HammingWindowTest, MinimumSize) {
    HammingWindow ham(2);
    const auto& coeffs = ham.coefficients();
    EXPECT_NEAR(coeffs[0], 0.08, COEFF_TOL);
    EXPECT_NEAR(coeffs[1], 0.08, COEFF_TOL);
}

// The formula, re-checked at several sizes: an off-by-one in the (N-1)
// denominator could easily pass at one size and fail at another.
TEST_F(HammingWindowTest, MultiSizeFormulaVerification) {
    for (std::size_t size : {8, 16, 32, 64, 128, 256, 4096}) {
        HammingWindow ham(size);
        const auto& coeffs = ham.coefficients();
        for (std::size_t n = 0; n < size; ++n) {
            EXPECT_NEAR(coeffs[n], hammingCoefficient(n, size), COEFF_TOL)
                << "size=" << size << " n=" << n;
        }
    }
}
