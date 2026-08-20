/**
 * @file test_HannWindow.cpp
 * @brief Tests the Hann window's formula and its documented DSP properties.
 *
 * Value semantics, operator() delegation and other behaviour inherited
 * unchanged from BaseWindow are tested once, in test_BaseWindow.cpp, rather
 * than re-verified per concrete window: HannWindow does not override any of
 * it, so repeating those tests here would exercise the exact same code path
 * without adding coverage.
 */

#include <gtest/gtest.h>
#include "window/HannWindow.hpp"
#include <cmath>
#include <numeric>
#include <numbers>
#include <vector>

using namespace stft;

static_assert(WindowFunction<HannWindow>);

class HannWindowTest : public ::testing::Test {
protected:
    static constexpr double TOL = 1e-10;
    static constexpr double COEFF_TOL = 1e-8;
    static constexpr std::size_t SMALL_SIZE = 16;
    static constexpr std::size_t MEDIUM_SIZE = 256;

    static double hannCoefficient(std::size_t n, std::size_t N) {
        return 0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * n / (N - 1)));
    }
};

TEST_F(HannWindowTest, ConstructionAndSize) {
    HannWindow hann(MEDIUM_SIZE);
    EXPECT_EQ(hann.size(), MEDIUM_SIZE);
}

TEST_F(HannWindowTest, FormulaCorrectness) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();

    for (std::size_t n = 0; n < MEDIUM_SIZE; ++n) {
        EXPECT_NEAR(coeffs[n], hannCoefficient(n, MEDIUM_SIZE), COEFF_TOL)
            << "Coefficient at index " << n << " does not match formula";
    }
}

TEST_F(HannWindowTest, CoefficientRange) {
    HannWindow hann(MEDIUM_SIZE);
    for (double c : hann.coefficients()) {
        EXPECT_GE(c, 0.0);
        EXPECT_LE(c, 1.0);
    }
}

// w[0] = w[N-1] = 0.5*(1 - cos(0)) = 0: Hann tapers fully to zero, unlike
// Hamming (see test_HammingWindow.cpp for the contrasting case).
TEST_F(HannWindowTest, ZeroEndpoints) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();

    EXPECT_NEAR(coeffs.front(), 0.0, COEFF_TOL);
    EXPECT_NEAR(coeffs.back(), 0.0, COEFF_TOL);
}

TEST_F(HannWindowTest, Symmetry) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();
    for (std::size_t i = 0; i < MEDIUM_SIZE / 2; ++i) {
        EXPECT_NEAR(coeffs[i], coeffs[MEDIUM_SIZE - 1 - i], COEFF_TOL);
    }
}

TEST_F(HannWindowTest, PeakAtCenter) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();
    const double centerValue = coeffs[MEDIUM_SIZE / 2];

    EXPECT_NEAR(centerValue, 1.0, 1e-4);
    for (double c : coeffs) {
        EXPECT_LE(c, centerValue + 1e-8);
    }
}

TEST_F(HannWindowTest, CoherentGainTheoretical) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();
    const double manualCg = std::accumulate(coeffs.begin(), coeffs.end(), 0.0)
                           / MEDIUM_SIZE;

    EXPECT_NEAR(hann.coherentGain(), 0.5, 0.01);
    EXPECT_NEAR(hann.coherentGain(), manualCg, TOL);
}

// Documented theoretical ENBW ~1.5 bins (between Hamming's 1.36 and
// Blackman's 1.73).
TEST_F(HannWindowTest, PowerBandwidth) {
    EXPECT_NEAR(HannWindow(MEDIUM_SIZE).powerBandwidth(), 1.5, 0.1);
}

TEST_F(HannWindowTest, ApplyWindowInPlace) {
    HannWindow hann(SMALL_SIZE);
    std::vector<double> signal(SMALL_SIZE, 1.0);
    hann.apply(signal);

    const auto& coeffs = hann.coefficients();
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal[i], coeffs[i], TOL);
    }
}

TEST_F(HannWindowTest, MinimumSize) {
    HannWindow hann(2);
    const auto& coeffs = hann.coefficients();
    EXPECT_NEAR(coeffs[0], 0.0, COEFF_TOL);
    EXPECT_NEAR(coeffs[1], 0.0, COEFF_TOL);
}

// The formula, re-checked at several sizes: an off-by-one in the (N-1)
// denominator could easily pass at one size and fail at another.
TEST_F(HannWindowTest, MultiSizeFormulaVerification) {
    for (std::size_t size : {8, 16, 32, 64, 128, 256, 4096}) {
        HannWindow hann(size);
        const auto& coeffs = hann.coefficients();
        for (std::size_t n = 0; n < size; ++n) {
            EXPECT_NEAR(coeffs[n], hannCoefficient(n, size), COEFF_TOL)
                << "size=" << size << " n=" << n;
        }
    }
}
