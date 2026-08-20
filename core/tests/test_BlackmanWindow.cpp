/**
 * @file test_BlackmanWindow.cpp
 * @brief Tests the Blackman window's formula and its documented DSP properties.
 *
 * Value semantics, operator() delegation and other behaviour inherited
 * unchanged from BaseWindow are tested once, in test_BaseWindow.cpp, rather
 * than re-verified per concrete window.
 */

#include <gtest/gtest.h>
#include "window/BlackmanWindow.hpp"
#include <cmath>
#include <numeric>
#include <numbers>
#include <vector>

using namespace stft;

static_assert(WindowFunction<BlackmanWindow>);

class BlackmanWindowTest : public ::testing::Test {
protected:
    static constexpr double TOL = 1e-10;
    static constexpr double COEFF_TOL = 1e-8;
    static constexpr std::size_t SMALL_SIZE = 16;
    static constexpr std::size_t MEDIUM_SIZE = 256;

    static double blackmanCoefficient(std::size_t n, std::size_t N) {
        const double angle = 2.0 * std::numbers::pi * n / (N - 1);
        return 0.42 - 0.50 * std::cos(angle) + 0.08 * std::cos(2.0 * angle);
    }
};

TEST_F(BlackmanWindowTest, ConstructionAndSize) {
    BlackmanWindow bw(MEDIUM_SIZE);
    EXPECT_EQ(bw.size(), MEDIUM_SIZE);
}

TEST_F(BlackmanWindowTest, FormulaCorrectness) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();

    for (std::size_t n = 0; n < MEDIUM_SIZE; ++n) {
        EXPECT_NEAR(coeffs[n], blackmanCoefficient(n, MEDIUM_SIZE), COEFF_TOL)
            << "Coefficient at index " << n << " does not match formula";
    }
}

TEST_F(BlackmanWindowTest, CoefficientRange) {
    BlackmanWindow bw(MEDIUM_SIZE);
    for (double c : bw.coefficients()) {
        EXPECT_GE(c, 0.0);
        EXPECT_LE(c, 1.0);
    }
}

// By construction, a0 - a1 + a2 = 0.42 - 0.50 + 0.08 = 0: Blackman tapers
// fully to zero, same as Hann but via a three-term formula.
TEST_F(BlackmanWindowTest, EndpointZeros) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();

    EXPECT_NEAR(coeffs.front(), 0.0, COEFF_TOL);
    EXPECT_NEAR(coeffs.back(), 0.0, COEFF_TOL);
}

TEST_F(BlackmanWindowTest, Symmetry) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();
    for (std::size_t i = 0; i < MEDIUM_SIZE / 2; ++i) {
        EXPECT_NEAR(coeffs[i], coeffs[MEDIUM_SIZE - 1 - i], COEFF_TOL);
    }
}

TEST_F(BlackmanWindowTest, PeakAtCenter) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();
    const double centerValue = coeffs[MEDIUM_SIZE / 2];

    for (double c : coeffs) {
        EXPECT_LE(c, centerValue + COEFF_TOL);
    }
}

TEST_F(BlackmanWindowTest, CoherentGainTheoretical) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();
    const double manualCg = std::accumulate(coeffs.begin(), coeffs.end(), 0.0)
                           / MEDIUM_SIZE;

    EXPECT_NEAR(bw.coherentGain(), 0.42, 0.01);
    EXPECT_NEAR(bw.coherentGain(), manualCg, TOL);
}

// Documented theoretical ENBW ~1.73 bins: the widest of the three, the
// price paid for the lowest sidelobes.
TEST_F(BlackmanWindowTest, PowerBandwidth) {
    EXPECT_NEAR(BlackmanWindow(MEDIUM_SIZE).powerBandwidth(), 1.73, 0.1);
}

TEST_F(BlackmanWindowTest, ApplyWindowInPlace) {
    BlackmanWindow bw(SMALL_SIZE);
    std::vector<double> signal(SMALL_SIZE, 1.0);
    bw.apply(signal);

    const auto& coeffs = bw.coefficients();
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal[i], coeffs[i], TOL);
    }
}

TEST_F(BlackmanWindowTest, MinimumSize) {
    BlackmanWindow bw(2);
    const auto& coeffs = bw.coefficients();
    EXPECT_NEAR(coeffs[0], 0.0, COEFF_TOL);
    EXPECT_NEAR(coeffs[1], 0.0, COEFF_TOL);
}

// The formula, re-checked at several sizes: an off-by-one in the (N-1)
// denominator could easily pass at one size and fail at another.
TEST_F(BlackmanWindowTest, MultiSizeFormulaVerification) {
    for (std::size_t size : {8, 16, 32, 64, 128, 256, 4096}) {
        BlackmanWindow bw(size);
        const auto& coeffs = bw.coefficients();
        for (std::size_t n = 0; n < size; ++n) {
            EXPECT_NEAR(coeffs[n], blackmanCoefficient(n, size), COEFF_TOL)
                << "size=" << size << " n=" << n;
        }
    }
}
