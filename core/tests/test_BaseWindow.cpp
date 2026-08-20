/**
 * @file test_BaseWindow.cpp
 * @brief Tests BaseWindow's own machinery (storage, CRTP dispatch, DSP
 *        utilities, error handling, value semantics), independently of any
 *        real window's formula.
 *
 * Two minimal concrete windows exercise it: RectangularWindow (all
 * coefficients 1.0, the simplest possible generate()) and TriangularWindow
 * (a non-trivial, non-constant one), so the checks below are not tied to
 * Hann/Hamming/Blackman specifically.
 */

#include <gtest/gtest.h>
#include "window/BaseWindow.hpp"
#include <cmath>
#include <numeric>
#include <vector>

using namespace stft;

class RectangularWindow : public BaseWindow<RectangularWindow> {
    friend class BaseWindow<RectangularWindow>;

private:
    std::vector<double> generate(std::size_t N) const {
        return std::vector<double>(N, 1.0);
    }

public:
    explicit RectangularWindow(std::size_t N) : BaseWindow<RectangularWindow>(N) {}
};

/// w[n] = 1 - |2(n - (N-1)/2) / N|
class TriangularWindow : public BaseWindow<TriangularWindow> {
    friend class BaseWindow<TriangularWindow>;

private:
    std::vector<double> generate(std::size_t N) const {
        std::vector<double> coeffs(N);
        const double center = (N - 1.0) / 2.0;
        for (std::size_t i = 0; i < N; ++i) {
            coeffs[i] = 1.0 - std::abs(2.0 * (static_cast<double>(i) - center) / N);
        }
        return coeffs;
    }

public:
    explicit TriangularWindow(std::size_t N) : BaseWindow<TriangularWindow>(N) {}
};

ASSERT_WINDOW_FUNCTION(RectangularWindow);
ASSERT_WINDOW_FUNCTION(TriangularWindow);

class BaseWindowTest : public ::testing::Test {
protected:
    static constexpr double TOL = 1e-10;
    static constexpr std::size_t SMALL_SIZE = 8;
    static constexpr std::size_t MEDIUM_SIZE = 256;
};

TEST_F(BaseWindowTest, ConstructionAndSize) {
    RectangularWindow rect(SMALL_SIZE);
    EXPECT_EQ(rect.size(), SMALL_SIZE);

    TriangularWindow tri(MEDIUM_SIZE);
    EXPECT_EQ(tri.size(), MEDIUM_SIZE);
}

TEST_F(BaseWindowTest, CoefficientsAccess) {
    RectangularWindow rect(SMALL_SIZE);
    const auto& coeffs = rect.coefficients();

    EXPECT_EQ(coeffs.size(), SMALL_SIZE);
    for (double coeff : coeffs) {
        EXPECT_NEAR(coeff, 1.0, TOL);
    }
}

TEST_F(BaseWindowTest, InvalidSizeThrows) {
    EXPECT_THROW(RectangularWindow(0), std::invalid_argument);
    EXPECT_THROW(RectangularWindow(1), std::invalid_argument);
    EXPECT_NO_THROW(RectangularWindow(2));  // minimum valid size
}

// CG = 1.0 for rectangular (all coefficients 1), 0.5 for triangular (a
// triangle's mean is half its peak) — known closed-form values, plus a
// direct recomputation from the coefficients as a cross-check.
TEST_F(BaseWindowTest, CoherentGain) {
    RectangularWindow rect(100);
    EXPECT_NEAR(rect.coherentGain(), 1.0, TOL);

    TriangularWindow tri(100);
    EXPECT_NEAR(tri.coherentGain(), 0.5, TOL);

    const auto& triCoeffs = tri.coefficients();
    const double manualCg = std::accumulate(triCoeffs.begin(), triCoeffs.end(), 0.0)
                           / triCoeffs.size();
    EXPECT_NEAR(tri.coherentGain(), manualCg, TOL);
}

// ENBW = N * sum(w^2) / (sum w)^2. Rectangular: sum(w) = sum(w^2) = N, so
// ENBW = 1.0 exactly. Triangular: known closed-form value 4/3.
TEST_F(BaseWindowTest, PowerBandwidth) {
    RectangularWindow rect(100);
    EXPECT_NEAR(rect.powerBandwidth(), 1.0, TOL);

    TriangularWindow tri(100);
    EXPECT_NEAR(tri.powerBandwidth(), 4.0 / 3.0, 0.01);
}

TEST_F(BaseWindowTest, ApplyWindowInPlace) {
    RectangularWindow rect(SMALL_SIZE);
    std::vector<double> signal(SMALL_SIZE, 2.0);
    rect.apply(signal);

    // Rectangular coefficients are 1.0, so the signal is unchanged.
    for (double val : signal) {
        EXPECT_NEAR(val, 2.0, TOL);
    }
}

TEST_F(BaseWindowTest, ApplyWindowTriangular) {
    TriangularWindow tri(SMALL_SIZE);
    const std::vector<double> expected = tri.coefficients();

    std::vector<double> signal(SMALL_SIZE, 1.0);
    tri.apply(signal);

    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal[i], expected[i], TOL);
    }
}

TEST_F(BaseWindowTest, ApplyRejectsWrongSize) {
    RectangularWindow rect(SMALL_SIZE);
    std::vector<double> signal(SMALL_SIZE + 1, 1.0);
    EXPECT_THROW(rect.apply(signal), std::invalid_argument);
}

TEST_F(BaseWindowTest, CopySemantics) {
    RectangularWindow rect1(SMALL_SIZE);
    RectangularWindow rect2 = rect1;

    EXPECT_EQ(rect1.size(), rect2.size());
    EXPECT_EQ(rect1.coefficients(), rect2.coefficients());
}

TEST_F(BaseWindowTest, MoveSemantics) {
    RectangularWindow rect1(MEDIUM_SIZE);
    const std::size_t originalSize = rect1.size();

    RectangularWindow rect2 = std::move(rect1);

    EXPECT_EQ(rect2.size(), originalSize);
    EXPECT_EQ(rect2.coefficients().size(), originalSize);
}

// Two different CRTP instantiations (BaseWindow<Rectangular> vs
// BaseWindow<Triangular>) must not share any state.
TEST_F(BaseWindowTest, MultipleWindowInstances) {
    RectangularWindow rect(SMALL_SIZE);
    TriangularWindow tri(SMALL_SIZE);

    std::vector<double> signalRect(SMALL_SIZE, 2.0);
    std::vector<double> signalTri(SMALL_SIZE, 2.0);
    rect.apply(signalRect);
    tri.apply(signalTri);

    const auto& triCoeffs = tri.coefficients();
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signalRect[i], 2.0, TOL);                  // *1.0: unchanged
        EXPECT_NEAR(signalTri[i], 2.0 * triCoeffs[i], TOL);    // *coefficient
    }
}

// apply() must modify only the signal it is given, never coeffs_.
TEST_F(BaseWindowTest, CoefficientImmutability) {
    RectangularWindow rect(SMALL_SIZE);
    const std::vector<double> before = rect.coefficients();

    std::vector<double> signal(SMALL_SIZE, 1.0);
    rect.apply(signal);

    EXPECT_EQ(rect.coefficients(), before);
}

TEST_F(BaseWindowTest, MinimumSize) {
    RectangularWindow rect(2);
    EXPECT_EQ(rect.size(), 2u);
    EXPECT_NEAR(rect.coherentGain(), 1.0, TOL);

    std::vector<double> signal{1.0, 2.0};
    rect.apply(signal);
    EXPECT_NEAR(signal[0], 1.0, TOL);
    EXPECT_NEAR(signal[1], 2.0, TOL);
}

// The exact return types the WindowFunction concept requires (size_t,
// const vector<double>&, double, void) are already enforced by the
// ASSERT_WINDOW_FUNCTION checks above; nothing else to verify here.

TEST_F(BaseWindowTest, OperatorCallMatchesApply) {
    RectangularWindow w(MEDIUM_SIZE);

    std::vector<double> signalApply(MEDIUM_SIZE, 2.0);
    std::vector<double> signalCall(MEDIUM_SIZE, 2.0);
    w.apply(signalApply);
    w(signalCall);

    EXPECT_EQ(signalApply, signalCall);
}

TEST_F(BaseWindowTest, OperatorCallThrowsOnSizeMismatch) {
    RectangularWindow w(MEDIUM_SIZE);
    std::vector<double> wrongSize(MEDIUM_SIZE + 1, 1.0);
    EXPECT_THROW(w(wrongSize), std::invalid_argument);
}

// operator() must work identically through every CRTP instantiation, not
// just the one exercised above.
TEST_F(BaseWindowTest, OperatorCallWorksForDifferentWindowTypes) {
    TriangularWindow w(MEDIUM_SIZE);

    std::vector<double> signalApply(MEDIUM_SIZE, 3.0);
    std::vector<double> signalCall(MEDIUM_SIZE, 3.0);
    w.apply(signalApply);
    w(signalCall);

    EXPECT_EQ(signalApply, signalCall);
}
