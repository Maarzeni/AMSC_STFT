/**
 * @file test_BlackmanWindow.cpp
 * @brief Test suite for Blackman window function implementation.
 *
 * Tests the mathematical correctness, properties, and behavior of the
 * Blackman window (three-term cosine window with excellent sidelobe rejection).
 */

#include <gtest/gtest.h>
#include "window/BlackmanWindow.hpp"
#include "window/WindowConcept.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <numbers>

using namespace stft;

// ==========================================
// COMPILE-TIME CONCEPT VALIDATION
// ==========================================

/**
 * @brief Static assertion: BlackmanWindow must satisfy WindowFunction concept.
 *
 * This runs at compile time. If BlackmanWindow doesn't have the required
 * methods or signatures, compilation fails immediately with a clear error.
 */
static_assert(WindowFunction<BlackmanWindow>,
              "ERROR: BlackmanWindow must satisfy WindowFunction concept!");

// ==========================================
// TEST FIXTURE
// ==========================================

class BlackmanWindowTest : public ::testing::Test {
protected:
    const double TOL = 1e-10;
    const double COEFF_TOL = 1e-8;  // Tighter tolerance for formula verification
    const std::size_t SMALL_SIZE = 16;
    const std::size_t MEDIUM_SIZE = 256;
    const std::size_t LARGE_SIZE = 4096;

    /**
     * @brief Manually computes Blackman coefficients for verification.
     *
     * Used to verify the generate() method produces correct values.
     */
    static double blackmanCoefficient(std::size_t n, std::size_t N) {
        double angle = 2.0 * std::numbers::pi * n / (N - 1);
        return 0.42
               - 0.50 * std::cos(angle)
               + 0.08 * std::cos(2.0 * angle);
    }
};

// ==========================================
// TEST 1: Construction and Size
// ==========================================

/**
 * @brief Verifies that BlackmanWindow constructs correctly with various sizes.
 */
TEST_F(BlackmanWindowTest, ConstructionAndSize) {
    BlackmanWindow bw_small(SMALL_SIZE);
    EXPECT_EQ(bw_small.size(), SMALL_SIZE);

    BlackmanWindow bw_medium(MEDIUM_SIZE);
    EXPECT_EQ(bw_medium.size(), MEDIUM_SIZE);

    BlackmanWindow bw_large(LARGE_SIZE);
    EXPECT_EQ(bw_large.size(), LARGE_SIZE);
}

// ==========================================
// TEST 2: Mathematical Formula Correctness
// ==========================================

/**
 * @brief Verifies that generated coefficients match the mathematical formula.
 *
 * Formula: w[n] = 0.42 - 0.50*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))
 */
TEST_F(BlackmanWindowTest, FormulaCorrectness) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();

    for (std::size_t n = 0; n < MEDIUM_SIZE; ++n) {
        double expected = blackmanCoefficient(n, MEDIUM_SIZE);
        EXPECT_NEAR(coeffs[n], expected, COEFF_TOL)
            << "Coefficient at index " << n << " does not match formula";
    }
}

// ==========================================
// TEST 3: Coefficient Range [0, 1]
// ==========================================

/**
 * @brief Verifies that all coefficients are in the valid range [0, 1].
 *
 * Window functions must be non-negative and typically normalized to [0, 1].
 * The clamping in generate() ensures this is guaranteed, not just approximate.
 */
TEST_F(BlackmanWindowTest, CoefficientRange) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();

    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i) {
        EXPECT_GE(coeffs[i], 0.0)
            << "Coefficient at index " << i << " is negative";
        EXPECT_LE(coeffs[i], 1.0)
            << "Coefficient at index " << i << " exceeds 1.0";
    }
}

// ==========================================
// TEST 4: Endpoint Zeros
// ==========================================

/**
 * @brief Verifies that Blackman window tapers to (nearly) zero at endpoints.
 *
 * By design, the Blackman coefficients satisfy: 0.42 - 0.50 + 0.08 = 0
 * So at n=0 and n=N-1, the window should be approximately zero.
 */
TEST_F(BlackmanWindowTest, EndpointZeros) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();

    // At n=0: angle = 0, cos(0) = 1
    // w[0] = 0.42 - 0.50*1 + 0.08*1 = 0.42 - 0.50 + 0.08 = 0
    EXPECT_NEAR(coeffs[0], 0.0, COEFF_TOL)
        << "Blackman window should start at zero";

    // At n=N-1: angle = 2π, cos(2π) = 1
    // w[N-1] = 0.42 - 0.50*1 + 0.08*1 = 0
    EXPECT_NEAR(coeffs[MEDIUM_SIZE - 1], 0.0, COEFF_TOL)
        << "Blackman window should end at zero";
}

// ==========================================
// TEST 5: Symmetry
// ==========================================

/**
 * @brief Verifies that Blackman window is symmetric around the center.
 *
 * Window functions are typically symmetric: w[n] ≈ w[N-1-n]
 */
TEST_F(BlackmanWindowTest, Symmetry) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();

    for (std::size_t i = 0; i < MEDIUM_SIZE / 2; ++i) {
        EXPECT_NEAR(coeffs[i], coeffs[MEDIUM_SIZE - 1 - i], COEFF_TOL)
            << "Window is not symmetric at index " << i;
    }
}

// ==========================================
// TEST 6: Peak at Center
// ==========================================

/**
 * @brief Verifies that the window has its maximum at the center.
 *
 * All well-behaved window functions peak in the middle.
 */
TEST_F(BlackmanWindowTest, PeakAtCenter) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();

    std::size_t center = MEDIUM_SIZE / 2;
    double center_value = coeffs[center];

    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i) {
        EXPECT_LE(coeffs[i], center_value + COEFF_TOL)
            << "Peak should be at center, but found larger value at index " << i;
    }
}

// ==========================================
// TEST 7: Coherent Gain Theoretical Value
// ==========================================

/**
 * @brief Verifies that coherent gain matches the theoretical value.
 *
 * For Blackman window, CG = 0.42 (the a0 coefficient).
 * This is because the symmetric placement of the cosines means the average
 * approaches the DC component 0.42.
 */
TEST_F(BlackmanWindowTest, CoherentGainTheoretical) {
    BlackmanWindow bw(MEDIUM_SIZE);
    double cg = bw.coherentGain();

    // Theoretical value: 0.42
    EXPECT_NEAR(cg, 0.42, 0.01)
        << "Coherent gain should be approximately 0.42";

    // Verify by manual calculation
    const auto& coeffs = bw.coefficients();
    double sum = std::accumulate(coeffs.begin(), coeffs.end(), 0.0);
    double manual_cg = sum / MEDIUM_SIZE;
    EXPECT_NEAR(cg, manual_cg, TOL)
        << "Coherent gain calculation mismatch";
}

// ==========================================
// TEST 8: Power Bandwidth (ENBW)
// ==========================================

/**
 * @brief Verifies that power bandwidth is within expected range.
 *
 * Blackman window has theoretical ENBW ≈ 1.73 bins.
 * This is wider than Hann (1.50) or Hamming (1.30) due to the smoother taper.
 */
TEST_F(BlackmanWindowTest, PowerBandwidth) {
    BlackmanWindow bw(MEDIUM_SIZE);
    double pb = bw.powerBandwidth();

    // Theoretical range: ~1.65 to ~1.81
    EXPECT_GT(pb, 1.6) << "ENBW seems too low";
    EXPECT_LT(pb, 1.9) << "ENBW seems too high";

    // More precise theoretical value
    EXPECT_NEAR(pb, 1.73, 0.1)
        << "ENBW should be approximately 1.73";
}

// ==========================================
// TEST 9: Apply Window In-Place
// ==========================================

/**
 * @brief Tests that apply() correctly multiplies signal by window coefficients.
 */
TEST_F(BlackmanWindowTest, ApplyWindowInPlace) {
    BlackmanWindow bw(SMALL_SIZE);

    // Create signal with all ones
    std::vector<double> signal(SMALL_SIZE, 1.0);
    std::vector<double> original = signal;

    bw.apply(signal);

    // Signal should now equal window coefficients
    const auto& coeffs = bw.coefficients();
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal[i], coeffs[i], TOL)
            << "apply() did not correctly multiply at index " << i;
    }
}

// ==========================================
// TEST 10: Apply to Non-Trivial Signal
// ==========================================

/**
 * @brief Tests apply() with a signal having varying amplitudes.
 */
TEST_F(BlackmanWindowTest, ApplyToVaryingSignal) {
    BlackmanWindow bw(SMALL_SIZE);

    // Create signal: [0, 1, 2, 3, ..., 15]
    std::vector<double> signal(SMALL_SIZE);
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        signal[i] = static_cast<double>(i);
    }

    bw.apply(signal);

    // Verify: signal[i] = i * w[i]
    const auto& coeffs = bw.coefficients();
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        double expected = i * coeffs[i];
        EXPECT_NEAR(signal[i], expected, TOL)
            << "apply() mismatch at index " << i;
    }
}

// ==========================================
// TEST 11: Tapering Smoothness
// ==========================================

/**
 * @brief Verifies that the window tapers smoothly (no sharp discontinuities).
 *
 * All three-term windows should have smooth transitions due to the
 * overlapping cosine harmonics.
 */
TEST_F(BlackmanWindowTest, TaperingSmoothing) {
    BlackmanWindow bw(MEDIUM_SIZE);
    const auto& coeffs = bw.coefficients();

    // Check that differences between adjacent coefficients are small
    for (std::size_t i = 1; i < MEDIUM_SIZE; ++i) {
        double diff = std::abs(coeffs[i] - coeffs[i - 1]);
        // The difference should be smooth (no jumps)
        EXPECT_LT(diff, 0.1)
            << "Window has a sharp transition at index " << i;
    }
}

// ==========================================
// TEST 12: Scaling Property
// ==========================================

/**
 * @brief Verifies that scaling works correctly (scaling signal scales output).
 */
TEST_F(BlackmanWindowTest, ScalingProperty) {
    BlackmanWindow bw(SMALL_SIZE);

    std::vector<double> signal1(SMALL_SIZE, 2.0);
    std::vector<double> signal2(SMALL_SIZE, 4.0);

    bw.apply(signal1);
    bw.apply(signal2);

    // signal2 should be 2x signal1 (due to 2x input amplitude)
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal2[i], 2.0 * signal1[i], TOL)
            << "Scaling property failed at index " << i;
    }
}

// ==========================================
// TEST 13: Multiple Instances Independence
// ==========================================

/**
 * @brief Verifies that multiple Blackman windows don't interfere with each other.
 */
TEST_F(BlackmanWindowTest, MultipleInstancesIndependence) {
    BlackmanWindow bw1(SMALL_SIZE);
    BlackmanWindow bw2(MEDIUM_SIZE);

    EXPECT_EQ(bw1.size(), SMALL_SIZE);
    EXPECT_EQ(bw2.size(), MEDIUM_SIZE);

    EXPECT_NE(bw1.coefficients().size(), bw2.coefficients().size());

    // Apply to different signals should be independent
    std::vector<double> signal1(SMALL_SIZE, 1.0);
    std::vector<double> signal2(MEDIUM_SIZE, 1.0);

    bw1.apply(signal1);
    bw2.apply(signal2);

    // Verify both were applied correctly
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal1[i], bw1.coefficients()[i], TOL);
    }
    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i) {
        EXPECT_NEAR(signal2[i], bw2.coefficients()[i], TOL);
    }
}

// ==========================================
// TEST 14: Large Window Performance
// ==========================================

/**
 * @brief Tests that large windows can be created and used efficiently.
 */
TEST_F(BlackmanWindowTest, LargeWindowPerformance) {
    EXPECT_NO_THROW({
        BlackmanWindow bw(LARGE_SIZE);
        EXPECT_EQ(bw.size(), LARGE_SIZE);

        std::vector<double> signal(LARGE_SIZE, 1.0);
        bw.apply(signal);

        EXPECT_NEAR(bw.coherentGain(), 0.42, 0.01);
    });
}

// ==========================================
// TEST 15: Copy Semantics
// ==========================================

/**
 * @brief Tests that Blackman windows can be copied correctly.
 */
TEST_F(BlackmanWindowTest, CopySemantics) {
    BlackmanWindow bw1(SMALL_SIZE);
    BlackmanWindow bw2 = bw1; // Copy constructor

    EXPECT_EQ(bw1.size(), bw2.size());
    EXPECT_EQ(bw1.coefficients(), bw2.coefficients());

    // Modify via apply should work independently
    std::vector<double> signal1(SMALL_SIZE, 2.0);
    std::vector<double> signal2(SMALL_SIZE, 2.0);

    bw1.apply(signal1);
    bw2.apply(signal2);

    EXPECT_EQ(signal1, signal2);
}

// ==========================================
// TEST 16: Move Semantics
// ==========================================

/**
 * @brief Tests that windows can be moved efficiently.
 */
TEST_F(BlackmanWindowTest, MoveSemantics) {
    BlackmanWindow bw1(MEDIUM_SIZE);
    std::size_t original_size = bw1.size();

    BlackmanWindow bw2 = std::move(bw1);

    EXPECT_EQ(bw2.size(), original_size);
}

// ==========================================
// TEST 17: Edge Case - Minimum Size
// ==========================================

/**
 * @brief Tests the minimum valid window size (N=2).
 */
TEST_F(BlackmanWindowTest, MinimumSize) {
    BlackmanWindow bw(2);
    EXPECT_EQ(bw.size(), 2);

    const auto& coeffs = bw.coefficients();
    EXPECT_EQ(coeffs.size(), 2);

    // At n=0: w[0] = 0 (endpoint)
    EXPECT_NEAR(coeffs[0], 0.0, COEFF_TOL);

    // At n=1 (which is also N-1): w[1] = 0 (endpoint)
    EXPECT_NEAR(coeffs[1], 0.0, COEFF_TOL);
}

// ==========================================
// TEST 18: Comparison with Theoretical Formula
// ==========================================

/**
 * @brief Direct comparison of multiple window sizes against theory.
 */
TEST_F(BlackmanWindowTest, MultiSizeFormulasVerification) {
    std::size_t sizes[] = {8, 16, 32, 64, 128, 256};

    for (auto size : sizes) {
        BlackmanWindow bw(size);
        const auto& coeffs = bw.coefficients();

        for (std::size_t n = 0; n < size; ++n) {
            double expected = blackmanCoefficient(n, size);
            EXPECT_NEAR(coeffs[n], expected, COEFF_TOL)
                << "Formula mismatch at n=" << n << " for size=" << size;
        }
    }
}
