/**
 * @file test_HammingWindow.cpp
 * @brief Test suite for Hamming window function implementation.
 *
 * Tests the mathematical correctness, properties, and behavior of the
 * Hamming window (two-term cosine with optimized sidelobe suppression).
 */

#include <gtest/gtest.h>
#include "window/HammingWindow.hpp"
#include "window/BaseWindow.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <numbers>

using namespace stft;

// ==========================================
// COMPILE-TIME CONCEPT VALIDATION
// ==========================================

/**
 * @brief Static assertion: HammingWindow must satisfy WindowFunction concept.
 */
static_assert(WindowFunction<HammingWindow>,
              "ERROR: HammingWindow must satisfy WindowFunction concept!");

// ==========================================
// TEST FIXTURE
// ==========================================

class HammingWindowTest : public ::testing::Test {
protected:
    const double TOL = 1e-10;
    const double COEFF_TOL = 1e-8;
    const std::size_t SMALL_SIZE = 16;
    const std::size_t MEDIUM_SIZE = 256;
    const std::size_t LARGE_SIZE = 4096;

    /**
     * @brief Manually computes Hamming coefficients for verification.
     */
    static double hammingCoefficient(std::size_t n, std::size_t N) {
        return 0.54 - 0.46 * std::cos(
            2.0 * std::numbers::pi * n / (N - 1)
        );
    }
};

// ==========================================
// TEST 1: Construction and Size
// ==========================================

/**
 * @brief Verifies that HammingWindow constructs correctly with various sizes.
 */
TEST_F(HammingWindowTest, ConstructionAndSize) {
    HammingWindow ham_small(SMALL_SIZE);
    EXPECT_EQ(ham_small.size(), SMALL_SIZE);

    HammingWindow ham_medium(MEDIUM_SIZE);
    EXPECT_EQ(ham_medium.size(), MEDIUM_SIZE);

    HammingWindow ham_large(LARGE_SIZE);
    EXPECT_EQ(ham_large.size(), LARGE_SIZE);
}

// ==========================================
// TEST 2: Mathematical Formula Correctness
// ==========================================

/**
 * @brief Verifies that generated coefficients match the mathematical formula.
 *
 * Formula: w[n] = 0.54 - 0.46*cos(2πn/(N-1))
 */
TEST_F(HammingWindowTest, FormulaCorrectness) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();

    for (std::size_t n = 0; n < MEDIUM_SIZE; ++n) {
        double expected = hammingCoefficient(n, MEDIUM_SIZE);
        EXPECT_NEAR(coeffs[n], expected, COEFF_TOL)
            << "Coefficient at index " << n << " does not match formula";
    }
}

// ==========================================
// TEST 3: Coefficient Range [0.08, 1.0]
// ==========================================

/**
 * @brief Verifies that all coefficients are in the valid range for Hamming.
 *
 * Unlike Hann and Blackman, Hamming window does NOT go to zero at endpoints.
 * Minimum value is 0.54 - 0.46 = 0.08 (at the endpoints).
 * Maximum value is 0.54 + 0.46 = 1.0 (at the center).
 */
TEST_F(HammingWindowTest, CoefficientRange) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();

    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i) {
        EXPECT_GE(coeffs[i], 0.08 - COEFF_TOL)
            << "Coefficient at index " << i << " is below minimum";
        EXPECT_LE(coeffs[i], 1.0 + COEFF_TOL)
            << "Coefficient at index " << i << " exceeds maximum";
    }
}

// ==========================================
// TEST 4: Non-Zero Endpoints
// ==========================================

/**
 * @brief Verifies that Hamming window does NOT taper to zero at endpoints.
 *
 * This is the key difference from Hann and Blackman windows.
 * At n=0 and n=N-1: w = 0.54 - 0.46*cos(0) = 0.54 - 0.46 = 0.08
 */
TEST_F(HammingWindowTest, NonZeroEndpoints) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();

    // At n=0: angle = 0, cos(0) = 1
    // w[0] = 0.54 - 0.46*1 = 0.08
    EXPECT_NEAR(coeffs[0], 0.08, COEFF_TOL)
        << "Hamming window should start at 0.08";

    // At n=N-1: angle = 2π, cos(2π) = 1
    // w[N-1] = 0.54 - 0.46*1 = 0.08
    EXPECT_NEAR(coeffs[MEDIUM_SIZE - 1], 0.08, COEFF_TOL)
        << "Hamming window should end at 0.08";
}

// ==========================================
// TEST 5: Symmetry
// ==========================================

/**
 * @brief Verifies that Hamming window is symmetric around the center.
 */
TEST_F(HammingWindowTest, Symmetry) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();

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
 * Note: Due to the discrete sampling and rounding, the peak is very close
 * to 1.0 but may not be exactly 1.0. We use a slightly relaxed tolerance.
 */
TEST_F(HammingWindowTest, PeakAtCenter) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();

    std::size_t center = MEDIUM_SIZE / 2;
    double center_value = coeffs[center];

    // Center should be approximately 1.0 (0.54 + 0.46)
    // Use relaxed tolerance due to discrete sampling effects
    EXPECT_NEAR(center_value, 1.0, 1e-4)
        << "Peak should be approximately 1.0 at center";

    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i) {
        EXPECT_LE(coeffs[i], center_value + 1e-8)
            << "Peak should be at center, but found larger value at index " << i;
    }
}

// ==========================================
// TEST 7: Coherent Gain Theoretical Value
// ==========================================

/**
 * @brief Verifies that coherent gain matches the theoretical value.
 *
 * For Hamming window, CG = 0.54 (the a0 coefficient).
 */
TEST_F(HammingWindowTest, CoherentGainTheoretical) {
    HammingWindow ham(MEDIUM_SIZE);
    double cg = ham.coherentGain();

    // Theoretical value: 0.54
    EXPECT_NEAR(cg, 0.54, 0.01)
        << "Coherent gain should be approximately 0.54";

    // Verify by manual calculation
    const auto& coeffs = ham.coefficients();
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
 * Hamming window has theoretical ENBW ≈ 1.36 bins (not exactly 1.30).
 * The exact value depends on the window size and computation method.
 * This is narrower than Blackman (1.73) but wider than Hann (1.50).
 * Hamming's narrower main lobe gives it better frequency resolution.
 */
TEST_F(HammingWindowTest, PowerBandwidth) {
    HammingWindow ham(MEDIUM_SIZE);
    double pb = ham.powerBandwidth();

    // Theoretical range: ~1.30 to ~1.40
    // (actual value is typically around 1.36 for typical window sizes)
    EXPECT_GT(pb, 1.25) << "ENBW seems too low";
    EXPECT_LT(pb, 1.45) << "ENBW seems too high";

    // Practical range check
    EXPECT_NEAR(pb, 1.36, 0.1)
        << "ENBW should be in the range 1.26-1.46";
}

// ==========================================
// TEST 9: Apply Window In-Place
// ==========================================

/**
 * @brief Tests that apply() correctly multiplies signal by window coefficients.
 */
TEST_F(HammingWindowTest, ApplyWindowInPlace) {
    HammingWindow ham(SMALL_SIZE);

    // Create signal with all ones
    std::vector<double> signal(SMALL_SIZE, 1.0);

    ham.apply(signal);

    // Signal should now equal window coefficients
    const auto& coeffs = ham.coefficients();
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
TEST_F(HammingWindowTest, ApplyToVaryingSignal) {
    HammingWindow ham(SMALL_SIZE);

    // Create signal: [0, 1, 2, 3, ..., 15]
    std::vector<double> signal(SMALL_SIZE);
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        signal[i] = static_cast<double>(i);
    }

    ham.apply(signal);

    // Verify: signal[i] = i * w[i]
    const auto& coeffs = ham.coefficients();
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        double expected = i * coeffs[i];
        EXPECT_NEAR(signal[i], expected, TOL)
            << "apply() mismatch at index " << i;
    }
}

// ==========================================
// TEST 11: Smoothness (Smaller Transitions)
// ==========================================

/**
 * @brief Verifies that the window has smooth transitions.
 *
 * Hamming is a two-term window (smoother than Hann's one-term,
 * but less smooth than Blackman's three-term).
 */
TEST_F(HammingWindowTest, SmoothTransitions) {
    HammingWindow ham(MEDIUM_SIZE);
    const auto& coeffs = ham.coefficients();

    // Check that differences between adjacent coefficients are reasonable
    for (std::size_t i = 1; i < MEDIUM_SIZE; ++i) {
        double diff = std::abs(coeffs[i] - coeffs[i - 1]);
        EXPECT_LT(diff, 0.1)
            << "Window has a sharp transition at index " << i;
    }
}

// ==========================================
// TEST 12: Comparison with Hann Window
// ==========================================

/**
 * @brief Verifies key differences between Hamming and Hann.
 *
 * - Hann: 0.5 - 0.5*cos(...) → endpoints = 0
 * - Hamming: 0.54 - 0.46*cos(...) → endpoints = 0.08
 */
TEST_F(HammingWindowTest, ComparisonWithHann) {
    HammingWindow ham(SMALL_SIZE);
    const auto& ham_coeffs = ham.coefficients();

    // Hamming endpoints should NOT be zero
    EXPECT_GT(ham_coeffs[0], 0.01)
        << "Hamming endpoints should be non-zero (0.08)";
    EXPECT_GT(ham_coeffs[SMALL_SIZE - 1], 0.01)
        << "Hamming endpoints should be non-zero (0.08)";

    // Hamming endpoints should be 0.08
    EXPECT_NEAR(ham_coeffs[0], 0.08, COEFF_TOL);
    EXPECT_NEAR(ham_coeffs[SMALL_SIZE - 1], 0.08, COEFF_TOL);
}

// ==========================================
// TEST 13: Scaling Property
// ==========================================

/**
 * @brief Verifies that scaling works correctly.
 */
TEST_F(HammingWindowTest, ScalingProperty) {
    HammingWindow ham(SMALL_SIZE);

    std::vector<double> signal1(SMALL_SIZE, 2.0);
    std::vector<double> signal2(SMALL_SIZE, 4.0);

    ham.apply(signal1);
    ham.apply(signal2);

    // signal2 should be 2x signal1
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal2[i], 2.0 * signal1[i], TOL)
            << "Scaling property failed at index " << i;
    }
}

// ==========================================
// TEST 14: Multiple Instances Independence
// ==========================================

/**
 * @brief Verifies that multiple windows don't interfere.
 */
TEST_F(HammingWindowTest, MultipleInstancesIndependence) {
    HammingWindow ham1(SMALL_SIZE);
    HammingWindow ham2(MEDIUM_SIZE);

    EXPECT_EQ(ham1.size(), SMALL_SIZE);
    EXPECT_EQ(ham2.size(), MEDIUM_SIZE);

    std::vector<double> signal1(SMALL_SIZE, 1.0);
    std::vector<double> signal2(MEDIUM_SIZE, 1.0);

    ham1.apply(signal1);
    ham2.apply(signal2);

    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal1[i], ham1.coefficients()[i], TOL);
    }
    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i) {
        EXPECT_NEAR(signal2[i], ham2.coefficients()[i], TOL);
    }
}

// ==========================================
// TEST 15: Large Window Performance
// ==========================================

/**
 * @brief Tests that large windows can be created and used efficiently.
 */
TEST_F(HammingWindowTest, LargeWindowPerformance) {
    EXPECT_NO_THROW({
        HammingWindow ham(LARGE_SIZE);
        EXPECT_EQ(ham.size(), LARGE_SIZE);

        std::vector<double> signal(LARGE_SIZE, 1.0);
        ham.apply(signal);

        EXPECT_NEAR(ham.coherentGain(), 0.54, 0.01);
    });
}

// ==========================================
// TEST 16: Copy Semantics
// ==========================================

/**
 * @brief Tests that windows can be copied correctly.
 */
TEST_F(HammingWindowTest, CopySemantics) {
    HammingWindow ham1(SMALL_SIZE);
    HammingWindow ham2 = ham1;

    EXPECT_EQ(ham1.size(), ham2.size());
    EXPECT_EQ(ham1.coefficients(), ham2.coefficients());
}

// ==========================================
// TEST 17: Move Semantics
// ==========================================

/**
 * @brief Tests that windows can be moved efficiently.
 */
TEST_F(HammingWindowTest, MoveSemantics) {
    HammingWindow ham1(MEDIUM_SIZE);
    std::size_t original_size = ham1.size();

    HammingWindow ham2 = std::move(ham1);

    EXPECT_EQ(ham2.size(), original_size);
}

// ==========================================
// TEST 18: Edge Case - Minimum Size
// ==========================================

/**
 * @brief Tests the minimum valid window size (N=2).
 */
TEST_F(HammingWindowTest, MinimumSize) {
    HammingWindow ham(2);
    EXPECT_EQ(ham.size(), 2);

    const auto& coeffs = ham.coefficients();
    EXPECT_EQ(coeffs.size(), 2);

    // Both endpoints should be 0.08
    EXPECT_NEAR(coeffs[0], 0.08, COEFF_TOL);
    EXPECT_NEAR(coeffs[1], 0.08, COEFF_TOL);
}

// ==========================================
// TEST 19: Multi-Size Formula Verification
// ==========================================

/**
 * @brief Direct comparison across multiple window sizes.
 */
TEST_F(HammingWindowTest, MultiSizeFormulasVerification) {
    std::size_t sizes[] = {8, 16, 32, 64, 128, 256};

    for (auto size : sizes) {
        HammingWindow ham(size);
        const auto& coeffs = ham.coefficients();

        for (std::size_t n = 0; n < size; ++n) {
            double expected = hammingCoefficient(n, size);
            EXPECT_NEAR(coeffs[n], expected, COEFF_TOL)
                << "Formula mismatch at n=" << n << " for size=" << size;
        }
    }
}

// ==========================================
// TEST 20: Narrower Main Lobe Property
// ==========================================

/**
 * @brief Verifies that Hamming has narrower effective bandwidth than Blackman.
 *
 * Narrower main lobe → better frequency resolution at the cost of sidelobes.
 */
TEST_F(HammingWindowTest, NarrowerMainLobe) {
    HammingWindow ham(MEDIUM_SIZE);
    double ham_enbw = ham.powerBandwidth();

    // Hamming ENBW (~1.30) should be less than Blackman ENBW (~1.73)
    // This is a relative property check
    EXPECT_LT(ham_enbw, 1.5)
        << "Hamming should have ENBW < 1.5 for good frequency resolution";
}

/**
 * @brief operator() produces the same result as apply().
 */
TEST_F(HammingWindowTest, OperatorCallMatchesApply) {
    HammingWindow w(MEDIUM_SIZE);

    std::vector<double> signal_apply(MEDIUM_SIZE, 1.0);
    std::vector<double> signal_call(MEDIUM_SIZE, 1.0);

    w.apply(signal_apply);
    w(signal_call);

    ASSERT_EQ(signal_apply.size(), signal_call.size());
    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i)
        EXPECT_DOUBLE_EQ(signal_apply[i], signal_call[i]);
}
