/**
 * @file test_HannWindow.cpp
 * @brief Test suite for Hann window function implementation.
 *
 * Tests the mathematical correctness, properties, and behavior of the
 * Hann window (raised cosine with zero endpoints).
 */

#include <gtest/gtest.h>
#include "window/HannWindow.hpp"
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
 * @brief Static assertion: HannWindow must satisfy WindowFunction concept.
 */
static_assert(WindowFunction<HannWindow>,
              "ERROR: HannWindow must satisfy WindowFunction concept!");

// ==========================================
// TEST FIXTURE
// ==========================================

class HannWindowTest : public ::testing::Test {
protected:
    const double TOL = 1e-10;
    const double COEFF_TOL = 1e-8;
    const std::size_t SMALL_SIZE = 16;
    const std::size_t MEDIUM_SIZE = 256;
    const std::size_t LARGE_SIZE = 4096;

    /**
     * @brief Manually computes Hann coefficients for verification.
     */
    static double hannCoefficient(std::size_t n, std::size_t N) {
        return 0.5 * (1.0 - std::cos(
            2.0 * std::numbers::pi * n / (N - 1)
        ));
    }
};

// ==========================================
// TEST 1: Construction and Size
// ==========================================

/**
 * @brief Verifies that HannWindow constructs correctly with various sizes.
 */
TEST_F(HannWindowTest, ConstructionAndSize) {
    HannWindow hann_small(SMALL_SIZE);
    EXPECT_EQ(hann_small.size(), SMALL_SIZE);

    HannWindow hann_medium(MEDIUM_SIZE);
    EXPECT_EQ(hann_medium.size(), MEDIUM_SIZE);

    HannWindow hann_large(LARGE_SIZE);
    EXPECT_EQ(hann_large.size(), LARGE_SIZE);
}

// ==========================================
// TEST 2: Mathematical Formula Correctness
// ==========================================

/**
 * @brief Verifies that generated coefficients match the mathematical formula.
 *
 * Formula: w[n] = 0.5*(1 - cos(2πn/(N-1)))
 */
TEST_F(HannWindowTest, FormulaCorrectness) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();

    for (std::size_t n = 0; n < MEDIUM_SIZE; ++n) {
        double expected = hannCoefficient(n, MEDIUM_SIZE);
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
 * Clamping ensures this is guaranteed, not just approximate.
 */
TEST_F(HannWindowTest, CoefficientRange) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();

    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i) {
        EXPECT_GE(coeffs[i], 0.0)
            << "Coefficient at index " << i << " is negative";
        EXPECT_LE(coeffs[i], 1.0)
            << "Coefficient at index " << i << " exceeds 1.0";
    }
}

// ==========================================
// TEST 4: Zero Endpoints
// ==========================================

/**
 * @brief Verifies that Hann window tapers to zero at endpoints.
 *
 * At n=0 and n=N-1: w = 0.5*(1 - cos(0)) = 0.5*(1 - 1) = 0
 */
TEST_F(HannWindowTest, ZeroEndpoints) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();

    // At n=0: angle = 0, cos(0) = 1
    // w[0] = 0.5*(1 - 1) = 0
    EXPECT_NEAR(coeffs[0], 0.0, COEFF_TOL)
        << "Hann window should start at zero";

    // At n=N-1: angle = 2π, cos(2π) = 1
    // w[N-1] = 0.5*(1 - 1) = 0
    EXPECT_NEAR(coeffs[MEDIUM_SIZE - 1], 0.0, COEFF_TOL)
        << "Hann window should end at zero";
}

// ==========================================
// TEST 5: Symmetry
// ==========================================

/**
 * @brief Verifies that Hann window is symmetric around the center.
 */
TEST_F(HannWindowTest, Symmetry) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();

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
 * For Hann: center value = 0.5*(1 - cos(π)) = 0.5*(1 - (-1)) = 1.0
 */
TEST_F(HannWindowTest, PeakAtCenter) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();

    std::size_t center = MEDIUM_SIZE / 2;
    double center_value = coeffs[center];

    // Center should be approximately 1.0
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
 * For Hann window, CG = 0.5 (the a0 coefficient).
 */
TEST_F(HannWindowTest, CoherentGainTheoretical) {
    HannWindow hann(MEDIUM_SIZE);
    double cg = hann.coherentGain();

    // Theoretical value: 0.5
    EXPECT_NEAR(cg, 0.5, 0.01)
        << "Coherent gain should be approximately 0.5";

    // Verify by manual calculation
    const auto& coeffs = hann.coefficients();
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
 * Hann window has theoretical ENBW ≈ 1.50 bins.
 * This is between Hamming (1.36) and Blackman (1.73).
 */
TEST_F(HannWindowTest, PowerBandwidth) {
    HannWindow hann(MEDIUM_SIZE);
    double pb = hann.powerBandwidth();

    // Theoretical range: ~1.45 to ~1.55
    EXPECT_GT(pb, 1.45) << "ENBW seems too low";
    EXPECT_LT(pb, 1.55) << "ENBW seems too high";

    // Practical range check
    EXPECT_NEAR(pb, 1.5, 0.1)
        << "ENBW should be in the range 1.40-1.60";
}

// ==========================================
// TEST 9: Apply Window In-Place
// ==========================================

/**
 * @brief Tests that apply() correctly multiplies signal by window coefficients.
 */
TEST_F(HannWindowTest, ApplyWindowInPlace) {
    HannWindow hann(SMALL_SIZE);

    // Create signal with all ones
    std::vector<double> signal(SMALL_SIZE, 1.0);

    hann.apply(signal);

    // Signal should now equal window coefficients
    const auto& coeffs = hann.coefficients();
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
TEST_F(HannWindowTest, ApplyToVaryingSignal) {
    HannWindow hann(SMALL_SIZE);

    // Create signal: [0, 1, 2, 3, ..., 15]
    std::vector<double> signal(SMALL_SIZE);
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        signal[i] = static_cast<double>(i);
    }

    hann.apply(signal);

    // Verify: signal[i] = i * w[i]
    const auto& coeffs = hann.coefficients();
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        double expected = i * coeffs[i];
        EXPECT_NEAR(signal[i], expected, TOL)
            << "apply() mismatch at index " << i;
    }
}

// ==========================================
// TEST 11: Smooth Tapering
// ==========================================

/**
 * @brief Verifies that the window tapers smoothly (no sharp discontinuities).
 */
TEST_F(HannWindowTest, SmoothTapering) {
    HannWindow hann(MEDIUM_SIZE);
    const auto& coeffs = hann.coefficients();

    // Check that differences between adjacent coefficients are small
    for (std::size_t i = 1; i < MEDIUM_SIZE; ++i) {
        double diff = std::abs(coeffs[i] - coeffs[i - 1]);
        EXPECT_LT(diff, 0.1)
            << "Window has a sharp transition at index " << i;
    }
}

// ==========================================
// TEST 12: Scaling Property
// ==========================================

/**
 * @brief Verifies that scaling works correctly.
 */
TEST_F(HannWindowTest, ScalingProperty) {
    HannWindow hann(SMALL_SIZE);

    std::vector<double> signal1(SMALL_SIZE, 2.0);
    std::vector<double> signal2(SMALL_SIZE, 4.0);

    hann.apply(signal1);
    hann.apply(signal2);

    // signal2 should be 2x signal1
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal2[i], 2.0 * signal1[i], TOL)
            << "Scaling property failed at index " << i;
    }
}

// ==========================================
// TEST 13: Multiple Instances Independence
// ==========================================

/**
 * @brief Verifies that multiple windows don't interfere.
 */
TEST_F(HannWindowTest, MultipleInstancesIndependence) {
    HannWindow hann1(SMALL_SIZE);
    HannWindow hann2(MEDIUM_SIZE);

    EXPECT_EQ(hann1.size(), SMALL_SIZE);
    EXPECT_EQ(hann2.size(), MEDIUM_SIZE);

    std::vector<double> signal1(SMALL_SIZE, 1.0);
    std::vector<double> signal2(MEDIUM_SIZE, 1.0);

    hann1.apply(signal1);
    hann2.apply(signal2);

    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal1[i], hann1.coefficients()[i], TOL);
    }
    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i) {
        EXPECT_NEAR(signal2[i], hann2.coefficients()[i], TOL);
    }
}

// ==========================================
// TEST 14: Large Window Performance
// ==========================================

/**
 * @brief Tests that large windows can be created and used efficiently.
 */
TEST_F(HannWindowTest, LargeWindowPerformance) {
    EXPECT_NO_THROW({
        HannWindow hann(LARGE_SIZE);
        EXPECT_EQ(hann.size(), LARGE_SIZE);

        std::vector<double> signal(LARGE_SIZE, 1.0);
        hann.apply(signal);

        EXPECT_NEAR(hann.coherentGain(), 0.5, 0.01);
    });
}

// ==========================================
// TEST 15: Copy Semantics
// ==========================================

/**
 * @brief Tests that windows can be copied correctly.
 */
TEST_F(HannWindowTest, CopySemantics) {
    HannWindow hann1(SMALL_SIZE);
    HannWindow hann2 = hann1;

    EXPECT_EQ(hann1.size(), hann2.size());
    EXPECT_EQ(hann1.coefficients(), hann2.coefficients());
}

// ==========================================
// TEST 16: Move Semantics
// ==========================================

/**
 * @brief Tests that windows can be moved efficiently.
 */
TEST_F(HannWindowTest, MoveSemantics) {
    HannWindow hann1(MEDIUM_SIZE);
    std::size_t original_size = hann1.size();

    HannWindow hann2 = std::move(hann1);

    EXPECT_EQ(hann2.size(), original_size);
}

// ==========================================
// TEST 17: Edge Case - Minimum Size
// ==========================================

/**
 * @brief Tests the minimum valid window size (N=2).
 */
TEST_F(HannWindowTest, MinimumSize) {
    HannWindow hann(2);
    EXPECT_EQ(hann.size(), 2);

    const auto& coeffs = hann.coefficients();
    EXPECT_EQ(coeffs.size(), 2);

    // Both endpoints should be 0
    EXPECT_NEAR(coeffs[0], 0.0, COEFF_TOL);
    EXPECT_NEAR(coeffs[1], 0.0, COEFF_TOL);
}

// ==========================================
// TEST 18: Multi-Size Formula Verification
// ==========================================

/**
 * @brief Direct comparison across multiple window sizes.
 */
TEST_F(HannWindowTest, MultiSizeFormulasVerification) {
    std::size_t sizes[] = {8, 16, 32, 64, 128, 256};

    for (auto size : sizes) {
        HannWindow hann(size);
        const auto& coeffs = hann.coefficients();

        for (std::size_t n = 0; n < size; ++n) {
            double expected = hannCoefficient(n, size);
            EXPECT_NEAR(coeffs[n], expected, COEFF_TOL)
                << "Formula mismatch at n=" << n << " for size=" << size;
        }
    }
}

// ==========================================
// TEST 19: Comparison with Window Properties
// ==========================================

/**
 * @brief Verifies key properties of Hann window compared to others.
 *
 * - Endpoints: zero (like Blackman, unlike Hamming)
 * - Main lobe width: 4 bins
 * - ENBW: 1.5 bins (middle ground between Hamming and Blackman)
 */
TEST_F(HannWindowTest, GeneralPurposeBehavior) {
    HannWindow hann(SMALL_SIZE);
    const auto& coeffs = hann.coefficients();

    // Endpoints should be zero (good for spectral leakage suppression)
    EXPECT_NEAR(coeffs[0], 0.0, COEFF_TOL);
    EXPECT_NEAR(coeffs[SMALL_SIZE - 1], 0.0, COEFF_TOL);

    // Peak should be at center
    std::size_t center = SMALL_SIZE / 2;
    double max_value = *std::max_element(coeffs.begin(), coeffs.end());
    EXPECT_NEAR(coeffs[center], max_value, 1e-4);
}

// ==========================================
// TEST 20: Signal Energy Preservation Check
// ==========================================

/**
 * @brief Verifies that windowing preserves reasonable energy levels.
 *
 * When a constant signal is windowed, the result should have
 * energy proportional to the coherent gain.
 */
TEST_F(HannWindowTest, EnergyPreservation) {
    HannWindow hann(MEDIUM_SIZE);

    // Create constant signal
    std::vector<double> signal(MEDIUM_SIZE, 1.0);
    double original_energy = std::accumulate(signal.begin(), signal.end(), 0.0);

    hann.apply(signal);
    double windowed_energy = std::accumulate(signal.begin(), signal.end(), 0.0);

    // Windowed energy should be CG * original
    double expected_energy = hann.coherentGain() * original_energy;
    EXPECT_NEAR(windowed_energy, expected_energy, TOL)
        << "Energy is not preserved correctly";
}
