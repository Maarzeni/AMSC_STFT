/**
 * @file test_BaseWindow.cpp
 * @brief Comprehensive test suite for BaseWindow CRTP class.
 *
 * Tests the shared storage, CRTP dispatch, DSP utilities, and error handling
 * of the window function framework.
 */

#include <gtest/gtest.h>
#include "window/BaseWindow.hpp"
#include <vector>
#include <cmath>
#include <numeric>

using namespace stft;

// ==========================================
// DUMMY WINDOW IMPLEMENTATION FOR TESTING
// ==========================================

/**
 * @class RectangularWindow
 * @brief Rectangular (boxcar) window: all coefficients = 1.0
 *
 * Used as a minimal concrete implementation to test BaseWindow functionality.
 */
class RectangularWindow : public BaseWindow<RectangularWindow> {
    friend class BaseWindow<RectangularWindow>;

private:
    /**
     * @brief Generates rectangular window coefficients (all ones).
     * @param N Window size.
     * @return Vector of N ones.
     */
    std::vector<double> generate(std::size_t N) const {
        return std::vector<double>(N, 1.0);
    }

public:
    /**
     * @brief Constructs a rectangular window.
     * @param N Frame size.
     */
    explicit RectangularWindow(std::size_t N)
        : BaseWindow<RectangularWindow>(N) {}
};

/**
 * @class TriangularWindow
 * @brief Triangular window: coefficients form a triangle peak.
 *
 * Formula: w[n] = 1 - |2(n - (N-1)/2) / N|
 */
class TriangularWindow : public BaseWindow<TriangularWindow> {
    friend class BaseWindow<TriangularWindow>;

private:
    /**
     * @brief Generates triangular window coefficients.
     * @param N Window size.
     * @return Vector of N triangular coefficients.
     */
    std::vector<double> generate(std::size_t N) const {
        std::vector<double> coeffs(N);
        double center = (N - 1.0) / 2.0;
        for (std::size_t i = 0; i < N; ++i) {
            coeffs[i] = 1.0 - std::abs(2.0 * (i - center) / N);
        }
        return coeffs;
    }

public:
    /**
     * @brief Constructs a triangular window.
     * @param N Frame size.
     */
    explicit TriangularWindow(std::size_t N)
        : BaseWindow<TriangularWindow>(N) {}
};

// ==========================================
// TEST FIXTURE
// ==========================================

class BaseWindowTest : public ::testing::Test {
protected:
    const double TOL = 1e-10;
    const std::size_t SMALL_SIZE = 8;
    const std::size_t MEDIUM_SIZE = 256;
    const std::size_t LARGE_SIZE = 4096;
};

// ==========================================
// TEST 1: Construction and Size Validation
// ==========================================

/**
 * @brief Verifies that windows are constructed correctly and store size.
 */
TEST_F(BaseWindowTest, ConstructionAndSize) {
    RectangularWindow rect(SMALL_SIZE);
    EXPECT_EQ(rect.size(), SMALL_SIZE);

    TriangularWindow tri(MEDIUM_SIZE);
    EXPECT_EQ(tri.size(), MEDIUM_SIZE);

    RectangularWindow large(LARGE_SIZE);
    EXPECT_EQ(large.size(), LARGE_SIZE);
}

// ==========================================
// TEST 2: Coefficients Access
// ==========================================

/**
 * @brief Tests that coefficients() returns correct data.
 */
TEST_F(BaseWindowTest, CoefficientsAccess) {
    RectangularWindow rect(SMALL_SIZE);
    const auto& coeffs = rect.coefficients();

    EXPECT_EQ(coeffs.size(), SMALL_SIZE);

    // Rectangular window should have all ones
    for (double coeff : coeffs) {
        EXPECT_NEAR(coeff, 1.0, TOL);
    }
}

// ==========================================
// TEST 3: Invalid Size Handling
// ==========================================

/**
 * @brief Verifies that windows reject invalid sizes (< 2).
 */
TEST_F(BaseWindowTest, InvalidSizeThrows) {
    EXPECT_THROW(RectangularWindow(0), std::invalid_argument);
    EXPECT_THROW(RectangularWindow(1), std::invalid_argument);

    // Size 2 should be OK (minimum valid)
    EXPECT_NO_THROW(RectangularWindow(2));
}

// ==========================================
// TEST 4: Coherent Gain Calculation
// ==========================================

/**
 * @brief Tests coherent gain (mean of window coefficients).
 *
 * For rectangular window: CG = 1.0 (all coefficients are 1)
 * For triangular window: CG = 0.5 (triangle averages to 0.5)
 */
TEST_F(BaseWindowTest, CoherentGain) {
    RectangularWindow rect(100);
    double cg_rect = rect.coherentGain();
    EXPECT_NEAR(cg_rect, 1.0, TOL);

    TriangularWindow tri(100);
    double cg_tri = tri.coherentGain();
    EXPECT_NEAR(cg_tri, 0.5, TOL);

    // Manual verification for triangle
    const auto& tri_coeffs = tri.coefficients();
    double manual_sum = std::accumulate(tri_coeffs.begin(), tri_coeffs.end(), 0.0);
    double manual_cg = manual_sum / tri_coeffs.size();
    EXPECT_NEAR(cg_tri, manual_cg, TOL);
}

// ==========================================
// TEST 5: Power Bandwidth (ENBW)
// ==========================================

/**
 * @brief Tests equivalent noise bandwidth calculation.
 *
 * ENBW = N * sum(w[n]^2) / (sum w[n])^2
 */
TEST_F(BaseWindowTest, PowerBandwidth) {
    RectangularWindow rect(100);
    double enbw_rect = rect.powerBandwidth();
    // For rectangular: sum(w) = N, sum(w^2) = N
    // ENBW = N * N / (N^2) = 1.0
    EXPECT_NEAR(enbw_rect, 1.0, TOL);

    TriangularWindow tri(100);
    double enbw_tri = tri.powerBandwidth();
    // Triangular window should have ENBW ≈ 1.33 (theoretical value)
    EXPECT_NEAR(enbw_tri, 1.333, 0.01);
}

// ==========================================
// TEST 6: Apply Window In-Place
// ==========================================

/**
 * @brief Tests in-place window application to a signal.
 */
TEST_F(BaseWindowTest, ApplyWindowInPlace) {
    RectangularWindow rect(SMALL_SIZE);

    // Create a test signal with all ones
    std::vector<double> signal(SMALL_SIZE, 2.0);
    rect.apply(signal);

    // Rectangular window has all coeffs = 1, so signal should be unchanged
    for (double val : signal) {
        EXPECT_NEAR(val, 2.0, TOL);
    }
}

// ==========================================
// TEST 7: Apply Window with Triangular
// ==========================================

/**
 * @brief Tests windowing with triangular window (non-trivial coefficients).
 */
TEST_F(BaseWindowTest, ApplyWindowTriangular) {
    TriangularWindow tri(SMALL_SIZE);

    std::vector<double> signal(SMALL_SIZE, 1.0);
    std::vector<double> expected = tri.coefficients();

    tri.apply(signal);

    // Signal should now equal the window coefficients
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal[i], expected[i], TOL);
    }
}

// ==========================================
// TEST 8: Apply Rejects Wrong Size
// ==========================================

/**
 * @brief Verifies that apply() throws on size mismatch.
 */
TEST_F(BaseWindowTest, ApplyRejectsWrongSize) {
    RectangularWindow rect(SMALL_SIZE);

    std::vector<double> signal(SMALL_SIZE + 1, 1.0);

    EXPECT_THROW(rect.apply(signal), std::invalid_argument);
}

// ==========================================
// TEST 9: Copy Semantics
// ==========================================

/**
 * @brief Tests that windows can be copied correctly.
 */
TEST_F(BaseWindowTest, CopySemantics) {
    RectangularWindow rect1(SMALL_SIZE);
    RectangularWindow rect2 = rect1; // Copy constructor

    EXPECT_EQ(rect1.size(), rect2.size());
    EXPECT_EQ(rect1.coefficients(), rect2.coefficients());

    // Modify rect2 by applying to a signal
    std::vector<double> signal1(SMALL_SIZE, 1.0);
    rect2.apply(signal1);

    // rect1 should be unaffected
    std::vector<double> signal2(SMALL_SIZE, 1.0);
    rect1.apply(signal2);
    EXPECT_EQ(signal1, signal2);
}

// ==========================================
// TEST 10: Move Semantics
// ==========================================

/**
 * @brief Tests that windows can be moved efficiently.
 */
TEST_F(BaseWindowTest, MoveSemantics) {
    RectangularWindow rect1(MEDIUM_SIZE);
    std::size_t original_size = rect1.size();

    RectangularWindow rect2 = std::move(rect1);

    EXPECT_EQ(rect2.size(), original_size);
    EXPECT_EQ(rect2.coefficients().size(), original_size);
}

// ==========================================
// TEST 11: Large Window Performance
// ==========================================

/**
 * @brief Verifies that large windows can be created and used efficiently.
 */
TEST_F(BaseWindowTest, LargeWindowPerformance) {
    EXPECT_NO_THROW({
        RectangularWindow large(LARGE_SIZE);
        EXPECT_EQ(large.size(), LARGE_SIZE);
        EXPECT_EQ(large.coefficients().size(), LARGE_SIZE);

        std::vector<double> signal(LARGE_SIZE, 1.0);
        large.apply(signal);
    });
}

// ==========================================
// TEST 12: Multiple Window Instances
// ==========================================

/**
 * @brief Tests that multiple windows can coexist without interference.
 */
TEST_F(BaseWindowTest, MultipleWindowInstances) {
    RectangularWindow rect(SMALL_SIZE);
    TriangularWindow tri(SMALL_SIZE);

    const auto& rect_coeffs = rect.coefficients();
    const auto& tri_coeffs = tri.coefficients();

    // Verify they are different
    bool all_different = true;
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        if (std::abs(rect_coeffs[i] - tri_coeffs[i]) < TOL) {
            all_different = false;
            break;
        }
    }
    EXPECT_TRUE(all_different);

    // Apply to different signals independently
    std::vector<double> signal_rect(SMALL_SIZE, 2.0);
    std::vector<double> signal_tri(SMALL_SIZE, 2.0);

    rect.apply(signal_rect);
    tri.apply(signal_tri);

    // Rectangular window: signal[i] *= 1.0 → unchanged
    // Triangular window: signal[i] *= tri_coeffs[i] → different
    for (std::size_t i = 0; i < SMALL_SIZE; ++i) {
        EXPECT_NEAR(signal_rect[i], 2.0, TOL);
        EXPECT_NEAR(signal_tri[i], 2.0 * tri_coeffs[i], TOL);
    }
}

// ==========================================
// TEST 13: Coefficient Immutability
// ==========================================

/**
 * @brief Verifies that coefficients() returns const reference (read-only).
 *
 * This is a compile-time check, but we also verify values don't change.
 */
TEST_F(BaseWindowTest, CoefficientImmutability) {
    RectangularWindow rect(SMALL_SIZE);

    const auto& coeffs1 = rect.coefficients();
    std::vector<double> expected(coeffs1);

    // Try to apply and verify coefficients haven't changed internally
    std::vector<double> signal(SMALL_SIZE, 1.0);
    rect.apply(signal);

    const auto& coeffs2 = rect.coefficients();
    EXPECT_EQ(coeffs1, coeffs2);
    EXPECT_EQ(coeffs2, expected);
}

// ==========================================
// TEST 14: Multiple Applications
// ==========================================

/**
 * @brief Tests that apply() can be called multiple times with correct accumulation.
 */
TEST_F(BaseWindowTest, MultipleApplications) {
    RectangularWindow rect(SMALL_SIZE);

    std::vector<double> signal(SMALL_SIZE, 1.0);

    // Apply once
    rect.apply(signal);
    for (double val : signal) {
        EXPECT_NEAR(val, 1.0, TOL); // Still 1 (multiply by 1)
    }

    // Apply again
    rect.apply(signal);
    for (double val : signal) {
        EXPECT_NEAR(val, 1.0, TOL); // Still 1
    }
}

// ==========================================
// TEST 15: Edge Case - Size 2
// ==========================================

/**
 * @brief Tests the minimum valid window size (N=2).
 */
TEST_F(BaseWindowTest, MinimumSize) {
    RectangularWindow rect(2);
    EXPECT_EQ(rect.size(), 2);
    EXPECT_EQ(rect.coefficients().size(), 2);

    double cg = rect.coherentGain();
    EXPECT_NEAR(cg, 1.0, TOL);

    std::vector<double> signal{1.0, 2.0};
    rect.apply(signal);
    EXPECT_NEAR(signal[0], 1.0, TOL);
    EXPECT_NEAR(signal[1], 2.0, TOL);
}

// ==========================================
// CONCEPT TESTS
// ==========================================

/**
 * @brief Compile-time verification that window types satisfy WindowFunction concept.
 *
 * @details
 * These static assertions are evaluated at compile time. If they fail, the
 * compilation will fail with a clear error message indicating which requirement
 * is not satisfied.
 *
 * The WindowFunction concept requires:
 *   - Constructibility from std::size_t
 *   - size() → std::size_t
 *   - apply(std::vector<double>&) → void
 *   - coefficients() → const std::vector<double>&
 *   - coherentGain() → double
 *   - powerBandwidth() → double
 */

// ==========================================
// TEST 16: Concept Verification - RectangularWindow
// ==========================================

/**
 * @brief Verifies that RectangularWindow satisfies the WindowFunction concept.
 */
TEST_F(BaseWindowTest, ConceptRectangularWindow) {
    // This test verifies the concept is satisfied at runtime through usage.
    // The actual concept check happens at compile-time via ASSERT_WINDOW_FUNCTION.
    
    RectangularWindow rect(SMALL_SIZE);
    
    // These operations are guaranteed to work by the concept
    EXPECT_EQ(rect.size(), SMALL_SIZE);
    
    const auto& coeffs = rect.coefficients();
    EXPECT_EQ(coeffs.size(), SMALL_SIZE);
    
    double cg = rect.coherentGain();
    EXPECT_GT(cg, 0.0);
    EXPECT_LE(cg, 1.0);
    
    double pw = rect.powerBandwidth();
    EXPECT_GT(pw, 0.0);
    
    std::vector<double> signal(SMALL_SIZE, 1.0);
    rect.apply(signal);  // Should not throw
    
    EXPECT_NO_FATAL_FAILURE();
}

// ==========================================
// TEST 17: Concept Verification - TriangularWindow
// ==========================================

/**
 * @brief Verifies that TriangularWindow satisfies the WindowFunction concept.
 */
TEST_F(BaseWindowTest, ConceptTriangularWindow) {
    TriangularWindow tri(SMALL_SIZE);
    
    EXPECT_EQ(tri.size(), SMALL_SIZE);
    
    const auto& coeffs = tri.coefficients();
    EXPECT_EQ(coeffs.size(), SMALL_SIZE);
    
    double cg = tri.coherentGain();
    EXPECT_GT(cg, 0.0);
    EXPECT_LE(cg, 1.0);
    
    double pw = tri.powerBandwidth();
    EXPECT_GT(pw, 0.0);
    
    std::vector<double> signal(SMALL_SIZE, 1.0);
    tri.apply(signal);  // Should not throw
    
    EXPECT_NO_FATAL_FAILURE();
}

// ==========================================
// TEST 18: Concept Requirements - Return Types
// ==========================================

/**
 * @brief Compile-time check: verifies return types match WindowFunction requirements.
 *
 * @details
 * This test exists to verify (at runtime) that the methods return the correct
 * types as specified by the WindowFunction concept. The compile-time verification
 * happens via ASSERT_WINDOW_FUNCTION macro.
 */
TEST_F(BaseWindowTest, ConceptReturnTypes) {
    RectangularWindow rect(SMALL_SIZE);
    
    // size() must return std::size_t (not int, not size_type, exactly std::size_t)
    static_assert(std::same_as<decltype(rect.size()), std::size_t>);

    // coefficients() must return const std::vector<double>&
    static_assert(std::same_as<
        decltype(rect.coefficients()),
        const std::vector<double>&
    >);

    // coherentGain() must return double
    static_assert(std::same_as<decltype(rect.coherentGain()), double>);

    // powerBandwidth() must return double
    static_assert(std::same_as<decltype(rect.powerBandwidth()), double>);

    // apply() must return void
    std::vector<double> signal(SMALL_SIZE, 1.0);
    static_assert(std::same_as<decltype(rect.apply(signal)), void>);
}

// ==========================================
// TEST 19: Concept - Constructibility Check
// ==========================================

/**
 * @brief Verifies that window types are constructible from std::size_t.
 *
 * @details
 * The WindowFunction concept requires that a type W is constructible from
 * a single std::size_t argument. This is critical for generic code that
 * creates windows with a given frame size.
 */
TEST_F(BaseWindowTest, ConceptConstructibility) {
    // These should all compile and not throw
    RectangularWindow rect(SMALL_SIZE);
    TriangularWindow tri(MEDIUM_SIZE);
    
    EXPECT_EQ(rect.size(), SMALL_SIZE);
    EXPECT_EQ(tri.size(), MEDIUM_SIZE);
    
    // Verify that std::constructible_from is satisfied
    static_assert(std::constructible_from<RectangularWindow, std::size_t>);
    static_assert(std::constructible_from<TriangularWindow, std::size_t>);
}

// ==========================================
// MACRO VERIFICATION
// ==========================================

/**
 * @brief Compile-time verification macro for both test windows.
 *
 * @details
 * The ASSERT_WINDOW_FUNCTION macro should be placed at global scope in
 * production code (e.g., at the end of each concrete window header).
 * Here, we verify it works correctly by invoking it for our test classes.
 *
 * If either assertion fails, the compilation fails with a clear message.
 */
ASSERT_WINDOW_FUNCTION(RectangularWindow);
ASSERT_WINDOW_FUNCTION(TriangularWindow);

// ==========================================
// CALLABLE OBJECT (FUNCTOR) TESTS
// ==========================================

/**
 * @brief operator() produces the same result as apply().
 *
 * @details
 * Both call paths must be equivalent: apply() is the canonical implementation
 * and operator() delegates to it, so element-wise results must match exactly.
 */
TEST_F(BaseWindowTest, OperatorCallMatchesApply) {
    RectangularWindow w(MEDIUM_SIZE);

    std::vector<double> signal_apply(MEDIUM_SIZE, 2.0);
    std::vector<double> signal_call(MEDIUM_SIZE, 2.0);

    w.apply(signal_apply);
    w(signal_call);

    ASSERT_EQ(signal_apply.size(), signal_call.size());
    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i)
        EXPECT_DOUBLE_EQ(signal_apply[i], signal_call[i]);
}

/**
 * @brief operator() propagates the size-mismatch exception from apply().
 */
TEST_F(BaseWindowTest, OperatorCallThrowsOnSizeMismatch) {
    RectangularWindow w(MEDIUM_SIZE);
    std::vector<double> wrong_size(MEDIUM_SIZE + 1, 1.0);
    EXPECT_THROW(w(wrong_size), std::invalid_argument);
}

/**
 * @brief operator() works on both RectangularWindow and TriangularWindow,
 *        verifying the functor interface through the CRTP hierarchy.
 */
TEST_F(BaseWindowTest, OperatorCallWorksForDifferentWindowTypes) {
    TriangularWindow w(MEDIUM_SIZE);

    std::vector<double> signal_apply(MEDIUM_SIZE, 3.0);
    std::vector<double> signal_call(MEDIUM_SIZE, 3.0);

    w.apply(signal_apply);
    w(signal_call);

    for (std::size_t i = 0; i < MEDIUM_SIZE; ++i)
        EXPECT_DOUBLE_EQ(signal_apply[i], signal_call[i]);
}
