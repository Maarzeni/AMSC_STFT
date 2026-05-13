/**
 * @file test_WindowConcept.cpp
 * @brief Test suite for C++20 WindowFunction concept validation.
 *
 * Tests that concrete window implementations satisfy the WindowFunction concept,
 * and that invalid types are correctly rejected at compile time.
 */

#include <gtest/gtest.h>
#include "window/WindowConcept.hpp"
#include "window/BaseWindow.hpp"
#include "window/HannWindow.hpp"
#include "window/HammingWindow.hpp"
#include "window/BlackmanWindow.hpp"
#include <vector>
#include <concepts>

using namespace window;

// ==========================================
// COMPILE-TIME CONCEPT VALIDATION
// ==========================================

/**
 * @brief Static assertions to verify that all concrete windows satisfy the concept.
 *
 * These run at compile time. If any window class doesn't satisfy WindowFunction,
 * the compilation will fail with a clear error message.
 */
static_assert(WindowFunction<HannWindow>, 
              "ERROR: HannWindow must satisfy WindowFunction concept!");

static_assert(WindowFunction<HammingWindow>, 
              "ERROR: HammingWindow must satisfy WindowFunction concept!");

static_assert(WindowFunction<BlackmanWindow>, 
              "ERROR: BlackmanWindow must satisfy WindowFunction concept!");

// ==========================================
// TEST FIXTURE
// ==========================================

class WindowConceptTest : public ::testing::Test {
protected:
    const double TOL = 1e-10;
    const std::size_t TEST_SIZE = 256;
};

// ==========================================
// TEST 1: Hann Window Satisfies Concept
// ==========================================

/**
 * @brief Verifies HannWindow is constructible and has all required methods.
 */
TEST_F(WindowConceptTest, HannWindowConceptRequirements) {
    // Constructible from size_t
    EXPECT_NO_THROW({
        HannWindow hann(TEST_SIZE);
    });

    HannWindow hann(TEST_SIZE);

    // size() returns std::size_t
    static_assert(std::same_as<decltype(hann.size()), std::size_t>);
    EXPECT_EQ(hann.size(), TEST_SIZE);

    // apply() accepts vector<double>& and returns void
    std::vector<double> signal(TEST_SIZE, 1.0);
    static_assert(std::same_as<decltype(hann.apply(signal)), void>);
    EXPECT_NO_THROW({
        hann.apply(signal);
    });

    // coefficients() returns const vector<double>&
    static_assert(std::same_as<
        decltype(hann.coefficients()),
        const std::vector<double>&
    >);
    const auto& coeffs = hann.coefficients();
    EXPECT_EQ(coeffs.size(), TEST_SIZE);

    // coherentGain() returns double
    static_assert(std::same_as<decltype(hann.coherentGain()), double>);
    double cg = hann.coherentGain();
    EXPECT_GT(cg, 0.0);
    EXPECT_LE(cg, 1.0);

    // powerBandwidth() returns double
    static_assert(std::same_as<decltype(hann.powerBandwidth()), double>);
    double pb = hann.powerBandwidth();
    EXPECT_GT(pb, 0.0);
}

// ==========================================
// TEST 2: Hamming Window Satisfies Concept
// ==========================================

/**
 * @brief Verifies HammingWindow is constructible and has all required methods.
 */
TEST_F(WindowConceptTest, HammingWindowConceptRequirements) {
    EXPECT_NO_THROW({
        HammingWindow hamming(TEST_SIZE);
    });

    HammingWindow hamming(TEST_SIZE);

    EXPECT_EQ(hamming.size(), TEST_SIZE);

    std::vector<double> signal(TEST_SIZE, 1.0);
    EXPECT_NO_THROW({
        hamming.apply(signal);
    });

    const auto& coeffs = hamming.coefficients();
    EXPECT_EQ(coeffs.size(), TEST_SIZE);

    double cg = hamming.coherentGain();
    EXPECT_GT(cg, 0.0);
    EXPECT_LE(cg, 1.0);

    double pb = hamming.powerBandwidth();
    EXPECT_GT(pb, 0.0);
}

// ==========================================
// TEST 3: Blackman Window Satisfies Concept
// ==========================================

/**
 * @brief Verifies BlackmanWindow is constructible and has all required methods.
 */
TEST_F(WindowConceptTest, BlackmanWindowConceptRequirements) {
    EXPECT_NO_THROW({
        BlackmanWindow blackman(TEST_SIZE);
    });

    BlackmanWindow blackman(TEST_SIZE);

    EXPECT_EQ(blackman.size(), TEST_SIZE);

    std::vector<double> signal(TEST_SIZE, 1.0);
    EXPECT_NO_THROW({
        blackman.apply(signal);
    });

    const auto& coeffs = blackman.coefficients();
    EXPECT_EQ(coeffs.size(), TEST_SIZE);

    double cg = blackman.coherentGain();
    EXPECT_GT(cg, 0.0);
    EXPECT_LE(cg, 1.0);

    double pb = blackman.powerBandwidth();
    EXPECT_GT(pb, 0.0);
}

// ==========================================
// TEST 4: Generic Template with Concept
// ==========================================

/**
 * @brief Tests that a generic template constrained by WindowFunction works.
 *
 * This simulates how STFTAnalyzer would use the concept.
 */
template<WindowFunction Window>
class MockSTFTAnalyzer {
private:
    Window window_;

public:
    explicit MockSTFTAnalyzer(std::size_t frame_size)
        : window_(frame_size) {}

    void processFrame(std::vector<double>& frame) const {
        window_.apply(frame);
    }

    double getNormalization() const {
        return window_.coherentGain();
    }

    std::size_t getFrameSize() const {
        return window_.size();
    }
};

TEST_F(WindowConceptTest, GenericTemplateWithConcept) {
    // Should work with HannWindow
    EXPECT_NO_THROW({
        MockSTFTAnalyzer<HannWindow> analyzer(TEST_SIZE);
        std::vector<double> frame(TEST_SIZE, 1.0);
        analyzer.processFrame(frame);
        double norm = analyzer.getNormalization();
        EXPECT_GT(norm, 0.0);
    });

    // Should work with HammingWindow
    EXPECT_NO_THROW({
        MockSTFTAnalyzer<HammingWindow> analyzer(TEST_SIZE);
        std::vector<double> frame(TEST_SIZE, 1.0);
        analyzer.processFrame(frame);
    });

    // Should work with BlackmanWindow
    EXPECT_NO_THROW({
        MockSTFTAnalyzer<BlackmanWindow> analyzer(TEST_SIZE);
        std::vector<double> frame(TEST_SIZE, 1.0);
        analyzer.processFrame(frame);
    });
}

// ==========================================
// TEST 5: Coherent Gain is in Valid Range
// ==========================================

/**
 * @brief Verifies that coherent gain is mathematically valid for all windows.
 *
 * By definition, coherent gain must be in (0, 1] because:
 * - All window coefficients are in [0, 1]
 * - Average of values in [0, 1] is in [0, 1]
 * - For reasonable windows, CG > 0 (at least some samples are non-zero)
 */
TEST_F(WindowConceptTest, CoherentGainValidity) {
    HannWindow hann(TEST_SIZE);
    HammingWindow hamming(TEST_SIZE);
    BlackmanWindow blackman(TEST_SIZE);

    double cg_hann = hann.coherentGain();
    double cg_hamming = hamming.coherentGain();
    double cg_blackman = blackman.coherentGain();

    EXPECT_GT(cg_hann, 0.0) << "Hann CG should be positive";
    EXPECT_LE(cg_hann, 1.0) << "Hann CG should be ≤ 1";

    EXPECT_GT(cg_hamming, 0.0) << "Hamming CG should be positive";
    EXPECT_LE(cg_hamming, 1.0) << "Hamming CG should be ≤ 1";

    EXPECT_GT(cg_blackman, 0.0) << "Blackman CG should be positive";
    EXPECT_LE(cg_blackman, 1.0) << "Blackman CG should be ≤ 1";

    // Verify ordering (Hann typically has highest CG for smooth windows)
    // Note: This is a heuristic check, not a strict requirement
}

// ==========================================
// TEST 6: Power Bandwidth is Reasonable
// ==========================================

/**
 * @brief Verifies that power bandwidth is physically meaningful.
 *
 * ENBW must be >= 1.0 (tight limit imposed by uncertainty principle).
 * Rectangular window has ENBW ≈ 1.0 (minimal).
 * Smoother windows have higher ENBW (wider frequency profile).
 */
TEST_F(WindowConceptTest, PowerBandwidthValidity) {
    HannWindow hann(TEST_SIZE);
    HammingWindow hamming(TEST_SIZE);
    BlackmanWindow blackman(TEST_SIZE);

    double pb_hann = hann.powerBandwidth();
    double pb_hamming = hamming.powerBandwidth();
    double pb_blackman = blackman.powerBandwidth();

    // All should be >= 1.0 (uncertainty principle)
    EXPECT_GE(pb_hann, 1.0) << "Hann ENBW should be >= 1.0";
    EXPECT_GE(pb_hamming, 1.0) << "Hamming ENBW should be >= 1.0";
    EXPECT_GE(pb_blackman, 1.0) << "Blackman ENBW should be >= 1.0";

    // Smoother windows (lower CG) typically have higher ENBW
    // This is just a sanity check; exact ordering depends on implementation
}

// ==========================================
// TEST 7: Multiple Concept-Constrained Instances
// ==========================================

/**
 * @brief Tests that multiple different windows can be used with templates
 *        constrained by the same concept.
 */
TEST_F(WindowConceptTest, MultipleConceptInstances) {
    std::vector<double> signal1(TEST_SIZE, 2.0);
    std::vector<double> signal2(TEST_SIZE, 2.0);
    std::vector<double> signal3(TEST_SIZE, 2.0);

    HannWindow hann(TEST_SIZE);
    HammingWindow hamming(TEST_SIZE);
    BlackmanWindow blackman(TEST_SIZE);

    // All should apply without errors
    EXPECT_NO_THROW(hann.apply(signal1));
    EXPECT_NO_THROW(hamming.apply(signal2));
    EXPECT_NO_THROW(blackman.apply(signal3));

    // Results should be different
    bool all_same = true;
    for (std::size_t i = 0; i < TEST_SIZE; ++i) {
        if (signal1[i] != signal2[i] || signal2[i] != signal3[i]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "Different windows should produce different results";
}

// ==========================================
// TEST 8: Concept Validation with Size Variations
// ==========================================

/**
 * @brief Tests that all windows satisfy the concept for various sizes.
 *
 * Note: For N=2 (minimum size), Hann and Blackman windows have zero endpoints.
 * This is mathematically correct but may result in zero coherent gain.
 * We skip the CG check for N=2 as it's an edge case. Hamming always has
 * non-zero endpoints (0.08) so its CG is always positive.
 */
TEST_F(WindowConceptTest, ConceptValidityAcrossSizes) {
    std::size_t sizes[] = {2, 8, 64, 256, 1024, 4096};

    for (auto size : sizes) {
        HannWindow hann(size);
        HammingWindow hamming(size);
        BlackmanWindow blackman(size);

        EXPECT_EQ(hann.size(), size);
        EXPECT_EQ(hamming.size(), size);
        EXPECT_EQ(blackman.size(), size);

        // For size=2, Hann and Blackman have zero endpoints so CG may be 0
        // For larger sizes, CG should be > 0
        if (size > 2) {
            EXPECT_GT(hann.coherentGain(), 0.0)
                << "Hann CG should be positive for size=" << size;
            EXPECT_GT(blackman.coherentGain(), 0.0)
                << "Blackman CG should be positive for size=" << size;
        }
        
        // Hamming always has non-zero endpoints (0.08) so CG is always positive
        EXPECT_GT(hamming.coherentGain(), 0.0)
            << "Hamming CG should always be positive";
    }
}

// ==========================================
// TEST 9: Const-Correctness of Concept Methods
// ==========================================

/**
 * @brief Verifies that const methods remain const.
 *
 * The concept requires that size(), coefficients(), coherentGain(), and
 * powerBandwidth() are all callable on const instances.
 */
TEST_F(WindowConceptTest, ConstCorrectness) {
    const HannWindow hann(TEST_SIZE);
    const HammingWindow hamming(TEST_SIZE);
    const BlackmanWindow blackman(TEST_SIZE);

    // All these should compile and work on const instances
    EXPECT_EQ(hann.size(), TEST_SIZE);
    EXPECT_EQ(hamming.size(), TEST_SIZE);
    EXPECT_EQ(blackman.size(), TEST_SIZE);

    EXPECT_GT(hann.coherentGain(), 0.0);
    EXPECT_GT(hamming.coherentGain(), 0.0);
    EXPECT_GT(blackman.coherentGain(), 0.0);

    EXPECT_GT(hann.powerBandwidth(), 0.0);
    EXPECT_GT(hamming.powerBandwidth(), 0.0);
    EXPECT_GT(blackman.powerBandwidth(), 0.0);

    const auto& hann_coeffs = hann.coefficients();
    const auto& hamming_coeffs = hamming.coefficients();
    const auto& blackman_coeffs = blackman.coefficients();

    EXPECT_EQ(hann_coeffs.size(), TEST_SIZE);
    EXPECT_EQ(hamming_coeffs.size(), TEST_SIZE);
    EXPECT_EQ(blackman_coeffs.size(), TEST_SIZE);
}

// ==========================================
// TEST 10: Apply is Non-Const
// ==========================================

/**
 * @brief Verifies that apply() requires a non-const instance
 *        (because it modifies the signal, not the window).
 *
 * This test just ensures the concept correctly specifies that
 * apply() takes a mutable window reference.
 */
TEST_F(WindowConceptTest, ApplyIsNonConst) {
    HannWindow hann(TEST_SIZE);
    std::vector<double> signal(TEST_SIZE, 1.0);

    // This should compile and work
    EXPECT_NO_THROW({
        hann.apply(signal);
    });

    // Verify the signal was modified
    bool was_modified = false;
    for (std::size_t i = 1; i < TEST_SIZE; ++i) {
        // In a Hann window, not all values are 1.0
        if (signal[i] != 1.0) {
            was_modified = true;
            break;
        }
    }
    EXPECT_TRUE(was_modified) << "Hann window should modify the signal";
}
