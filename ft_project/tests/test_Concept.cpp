#include <gtest/gtest.h>
#include "fft/BaseFFT.hpp"
#include <vector>
#include <complex>

using namespace amsc_stft;

// ==========================================
// MOCK CLASSES (For isolated testing)
// ==========================================

// 1. A perfectly valid mock class that satisfies the IsFFT concept.
// It doesn't do any math, it just has the right signatures.
class ValidMockFFT {
public:
    void forward([[maybe_unused]] std::vector<std::complex<double>>& data) {
        // Empty implementation
    }

    void inverse([[maybe_unused]] std::vector<std::complex<double>>& data) {
        // Empty implementation
    }
};

// 2. An invalid mock class that intentionally lacks the 'inverse' method.
class InvalidMockFFT {
public:
    void forward([[maybe_unused]] std::vector<std::complex<double>>& data) {
        // Empty implementation
    }
    // Intentionally missing inverse()
};

// ==========================================
// COMPILE-TIME TESTS
// ==========================================

// Verify that the Concept accepts the valid mock
static_assert(IsFFT<ValidMockFFT>, "ERROR: ValidMockFFT should satisfy the IsFFT concept!");

// Verify that the Concept rejects the invalid mock
static_assert(!IsFFT<InvalidMockFFT>, "ERROR: InvalidMockFFT should NOT satisfy the IsFFT concept!");

// ==========================================
// RUNTIME TEST
// ==========================================

TEST(ConceptTest, CompileTimeChecksPassed) {
    // If the program compiles, the static_asserts above passed successfully.
    // This runtime test simply reports success to GTest.
    SUCCEED() << "C++20 Concepts are working perfectly and are fully isolated!";
}