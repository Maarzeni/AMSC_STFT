#include <gtest/gtest.h>
#include "fft/BaseFFT.hpp"
#include <complex>
#include <vector>

using namespace stft;

// A type with the right forward()/inverse() signatures satisfies IsFFT...
class ValidMockFFT {
public:
    void forward([[maybe_unused]] std::vector<std::complex<double>>& data) {}
    void inverse([[maybe_unused]] std::vector<std::complex<double>>& data) {}
};

// ...one missing inverse() does not.
class InvalidMockFFT {
public:
    void forward([[maybe_unused]] std::vector<std::complex<double>>& data) {}
};

static_assert(IsFFT<ValidMockFFT>,
             "ValidMockFFT has the required signatures and should satisfy IsFFT.");
static_assert(!IsFFT<InvalidMockFFT>,
             "InvalidMockFFT is missing inverse() and should not satisfy IsFFT.");

TEST(BaseFFTTest, ConceptChecksCompile) {
    // Nothing to run: if this translation unit compiled, the static_asserts
    // above already proved that IsFFT accepts and rejects the right types.
    SUCCEED();
}
