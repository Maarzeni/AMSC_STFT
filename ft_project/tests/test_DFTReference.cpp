#include <gtest/gtest.h>
#include "fft/RecursiveFFT.hpp"
#include "fft/IterativeFFT.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <numbers>
#include <random>

namespace stft {
namespace testing {

// Reference tests: validate the fast FFT implementations against a direct,
// naive O(N^2) DFT on small sizes. The direct DFT is trivially correct by
// construction and independent of the divide-and-conquer / iterative logic,
// so it provides an authoritative reference for the fast transforms.
//
// Convention (matching the library, verified by the impulse/constant tests):
//   Forward:  X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i * k * n / N)
//   Inverse:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2*pi*i * k * n / N)
class DFTReferenceTest : public ::testing::Test {
protected:
    RecursiveFFT recursive;
    IterativeFFT iterative;
    const double tolerance = 1e-9;

    // Direct (naive) DFT. sign = -1 for forward, +1 for inverse.
    // The inverse also applies the 1/N normalization.
    static std::vector<std::complex<double>> directDFT(
        const std::vector<std::complex<double>>& in, int sign) {
        const size_t N = in.size();
        std::vector<std::complex<double>> out(N, {0.0, 0.0});

        for (size_t k = 0; k < N; ++k) {
            std::complex<double> acc{0.0, 0.0};
            for (size_t n = 0; n < N; ++n) {
                const double angle =
                    sign * 2.0 * std::numbers::pi *
                    static_cast<double>(k) * static_cast<double>(n) /
                    static_cast<double>(N);
                acc += in[n] * std::complex<double>(std::cos(angle),
                                                    std::sin(angle));
            }
            out[k] = (sign > 0) ? acc / static_cast<double>(N) : acc;
        }
        return out;
    }

    void compareVectors(const std::vector<std::complex<double>>& v1,
                        const std::vector<std::complex<double>>& v2) {
        ASSERT_EQ(v1.size(), v2.size()) << "Vectors have different sizes.";
        for (size_t i = 0; i < v1.size(); ++i) {
            EXPECT_NEAR(v1[i].real(), v2[i].real(), tolerance)
                << "Real part mismatch at index " << i;
            EXPECT_NEAR(v1[i].imag(), v2[i].imag(), tolerance)
                << "Imaginary part mismatch at index " << i;
        }
    }

    // Deterministic pseudo-random complex signal of length N.
    static std::vector<std::complex<double>> randomSignal(size_t N,
                                                          unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<std::complex<double>> data(N);
        for (size_t i = 0; i < N; ++i) {
            data[i] = {dist(gen), dist(gen)};
        }
        return data;
    }
};

// Forward RecursiveFFT must match the direct DFT for N = 4, 8, 16.
TEST_F(DFTReferenceTest, RecursiveForwardMatchesDirectDFT) {
    for (size_t N : {size_t{4}, size_t{8}, size_t{16}}) {
        const auto original = randomSignal(N, /*seed=*/N * 7 + 1);
        const auto expected = directDFT(original, /*sign=*/-1);

        auto data = original;
        recursive.forward(data);

        compareVectors(data, expected);
    }
}

// Forward IterativeFFT must match the direct DFT for N = 4, 8, 16.
TEST_F(DFTReferenceTest, IterativeForwardMatchesDirectDFT) {
    for (size_t N : {size_t{4}, size_t{8}, size_t{16}}) {
        const auto original = randomSignal(N, /*seed=*/N * 13 + 3);
        const auto expected = directDFT(original, /*sign=*/-1);

        auto data = original;
        iterative.forward(data);

        compareVectors(data, expected);
    }
}

// Inverse RecursiveFFT must match the direct (normalized) inverse DFT.
TEST_F(DFTReferenceTest, RecursiveInverseMatchesDirectIDFT) {
    for (size_t N : {size_t{4}, size_t{8}, size_t{16}}) {
        const auto spectrum = randomSignal(N, /*seed=*/N * 17 + 5);
        const auto expected = directDFT(spectrum, /*sign=*/+1);

        auto data = spectrum;
        recursive.inverse(data);

        compareVectors(data, expected);
    }
}

// Cross-check the two fast implementations agree with each other on the
// forward transform (both are validated against the direct DFT above, so
// this guards against a shared-convention regression).
TEST_F(DFTReferenceTest, RecursiveAndIterativeAgree) {
    const size_t N = 16;
    const auto original = randomSignal(N, /*seed=*/99);

    auto a = original;
    auto b = original;
    recursive.forward(a);
    iterative.forward(b);

    compareVectors(a, b);
}

} // namespace testing
} // namespace stft
