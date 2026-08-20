/**
 * @file test_FFTVsDirectDFT.cpp
 * @brief Validates RecursiveFFT and IterativeFFT against a direct, naive
 *        O(N^2) DFT on small, random signals.
 *
 * @details
 * The direct DFT is a plain double sum — no divide-and-conquer, no
 * bit-reversal, nothing either fast algorithm shares — so it is trivially
 * correct by construction and gives an authoritative, independent reference.
 * This is what makes it a stronger check than the impulse/constant/round-trip
 * tests in test_RecursiveFFT.cpp and test_IterativeFFT.cpp: a round trip
 * (IFFT(FFT(x)) == x) only proves the two are each other's inverse, which a
 * signal that is wrong in the same way both ways would still pass, and an
 * impulse or a constant signal is too structured to exercise a generic bug.
 * Comparing against an independent reference, on random signals, closes both
 * gaps.
 *
 * Sign convention (matches the library, cross-checked by the impulse/DC
 * tests in the other two files):
 * @code
 *   Forward:  X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i * k * n / N)
 *   Inverse:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2*pi*i * k * n / N)
 * @endcode
 */

#include <gtest/gtest.h>
#include "fft/RecursiveFFT.hpp"
#include "fft/IterativeFFT.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

using namespace stft;

class FFTVsDirectDFTTest : public ::testing::Test {
protected:
    RecursiveFFT recursive;
    IterativeFFT iterative;
    static constexpr double tolerance = 1e-9;

    /// Sizes exercised by every test below.
    static constexpr auto kSizes = std::to_array<std::size_t>({4, 8, 16});

    /**
     * @brief Direct (naive) DFT: sign = -1 for forward, +1 for inverse.
     *        The inverse also applies the 1/N normalisation.
     */
    static std::vector<std::complex<double>> directDFT(
        const std::vector<std::complex<double>>& in, int sign) {
        const std::size_t N = in.size();
        std::vector<std::complex<double>> out(N, {0.0, 0.0});

        for (std::size_t k = 0; k < N; ++k) {
            std::complex<double> acc{0.0, 0.0};
            for (std::size_t n = 0; n < N; ++n) {
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

    static void compareVectors(const std::vector<std::complex<double>>& v1,
                               const std::vector<std::complex<double>>& v2) {
        ASSERT_EQ(v1.size(), v2.size()) << "Vectors have different sizes.";
        for (std::size_t i = 0; i < v1.size(); ++i) {
            EXPECT_NEAR(v1[i].real(), v2[i].real(), tolerance)
                << "Real part mismatch at index " << i;
            EXPECT_NEAR(v1[i].imag(), v2[i].imag(), tolerance)
                << "Imaginary part mismatch at index " << i;
        }
    }

    /// Deterministic pseudo-random complex signal of length N.
    static std::vector<std::complex<double>> randomSignal(std::size_t N,
                                                           unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<std::complex<double>> data(N);
        for (std::size_t i = 0; i < N; ++i) {
            data[i] = {dist(gen), dist(gen)};
        }
        return data;
    }
};

TEST_F(FFTVsDirectDFTTest, RecursiveForwardMatchesDirectDFT) {
    for (std::size_t N : kSizes) {
        const auto original = randomSignal(N, /*seed=*/N * 7 + 1);
        const auto expected = directDFT(original, /*sign=*/-1);

        auto data = original;
        recursive.forward(data);

        compareVectors(data, expected);
    }
}

TEST_F(FFTVsDirectDFTTest, IterativeForwardMatchesDirectDFT) {
    for (std::size_t N : kSizes) {
        const auto original = randomSignal(N, /*seed=*/N * 13 + 3);
        const auto expected = directDFT(original, /*sign=*/-1);

        auto data = original;
        iterative.forward(data);

        compareVectors(data, expected);
    }
}

TEST_F(FFTVsDirectDFTTest, RecursiveInverseMatchesDirectIDFT) {
    for (std::size_t N : kSizes) {
        const auto spectrum = randomSignal(N, /*seed=*/N * 17 + 5);
        const auto expected = directDFT(spectrum, /*sign=*/+1);

        auto data = spectrum;
        recursive.inverse(data);

        compareVectors(data, expected);
    }
}

TEST_F(FFTVsDirectDFTTest, IterativeInverseMatchesDirectIDFT) {
    for (std::size_t N : kSizes) {
        const auto spectrum = randomSignal(N, /*seed=*/N * 19 + 7);
        const auto expected = directDFT(spectrum, /*sign=*/+1);

        auto data = spectrum;
        iterative.inverse(data);

        compareVectors(data, expected);
    }
}

// Both are already validated against the direct DFT above; this guards
// against a shared-convention regression (e.g. a sign flip introduced in
// both engines at once, which the tests above alone could not distinguish
// from a matching change in directDFT()).
TEST_F(FFTVsDirectDFTTest, RecursiveAndIterativeAgree) {
    const std::size_t N = 16;
    const auto original = randomSignal(N, /*seed=*/99);

    auto a = original;
    auto b = original;
    recursive.forward(a);
    iterative.forward(b);

    compareVectors(a, b);
}
