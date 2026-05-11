#include "fft/IterativeFFT.hpp"
#include <cmath>
#include <numbers>
#include <bit>

namespace amsc_stft {

/**
 * @brief Executes the forward iterative FFT.
 * @param data Input complex vector.
 */
void IterativeFFT::forward_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;

    bitReverse(data);
    butterflyPass(data, false);
}

/**
 * @brief Executes the inverse iterative FFT.
 * @param data Input complex vector.
 */
void IterativeFFT::inverse_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;

    bitReverse(data);
    butterflyPass(data, true);

    // Normalize inverse FFT output
    const double norm = 1.0 / static_cast<double>(data.size());

    for (auto& sample : data) {
        sample *= norm;
    }
}

/**
 * @brief Reorders the input vector using bit-reversal permutation.
 * @param data Input complex vector.
 */
void IterativeFFT::bitReverse(
    std::vector<std::complex<double>>& data
) const {

    const size_t n = data.size();

    // Computes log2(n) assuming n is a power of two
    const int log_n = std::countr_zero(n);

    for (size_t i = 0; i < n; ++i) {

        size_t j = 0;

        for (int bit = 0; bit < log_n; ++bit) {

            if ((i >> bit) & 1) {
                j |= (1ULL << (log_n - 1 - bit));
            }
        }

        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

/**
 * @brief Executes iterative butterfly stages.
 * @param data Input complex vector.
 * @param inverse Enables inverse FFT when true.
 */
void IterativeFFT::butterflyPass(
    std::vector<std::complex<double>>& data,
    bool inverse
) const {

    const size_t n = data.size();

    const double pi = std::numbers::pi;
    const double angle_sign = inverse ? 1.0 : -1.0;

    for (size_t len = 2; len <= n; len <<= 1) {

        const double angle =
            angle_sign * 2.0 * pi / static_cast<double>(len);

        const std::complex<double> wlen(
            std::cos(angle),
            std::sin(angle)
        );

        for (size_t i = 0; i < n; i += len) {

            std::complex<double> w(1.0, 0.0);

            for (size_t j = 0; j < len / 2; ++j) {

                std::complex<double> u = data[i + j];

                std::complex<double> v =
                    data[i + j + len / 2] * w;

                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;

                w *= wlen;
            }
        }
    }
}

} // namespace amsc_stft