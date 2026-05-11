#include "fft/RecursiveFFT.hpp"
#include <cmath>
#include <numbers>

namespace amsc_stft {

/**
 * @brief Executes the forward recursive FFT.
 * @param data Input complex vector.
 */
void RecursiveFFT::forward_impl(std::vector<std::complex<double>>& data) {
    if (data.empty()) return;

    // Power-of-two validation is handled by BaseFFT
    fft_recursive(data, false);
}

/**
 * @brief Executes the inverse recursive FFT.
 * @param data Input complex vector.
 */
void RecursiveFFT::inverse_impl(std::vector<std::complex<double>>& data) {
    if (data.empty()) return;

    fft_recursive(data, true);

    // Normalization step
    const size_t n = data.size();
    const double norm = 1.0 / static_cast<double>(n);

    for (auto& sample : data) {
        sample *= norm;
    }
}

/**
 * @brief Recursive Cooley-Tukey FFT implementation.
 * @param data Input complex vector.
 * @param inverse Enables inverse FFT when true.
 */
void RecursiveFFT::fft_recursive(
    std::vector<std::complex<double>>& data,
    bool inverse
) {
    const size_t n = data.size();

    // Base case
    if (n <= 1) return;

    // Split input into even and odd indices
    std::vector<std::complex<double>> even(n / 2);
    std::vector<std::complex<double>> odd(n / 2);

    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = data[2 * i];
        odd[i] = data[2 * i + 1];
    }

    // Recursive FFT calls
    fft_recursive(even, inverse);
    fft_recursive(odd, inverse);

    // Combine step
    const double pi = std::numbers::pi;
    const double angle_sign = inverse ? 1.0 : -1.0;
    const double angle =
        angle_sign * 2.0 * pi / static_cast<double>(n);

    std::complex<double> w(1.0, 0.0);

    const std::complex<double> wn(
        std::cos(angle),
        std::sin(angle)
    );

    for (size_t k = 0; k < n / 2; ++k) {

        std::complex<double> t = w * odd[k];

        data[k] = even[k] + t;
        data[k + n / 2] = even[k] - t;

        w *= wn;
    }
}

} // namespace amsc_stft