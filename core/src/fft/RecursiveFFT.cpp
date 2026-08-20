#include "fft/RecursiveFFT.hpp"

#include <cmath>
#include <numbers>

namespace stft {

void RecursiveFFT::forward_impl(std::vector<std::complex<double>>& data) {
    if (data.empty()) return;
    fft_recursive(data, false);
}

void RecursiveFFT::inverse_impl(std::vector<std::complex<double>>& data) {
    if (data.empty()) return;

    fft_recursive(data, true);

    const double norm = 1.0 / static_cast<double>(data.size());
    for (auto& sample : data) {
        sample *= norm;
    }
}

void RecursiveFFT::fft_recursive(std::vector<std::complex<double>>& data,
                                 bool inverse) {
    const std::size_t n = data.size();

    // Base case: a single sample is already its own transform.
    if (n <= 1) return;

    // Split into even- and odd-indexed samples.
    std::vector<std::complex<double>> even(n / 2);
    std::vector<std::complex<double>> odd(n / 2);
    for (std::size_t i = 0; i < n / 2; ++i) {
        even[i] = data[2 * i];
        odd[i]  = data[2 * i + 1];
    }

    // Transform each half.
    fft_recursive(even, inverse);
    fft_recursive(odd, inverse);

    // Combine: data[k] = even[k] + w^k * odd[k], data[k+n/2] = even[k] - w^k * odd[k],
    // w = primitive n-th root of unity (sign flips between forward and inverse).
    const double angleSign = inverse ? 1.0 : -1.0;
    const double angle = angleSign * 2.0 * std::numbers::pi / static_cast<double>(n);
    const std::complex<double> wn(std::cos(angle), std::sin(angle));

    std::complex<double> w(1.0, 0.0);
    for (std::size_t k = 0; k < n / 2; ++k) {
        const std::complex<double> t = w * odd[k];
        data[k]         = even[k] + t;
        data[k + n / 2] = even[k] - t;
        w *= wn;
    }
}

} // namespace stft
