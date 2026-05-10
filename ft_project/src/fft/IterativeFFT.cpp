#include "fft/IterativeFFT.hpp"
#include <cmath>
#include <numbers>
#include <bit>

namespace amsc_stft {

void IterativeFFT::forward_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;

    bitReverse(data);
    butterflyPass(data, false);
}

void IterativeFFT::inverse_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;

    bitReverse(data);
    butterflyPass(data, true);

    // Normalizzazione IFFT
    const double norm = 1.0 / static_cast<double>(data.size());
    for (auto& sample : data) {
        sample *= norm;
    }
}

void IterativeFFT::bitReverse(std::vector<std::complex<double>>& data) const {
    const size_t n = data.size();
    // Usiamo std::countr_zero per trovare il log2(n) in modo efficiente (C++20)
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

void IterativeFFT::butterflyPass(std::vector<std::complex<double>>& data, bool inverse) const {
    const size_t n = data.size();
    const double pi = std::numbers::pi;
    const double angle_sign = inverse ? 1.0 : -1.0;

    for (size_t len = 2; len <= n; len <<= 1) {
        const double angle = angle_sign * 2.0 * pi / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));

        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<double> u = data[i + j];
                std::complex<double> v = data[i + j + len / 2] * w;
                
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                
                w *= wlen;
            }
        }
    }
}

} // namespace amsc_stft