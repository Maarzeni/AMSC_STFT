#include "fft/IterativeFFT.hpp"

#include <bit>
#include <cmath>
#include <numbers>

namespace stft {

void IterativeFFT::forward_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;
    bitReverse(data);
    butterflyPass(data, false);
}

void IterativeFFT::inverse_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;

    bitReverse(data);
    butterflyPass(data, true);

    const double norm = 1.0 / static_cast<double>(data.size());
    for (auto& sample : data) {
        sample *= norm;
    }
}

void IterativeFFT::bitReverse(std::vector<std::complex<double>>& data) const {
    const std::size_t n = data.size();
    const int logN = std::countr_zero(n);  // log2(n): n is a power of two

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = 0;
        for (int bit = 0; bit < logN; ++bit) {
            if ((i >> bit) & 1) {
                j |= (1ULL << (logN - 1 - bit));
            }
        }
        // (i < j) rather than (i != j): without it, every pair would be
        // swapped twice — once as (i, j), once as (j, i) — undoing itself.
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

void IterativeFFT::butterflyPass(std::vector<std::complex<double>>& data,
                                 bool inverse) const {
    const std::size_t n = data.size();
    const double angleSign = inverse ? 1.0 : -1.0;

    // One stage per doubling of the block size: 2, 4, 8, ... n. Stage `len`
    // combines pairs of already-transformed blocks of size len/2, computed
    // by the previous stage, into transformed blocks of size len.
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = angleSign * 2.0 * std::numbers::pi / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));

        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const std::complex<double> u = data[i + j];
                const std::complex<double> v = data[i + j + len / 2] * w;
                data[i + j]         = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

} // namespace stft
