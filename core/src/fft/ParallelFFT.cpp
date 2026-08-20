#include "fft/ParallelFFT.hpp"

#include <bit>
#include <cmath>
#include <numbers>
#include <omp.h>

namespace stft {

ParallelFFT::ParallelFFT(std::size_t num_threads) {
    // 0 asks OpenMP for the machine's thread count; a misconfigured
    // environment can still report 0, so fall back to single-threaded rather
    // than pass 0 on to `num_threads()` clauses, which reject it.
    num_threads_ = (num_threads == 0) ? static_cast<std::size_t>(omp_get_max_threads())
                                       : num_threads;
    if (num_threads_ == 0) {
        num_threads_ = 1;
    }
}

void ParallelFFT::forward_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;
    bitReverse(data);
    butterflyPass(data, false);
}

void ParallelFFT::inverse_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;

    bitReverse(data);
    butterflyPass(data, true);

    const std::size_t n = data.size();
    const double norm = 1.0 / static_cast<double>(n);

    #pragma omp parallel for num_threads(num_threads_) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        data[i] *= norm;
    }
}

void ParallelFFT::bitReverse(std::vector<std::complex<double>>& data) const {
    const std::size_t n = data.size();
    const int logN = std::countr_zero(n);  // log2(n): n is a power of two

    // Each swap only touches the pair (i, j) it owns, so the iterations are
    // independent and safe to split across threads. (i < j) rather than
    // (i != j): without it, every pair would be swapped twice — once as
    // (i, j), once as (j, i) — undoing itself.
    #pragma omp parallel for num_threads(num_threads_) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = 0;
        for (int bit = 0; bit < logN; ++bit) {
            if ((i >> bit) & 1) {
                j |= (1ULL << (logN - 1 - bit));
            }
        }
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

void ParallelFFT::butterflyPass(std::vector<std::complex<double>>& data,
                                bool inverse) const {
    const std::size_t n = data.size();
    const double angleSign = inverse ? 1.0 : -1.0;

    // One stage per doubling of the block size: 2, 4, 8, ... n. Stages run
    // strictly in order — stage `len` needs the previous stage's output —
    // but within a stage the blocks are independent, so that loop is the one
    // split across threads.
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = angleSign * 2.0 * std::numbers::pi / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));

        #pragma omp parallel for num_threads(num_threads_) schedule(static)
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
