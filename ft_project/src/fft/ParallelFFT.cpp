/**
 * @file ParallelFFT.cpp
 * @brief Implementation of the multi-threaded FFT.
 */

#include "fft/ParallelFFT.hpp"
#include <cmath>
#include <algorithm>
#include <utility>

namespace amsc_stft {

ParallelFFT::ParallelFFT(size_t num_threads)
    : num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) 
{
    if (num_threads_ == 0) {
        num_threads_ = 1; // Fallback just in case hardware_concurrency() returns 0
    }
}

void ParallelFFT::forward_impl(std::vector<std::complex<double>>& data) {
    const size_t n = data.size();
    if (n == 0) return;

    // In-place Bit-reversal permutation
    const size_t log_n = log2int(n);
    for (size_t i = 0; i < n; ++i) {
        size_t j = bitReverse(i, log_n);
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    // Parallel butterfly operations
    butterflyPassParallel(data, n, /* inverse = */ false);
}

void ParallelFFT::inverse_impl(std::vector<std::complex<double>>& data) {
    const size_t n = data.size();
    if (n == 0) return;

    // In-place Bit-reversal permutation
    const size_t log_n = log2int(n);
    for (size_t i = 0; i < n; ++i) {
        size_t j = bitReverse(i, log_n);
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    // Parallel butterfly operations
    butterflyPassParallel(data, n, /* inverse = */ true);

    // Normalization
    const double norm = 1.0 / static_cast<double>(n);
    for (auto& sample : data) {
        sample *= norm;
    }
}

size_t ParallelFFT::bitReverse(size_t i, size_t log_n) {
    size_t j = 0;
    for (size_t bit = 0; bit < log_n; ++bit) {
        if (i & (size_t(1) << bit)) {
            j |= (size_t(1) << (log_n - 1 - bit));
        }
    }
    return j;
}

size_t ParallelFFT::log2int(size_t n) {
    size_t log_n = 0;
    while ((size_t(1) << log_n) < n) {
        ++log_n;
    }
    return log_n;
}

void ParallelFFT::butterflyPassParallel(std::vector<std::complex<double>>& data, size_t n, bool inverse) const {
    const double pi = std::acos(-1.0);
    const double angle_sign = inverse ? 1.0 : -1.0;

    for (size_t len = 2; len <= n; len <<= 1) {
        const double angle = angle_sign * 2.0 * pi / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));

        const size_t num_blocks = n / len;
        
        // Use parallel execution for large enough stages
        if (num_blocks >= num_threads_ && len >= 16) {
            std::vector<std::thread> threads;
            threads.reserve(num_threads_);
            
            const size_t blocks_per_thread = (num_blocks + num_threads_ - 1) / num_threads_;

            for (size_t t = 0; t < num_threads_; ++t) {
                const size_t start_block = t * blocks_per_thread;
                const size_t end_block = std::min(start_block + blocks_per_thread, num_blocks);

                threads.emplace_back([&, start_block, end_block, len, wlen]() {
                    for (size_t block = start_block; block < end_block; ++block) {
                        const size_t i = block * len;
                        std::complex<double> w(1.0, 0.0);
                        for (size_t j = 0; j < len / 2; ++j) {
                            std::complex<double> u = data[i + j];
                            std::complex<double> v = data[i + j + len / 2] * w;
                            data[i + j] = u + v;
                            data[i + j + len / 2] = u - v;
                            w *= wlen;
                        }
                    }
                });
            }

            for (auto& thread : threads) {
                thread.join();
            }
        } else {
            // Serial fallback for small stages (overhead of threading isn't worth it here)
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
}

} // namespace amsc_stft