#include "fft/ParallelFFT.hpp"
#include <cmath>
#include <numbers>
#include <bit>
#include <omp.h>

namespace stft {

/**
 * @brief Constructs the parallel FFT engine and configures OpenMP.
 */
ParallelFFT::ParallelFFT(size_t num_threads) {
    // If 0, ask OpenMP for the maximum available hardware threads
    num_threads_ = (num_threads == 0) ? omp_get_max_threads() : num_threads;
    
    // Safety fallback just in case the environment is misconfigured
    if (num_threads_ == 0) {
        num_threads_ = 1; 
    }
}

/**
 * @brief Executes the forward parallel FFT.
 */
void ParallelFFT::forward_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;

    bitReverse(data);
    butterflyPass(data, false);
}

/**
 * @brief Executes the inverse parallel FFT.
 */
void ParallelFFT::inverse_impl(std::vector<std::complex<double>>& data) {
    if (data.size() <= 1) return;

    bitReverse(data);
    butterflyPass(data, true);

    // Parallelize the normalization step
    const double norm = 1.0 / static_cast<double>(data.size());
    const size_t n = data.size();

    #pragma omp parallel for num_threads(num_threads_) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        data[i] *= norm;
    }
}

/**
 * @brief Reorders the input vector using parallel bit-reversal permutation.
 */
void ParallelFFT::bitReverse(std::vector<std::complex<double>>& data) const {
    const size_t n = data.size();
    const int log_n = std::countr_zero(n);

    // This loop is safely parallelizable because each swap is independent.
    // Condition (i < j) guarantees elements are swapped exactly once.
    #pragma omp parallel for num_threads(num_threads_) schedule(static)
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
 * @brief Executes multi-threaded butterfly stages.
 */
void ParallelFFT::butterflyPass(std::vector<std::complex<double>>& data, bool inverse) const {
    const size_t n = data.size();
    const double pi = std::numbers::pi;
    const double angle_sign = inverse ? 1.0 : -1.0;

    // OUTER LOOP: Iterates over the stages (len = 2, 4, 8, 16...).
    // MUST be strictly sequential. Stage N cannot start before Stage N-1 finishes.
    for (size_t len = 2; len <= n; len <<= 1) {
        const double angle = angle_sign * 2.0 * pi / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));

        // INNER LOOP: Iterates over independent blocks of size 'len'.
        // This is perfectly parallelizable!
        #pragma omp parallel for num_threads(num_threads_) schedule(static)
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

} // namespace stft