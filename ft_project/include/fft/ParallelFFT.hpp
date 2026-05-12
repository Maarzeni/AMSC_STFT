/**
 * @file ParallelFFT.hpp
 * @brief Multi-threaded FFT implementation using CRTP.
 */

#ifndef PARALLEL_FFT_HPP
#define PARALLEL_FFT_HPP

#include "fft/BaseFFT.hpp"
#include <vector>
#include <complex>
#include <thread>
#include <cstddef>

namespace stft {

/**
 * @class ParallelFFT
 * @brief Parallel Cooley-Tukey FFT implementation.
 *
 * Uses multi-threading to accelerate butterfly computation.
 */
class ParallelFFT : public BaseFFT<ParallelFFT> {

    friend class BaseFFT<ParallelFFT>;

public:

    /**
     * @brief Constructs the FFT with a given number of threads.
     * @param num_threads Number of threads to use (0 = hardware concurrency).
     */
    explicit ParallelFFT(size_t num_threads = 0);

protected:

    /**
     * @brief Forward FFT implementation.
     * @param data Input complex vector.
     */
    void forward_impl(std::vector<std::complex<double>>& data);

    /**
     * @brief Inverse FFT implementation.
     * @param data Input complex vector.
     */
    void inverse_impl(std::vector<std::complex<double>>& data);

private:

    size_t num_threads_;

    /**
     * @brief Computes bit-reversed index.
     */
    static size_t bitReverse(size_t i, size_t log_n);

    /**
     * @brief Computes integer log2 of n (n must be power of two).
     */
    static size_t log2int(size_t n);

    /**
     * @brief Parallel butterfly computation kernel.
     * @param data Input complex vector.
     * @param n Signal size.
     * @param inverse Enables inverse FFT when true.
     */
    void butterflyPassParallel(
        std::vector<std::complex<double>>& data,
        size_t n,
        bool inverse
    ) const;
};

} // namespace stft

#endif // PARALLEL_FFT_HPP