/**
 * @file ParallelFFT.hpp
 * @brief Multi-threaded, in-place Cooley-Tukey FFT (OpenMP).
 */

#pragma once

#include "fft/BaseFFT.hpp"
#include <complex>
#include <cstddef>
#include <vector>

namespace stft {

/**
 * @brief Iterative Cooley-Tukey FFT, parallelised across OpenMP threads.
 *
 * The same algorithm as IterativeFFT — bit-reversal, then butterfly stages
 * of doubling size — with both loops split across `num_threads_` OpenMP
 * threads. The thread count is fixed once, at construction, and every
 * parallel region names it explicitly (`num_threads(num_threads_)`) rather
 * than reading the ambient OpenMP thread count, so an engine's parallelism
 * cannot change out from under it if something elsewhere in the process
 * later calls `omp_set_num_threads()`.
 */
class ParallelFFT : public BaseFFT<ParallelFFT> {
    friend class BaseFFT<ParallelFFT>;

public:
    /**
     * @brief Constructs the engine with a fixed thread count.
     * @param num_threads  Threads to use for every transform this instance
     *                     runs. 0 auto-detects the machine's thread count
     *                     at construction time (`omp_get_max_threads()`).
     */
    explicit ParallelFFT(std::size_t num_threads = 0);

private:
    std::size_t num_threads_;

    /// @brief BaseFFT hook: forward transform.
    void forward_impl(std::vector<std::complex<double>>& data);

    /// @brief BaseFFT hook: inverse transform, including the 1/n normalisation.
    void inverse_impl(std::vector<std::complex<double>>& data);

    /// @brief Permutes `data` into bit-reversed order, in parallel.
    void bitReverse(std::vector<std::complex<double>>& data) const;

    /**
     * @brief Combines bit-reversed `data` in place, stage by stage; each
     *        stage's independent blocks are split across threads.
     * @param data     Bit-reversed buffer to transform; its size must be a
     *                 power of two (checked by the caller).
     * @param inverse  Runs the inverse transform when true.
     */
    void butterflyPass(std::vector<std::complex<double>>& data,
                       bool inverse) const;
};

} // namespace stft
