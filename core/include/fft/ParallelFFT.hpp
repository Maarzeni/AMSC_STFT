/**
 * @file ParallelFFT.hpp
 * @brief Multi-threaded FFT implementation using OpenMP and CRTP.
 */

#ifndef PARALLEL_FFT_HPP
#define PARALLEL_FFT_HPP

#include "fft/BaseFFT.hpp"
#include <vector>
#include <complex>
#include <cstddef>

namespace stft {

/**
 * @class ParallelFFT
 * @brief Parallel Cooley-Tukey FFT implementation.
 *
 * Uses OpenMP multi-threading to accelerate bit-reversal and butterfly computation.
 */
class ParallelFFT : public BaseFFT<ParallelFFT> {

    friend class BaseFFT<ParallelFFT>;

public:

    /**
     * @brief Constructs the FFT with a given number of threads.
     * @param num_threads Number of threads to use (0 = auto-detect).
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
     * @brief Reorders the input using parallel bit-reversal permutation.
     */
    void bitReverse(std::vector<std::complex<double>>& data) const;

    /**
     * @brief Parallel butterfly computation kernel.
     * @param data Input complex vector.
     * @param inverse Enables inverse FFT when true.
     */
    void butterflyPass(
        std::vector<std::complex<double>>& data,
        bool inverse
    ) const;
};

} // namespace stft

#endif // PARALLEL_FFT_HPP