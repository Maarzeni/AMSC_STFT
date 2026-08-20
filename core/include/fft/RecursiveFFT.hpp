/**
 * @file RecursiveFFT.hpp
 * @brief Textbook recursive Cooley-Tukey FFT.
 */

#pragma once

#include "BaseFFT.hpp"
#include <complex>
#include <vector>

namespace stft {

/**
 * @brief Recursive, divide-and-conquer Cooley-Tukey FFT.
 *
 * Splits the input into even- and odd-indexed samples, transforms each half
 * recursively, and combines the results. The direct implementation of the
 * textbook algorithm: clear to read, but each recursive call allocates two
 * new vectors, which makes it the slowest of the three engines in this
 * project — IterativeFFT and ParallelFFT exist because of that cost.
 */
class RecursiveFFT : public BaseFFT<RecursiveFFT> {
    friend class BaseFFT<RecursiveFFT>;

private:
    /**
     * @brief Recursively splits, transforms and combines `data`.
     * @param data     Buffer to transform in place; its size must be a
     *                 power of two (checked by the caller).
     * @param inverse  Runs the inverse transform when true.
     */
    void fft_recursive(std::vector<std::complex<double>>& data, bool inverse);

    /// @brief BaseFFT hook: forward transform.
    void forward_impl(std::vector<std::complex<double>>& data);

    /// @brief BaseFFT hook: inverse transform, including the 1/n normalisation.
    void inverse_impl(std::vector<std::complex<double>>& data);
};

} // namespace stft
