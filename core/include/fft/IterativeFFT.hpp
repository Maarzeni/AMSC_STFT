/**
 * @file IterativeFFT.hpp
 * @brief In-place, non-recursive Cooley-Tukey FFT.
 */

#pragma once

#include "BaseFFT.hpp"
#include <complex>
#include <vector>

namespace stft {

/**
 * @brief Iterative, in-place Cooley-Tukey FFT.
 *
 * The same algorithm as RecursiveFFT, restructured to avoid recursion: the
 * input is first permuted into bit-reversed order, then combined bottom-up
 * in stages of doubling size (2, 4, 8, ... n), each stage overwriting `data`
 * directly. No recursive calls and no extra buffers, which is what makes it
 * faster than RecursiveFFT in practice.
 */
class IterativeFFT : public BaseFFT<IterativeFFT> {
    friend class BaseFFT<IterativeFFT>;

private:
    /// @brief BaseFFT hook: forward transform.
    void forward_impl(std::vector<std::complex<double>>& data);

    /// @brief BaseFFT hook: inverse transform, including the 1/n normalisation.
    void inverse_impl(std::vector<std::complex<double>>& data);

    /**
     * @brief Permutes `data` into bit-reversed order, in place.
     *
     * Element `i` moves to the position whose bits are `i`'s reversed —
     * the reordering the iterative butterfly stages below require. Its own
     * inverse: applying it twice restores the original order.
     */
    void bitReverse(std::vector<std::complex<double>>& data) const;

    /**
     * @brief Combines bit-reversed `data` in place, stage by stage.
     * @param data     Bit-reversed buffer to transform; its size must be a
     *                 power of two (checked by the caller).
     * @param inverse  Runs the inverse transform when true.
     */
    void butterflyPass(std::vector<std::complex<double>>& data,
                       bool inverse) const;
};

} // namespace stft
