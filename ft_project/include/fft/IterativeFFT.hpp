#ifndef ITERATIVE_FFT_HPP
#define ITERATIVE_FFT_HPP

#include "BaseFFT.hpp"
#include <vector>
#include <complex>

namespace amsc_stft {

/**
 * @class IterativeFFT
 * @brief Iterative Cooley-Tukey FFT implementation.
 * 
 * Uses bit-reversal permutation and in-place butterfly operations.
 * This implementation is more cache-friendly and performant than the recursive one.
 */
class IterativeFFT : public BaseFFT<IterativeFFT> {
    friend class BaseFFT<IterativeFFT>;

private:
    /**
     * @brief Forward FFT implementation for CRTP.
     */
    void forward_impl(std::vector<std::complex<double>>& data);

    /**
     * @brief Inverse FFT implementation for CRTP.
     */
    void inverse_impl(std::vector<std::complex<double>>& data);

    /**
     * @brief Reorders the vector using bit-reversal permutation.
     */
    void bit_reverse_reorder(std::vector<std::complex<double>>& data) const;

    /**
     * @brief Core butterfly computational kernel.
     */
    void compute_butterflies(std::vector<std::complex<double>>& data, bool inverse) const;
};

} // namespace amsc_stft

#endif // ITERATIVE_FFT_HPP