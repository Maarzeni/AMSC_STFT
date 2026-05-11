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
 */
class IterativeFFT : public BaseFFT<IterativeFFT> {

    friend class BaseFFT<IterativeFFT>;

private:

    /**
     * @brief Forward FFT implementation.
     */
    void forward_impl(std::vector<std::complex<double>>& data);

    /**
     * @brief Inverse FFT implementation.
     */
    void inverse_impl(std::vector<std::complex<double>>& data);

    /**
     * @brief Reorders the input using bit-reversal permutation.
     */
    void bitReverse(std::vector<std::complex<double>>& data) const;

    /**
     * @brief Executes iterative butterfly stages.
     */
    void butterflyPass(
        std::vector<std::complex<double>>& data,
        bool inverse
    ) const;
};

} // namespace amsc_stft

#endif // ITERATIVE_FFT_HPP