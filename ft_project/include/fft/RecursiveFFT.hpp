#ifndef RECURSIVE_FFT_HPP
#define RECURSIVE_FFT_HPP

#include "BaseFFT.hpp"
#include <vector>
#include <complex>

namespace amsc_stft {

/**
 * @class RecursiveFFT
 * @brief Recursive Cooley-Tukey FFT implementation.
 */
class RecursiveFFT : public BaseFFT<RecursiveFFT> {

    // Allows BaseFFT to access private implementation methods
    friend class BaseFFT<RecursiveFFT>;

private:

    /**
     * @brief Recursive FFT implementation.
     * @param data Input data.
     * @param inverse Enables inverse FFT when true.
     */
    void fft_recursive(std::vector<std::complex<double>>& data, bool inverse);

    /**
     * @brief Forward FFT implementation.
     */
    void forward_impl(std::vector<std::complex<double>>& data);

    /**
     * @brief Inverse FFT implementation.
     */
    void inverse_impl(std::vector<std::complex<double>>& data);
};

} // namespace amsc_stft

#endif // RECURSIVE_FFT_HPP