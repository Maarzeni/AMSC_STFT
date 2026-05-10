#ifndef RECURSIVE_FFT_HPP
#define RECURSIVE_FFT_HPP

#include "BaseFFT.hpp"
#include <vector>
#include <complex>

namespace amsc_stft {

/**
 * @class RecursiveFFT
 * @brief Recursive Cooley-Tukey FFT implementation using divide-and-conquer.
 * 
 * Adapts the classic recursive algorithm to the CRTP BaseFFT interface.
 */
class RecursiveFFT : public BaseFFT<RecursiveFFT> {
    // Concediamo l'accesso alla classe base per chiamare i metodi _impl privati
    friend class BaseFFT<RecursiveFFT>;

private:
    /**
     * @brief Core recursive implementation (in-place).
     * @param data Data to transform.
     * @param inverse If true, compute inverse FFT.
     */
    void fft_recursive(std::vector<std::complex<double>>& data, bool inverse);

    /**
     * @brief Forward FFT implementation required by BaseFFT.
     * Power-of-two validation is already handled by BaseFFT.
     */
    void forward_impl(std::vector<std::complex<double>>& data);

    /**
     * @brief Inverse FFT implementation required by BaseFFT.
     * Power-of-two validation is already handled by BaseFFT.
     */
    void inverse_impl(std::vector<std::complex<double>>& data);
};

} // namespace amsc_stft

#endif // RECURSIVE_FFT_HPP