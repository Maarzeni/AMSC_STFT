#ifndef BASE_FFT_HPP
#define BASE_FFT_HPP

#include <vector>
#include <complex>
#include <stdexcept>
#include <concepts>

namespace amsc_stft {

/**
 * @brief C++20 concept for FFT implementations.
 *
 * Requires forward() and inverse() methods
 * operating on complex vectors.
 */
template <typename T>
concept IsFFT = requires(T a, std::vector<std::complex<double>>& data) {
    { a.forward(data) } -> std::same_as<void>;
    { a.inverse(data) } -> std::same_as<void>;
};

/**
 * @class BaseFFT
 * @brief CRTP base class for FFT implementations.
 *
 * Provides common validation and dispatch logic.
 */
template <typename Derived>
class BaseFFT {
public:

    /**
     * @brief Executes forward FFT.
     * @param data Input complex vector.
     */
    void forward(std::vector<std::complex<double>>& data) {
        checkPowerOfTwo(data.size());
        static_cast<Derived*>(this)->forward_impl(data);
    }

    /**
     * @brief Executes inverse FFT.
     * @param data Input complex vector.
     */
    void inverse(std::vector<std::complex<double>>& data) {
        checkPowerOfTwo(data.size());
        static_cast<Derived*>(this)->inverse_impl(data);
    }

protected:

    BaseFFT() = default;

    /**
     * @brief Checks if the input size is a power of two.
     * @param n Input size.
     */
    void checkPowerOfTwo(size_t n) const {
        if (n == 0 || (n & (n - 1)) != 0) {
            throw std::invalid_argument(
                "FFT Error: The input size must be a power of 2."
            );
        }
    }
};

} // namespace amsc_stft

#endif // BASE_FFT_HPP