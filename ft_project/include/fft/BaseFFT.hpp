#ifndef BASE_FFT_HPP
#define BASE_FFT_HPP

#include <vector>
#include <complex>
#include <stdexcept>
#include <concepts>

namespace amsc_stft {

// ==========================================
// C++20 CONCEPT (Interface Contract)
// ==========================================
// A type T satisfies "IsFFT" if it possesses the public methods "forward" and "inverse"
// that take a vector of complex numbers and return nothing (void).
template <typename T>
concept IsFFT = requires(T a, std::vector<std::complex<double>>& data) {
    { a.forward(data) } -> std::same_as<void>;
    { a.inverse(data) } -> std::same_as<void>;
};

// CRTP pattern: BaseFFT takes 'Derived' as a template parameter
template <typename Derived>
class BaseFFT {
public:
    // Public method that the user (or STFTAnalyzer) will call
    void forward(std::vector<std::complex<double>>& data) {
        // 1. Perform safety checks valid for ALL FFT implementations
        checkPowerOfTwo(data.size());

        // 2. The CRTP magic: force the compiler to treat 'this'
        // as a pointer to the Derived class, and call its implementation.
        // This happens without any runtime overhead!
        static_cast<Derived*>(this)->forward_impl(data);
    }

    // Method for the Inverse Transform (IFFT)
    void inverse(std::vector<std::complex<double>>& data) {
        checkPowerOfTwo(data.size());
        static_cast<Derived*>(this)->inverse_impl(data);
    }

protected:
    // Prevents anyone from instantiating BaseFFT directly
    BaseFFT() = default;

    // Utility method available to all derived classes.
    // The Cooley-Tukey Radix-2 algorithm requires that N be a power of 2.
    void checkPowerOfTwo(size_t n) const {
        if (n == 0 || (n & (n - 1)) != 0) {
            throw std::invalid_argument("FFT Error: The input size must be a power of 2.");
        }
    }
};

} // namespace amsc_stft

#endif // BASE_FFT_HPP