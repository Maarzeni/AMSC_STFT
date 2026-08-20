/**
 * @file BaseFFT.hpp
 * @brief Shared contract and dispatch logic for every FFT engine.
 */

#pragma once

#include <complex>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace stft {

/**
 * @brief Compile-time contract every FFT engine must satisfy.
 *
 * Requires a `forward()` and an `inverse()` method, each transforming the
 * buffer in place and returning nothing. STFTAnalyzer and the benchmark
 * suite are templated on this concept, so passing a type that does not
 * satisfy it fails at the point of instantiation with a readable message,
 * rather than deep inside the implementation.
 */
template <typename T>
concept IsFFT = requires(T a, std::vector<std::complex<double>>& data) {
    { a.forward(data) } -> std::same_as<void>;
    { a.inverse(data) } -> std::same_as<void>;
};

/**
 * @brief CRTP base class shared by every FFT engine (RecursiveFFT,
 *        IterativeFFT, ParallelFFT).
 *
 * Static (compile-time) polymorphism: BaseFFT<Derived> dispatches to
 * `Derived::forward_impl()` / `Derived::inverse_impl()` through a
 * `static_cast`, so there is no virtual table and no runtime dispatch cost.
 * What it factors out is what every engine needs regardless of algorithm:
 * the power-of-two size check, done once here instead of once per engine.
 *
 * @tparam Derived The concrete FFT engine inheriting from this class. Must
 *                  declare `friend class BaseFFT<Derived>;` and implement
 *                  `forward_impl()` / `inverse_impl()`.
 */
template <typename Derived>
class BaseFFT {
public:
    /**
     * @brief Forward transform, in place.
     * @param data  Buffer to transform; its size must be a power of two.
     * @throws std::invalid_argument if `data.size()` is not a power of two.
     */
    void forward(std::vector<std::complex<double>>& data) {
        checkPowerOfTwo(data.size());
        static_cast<Derived*>(this)->forward_impl(data);
    }

    /**
     * @brief Inverse transform, in place.
     * @param data  Buffer to transform; its size must be a power of two.
     * @throws std::invalid_argument if `data.size()` is not a power of two.
     */
    void inverse(std::vector<std::complex<double>>& data) {
        checkPowerOfTwo(data.size());
        static_cast<Derived*>(this)->inverse_impl(data);
    }

protected:
    BaseFFT() = default;

private:
    /// @throws std::invalid_argument if `n` is not a power of two.
    void checkPowerOfTwo(std::size_t n) const {
        if (n == 0 || (n & (n - 1)) != 0) {
            throw std::invalid_argument(
                "FFT Error: The input size must be a power of 2.");
        }
    }
};

} // namespace stft
