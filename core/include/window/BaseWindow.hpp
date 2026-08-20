/**
 * @file BaseWindow.hpp
 * @brief CRTP base class providing shared storage and DSP utilities for
 *        every window function.
 *
 * @details
 * @code
 *   BaseWindow<HannWindow>   ← storage, apply(), coherentGain(), ...
 *        ↑
 *   HannWindow               ← only the coefficient formula
 * @endcode
 *
 * `BaseWindow<Derived>` uses the Curiously Recurring Template Pattern:
 * its constructor calls `static_cast<Derived*>(this)->generate(N)` to fill
 * the coefficient table. The compiler resolves that call at compile time —
 * no virtual table, dispatch can be inlined. Every derived class only needs
 * to implement `generate()`; `apply()`, `operator()`, `size()`,
 * `coefficients()`, `coherentGain()` and `powerBandwidth()` come from here.
 *
 * There is no runtime polymorphism in this hierarchy at all: code that picks
 * a window type from a string (see examples/main.cpp) does so by choosing
 * which template instantiation to call, not by dispatching through a common
 * base pointer.
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace stft {

/**
 * @brief Compile-time contract every window function must satisfy.
 *
 * @details
 * | Requirement                        | Why it's needed                    |
 * |-------------------------------------|-------------------------------------|
 * | `W(std::size_t)`                    | constructible with a frame size    |
 * | `w.size() -> std::size_t`           | STFTAnalyzer needs the frame size  |
 * | `w.apply(vector<double>&)`          | windows a signal frame, in place   |
 * | `w(vector<double>&)`                | functor syntax, delegates to apply |
 * | `w.coefficients() -> const vector&` | needed for export / inspection     |
 * | `w.coherentGain() -> double`        | magnitude normalisation            |
 * | `w.powerBandwidth() -> double`      | equivalent noise bandwidth         |
 *
 * STFTAnalyzer is templated on this concept, so passing a type that does
 * not satisfy it fails at the point of instantiation with a readable
 * message, rather than deep inside the implementation.
 */
template<typename W>
concept WindowFunction =
    std::constructible_from<W, std::size_t> &&
    requires(W w, const W cw, std::vector<double>& sig) {
        { cw.size() } -> std::same_as<std::size_t>;
        { w.apply(sig) } -> std::same_as<void>;
        { w(sig) } -> std::same_as<void>;
        { cw.coefficients() } -> std::same_as<const std::vector<double>&>;
        { cw.coherentGain() } -> std::same_as<double>;
        { cw.powerBandwidth() } -> std::same_as<double>;
    };

/// @brief Compile-time WindowFunction check with a readable failure message.
#define ASSERT_WINDOW_FUNCTION(Type)                                       \
    static_assert(                                                         \
        ::stft::WindowFunction<Type>,                                      \
        #Type " does not satisfy the WindowFunction concept. "             \
        "Check that it has: size(), apply(), coefficients(), "             \
        "coherentGain(), powerBandwidth()."                                \
    )

/**
 * @brief CRTP base providing precomputed storage and shared DSP utilities.
 * @tparam Derived  The concrete window class (e.g. HannWindow), which must
 *                  implement `std::vector<double> generate(std::size_t N) const`.
 */
template<typename Derived>
class BaseWindow {
    // A `requires` clause on the class template head cannot reference
    // BaseWindow itself without a circular dependency, so this guard against
    // BaseWindow<BaseWindow<...>> is a static_assert instead.
    static_assert(!std::same_as<Derived, BaseWindow>,
                 "Do not instantiate BaseWindow directly. "
                 "Use HannWindow, HammingWindow, or BlackmanWindow.");

public:
    /**
     * @brief Constructs the window and precomputes all N coefficients.
     * @param N  Frame size (number of coefficients). Must be >= 2.
     * @throws std::invalid_argument if N < 2.
     * @throws std::logic_error if `Derived::generate()` returns the wrong
     *         number of values.
     */
    explicit BaseWindow(std::size_t N) : N_(N) {
        if (N_ < 2) {
            throw std::invalid_argument(
                "Window size must be >= 2, got " + std::to_string(N_) + ".");
        }

        coeffs_ = static_cast<Derived*>(this)->generate(N_);

        if (coeffs_.size() != N_) {
            throw std::logic_error(
                "generate() returned " + std::to_string(coeffs_.size()) +
                " values, expected " + std::to_string(N_) + ".");
        }
    }

    // Value semantics throughout; no virtual base, so the destructor does
    // not need to be virtual either.
    BaseWindow(const BaseWindow&)            = default;
    BaseWindow& operator=(const BaseWindow&) = default;
    BaseWindow(BaseWindow&&)                 = default;
    BaseWindow& operator=(BaseWindow&&)      = default;
    ~BaseWindow()                            = default;

    /**
     * @brief Applies the window to a signal frame in place: `signal[i] *= coeffs_[i]`.
     * @param signal  Frame buffer, modified in place.
     * @throws std::invalid_argument if `signal.size() != size()`.
     */
    void apply(std::vector<double>& signal) const {
        if (signal.size() != N_) {
            throw std::invalid_argument(
                "Frame size (" + std::to_string(signal.size()) +
                ") does not match window size (" + std::to_string(N_) + ").");
        }
        for (std::size_t i = 0; i < N_; ++i) {
            signal[i] *= coeffs_[i];
        }
    }

    /// @brief Functor syntax: `window(frame)` is equivalent to `window.apply(frame)`.
    void operator()(std::vector<double>& signal) const { apply(signal); }

    /// @return N, the value passed to the constructor.
    [[nodiscard]] std::size_t size() const noexcept { return N_; }

    /// @return Read-only, zero-copy access to the precomputed coefficients.
    [[nodiscard]] const std::vector<double>& coefficients() const noexcept {
        return coeffs_;
    }

    /**
     * @brief Coherent gain: CG = (1/N) * sum(w[n]).
     *
     * How much the window attenuates a constant (DC) signal; dividing FFT
     * magnitudes by CG recovers physical amplitude units.
     * @return Coherent gain in (0, 1].
     */
    [[nodiscard]] double coherentGain() const noexcept {
        const double sum = std::accumulate(coeffs_.begin(), coeffs_.end(), 0.0);
        return sum / static_cast<double>(N_);
    }

    /**
     * @brief Equivalent noise bandwidth: ENBW = N * sum(w[n]^2) / (sum w[n])^2.
     * @return ENBW in frequency bins (dimensionless).
     */
    [[nodiscard]] double powerBandwidth() const noexcept {
        const double sumSq = std::inner_product(coeffs_.begin(), coeffs_.end(),
                                                 coeffs_.begin(), 0.0);
        const double cg = coherentGain();
        return (cg == 0.0) ? 0.0 : sumSq / (static_cast<double>(N_) * cg * cg);
    }

protected:
    std::size_t         N_;       ///< Window length (number of coefficients).
    std::vector<double> coeffs_;  ///< Precomputed coefficients, w[0..N-1].
};

} // namespace stft
