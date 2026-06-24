/**
 * @file BaseWindow.hpp
 * @brief CRTP base class that provides shared storage and algorithms
 *        for all window functions.
 *
 * @details
 * ─── Architecture recap ───────────────────────────────────────────────────────
 *
 * We deliberately do NOT inherit from any virtual base class here.
 * Runtime polymorphism (choosing the window type at runtime) is achieved
 * via `std::variant` + `std::visit`, NOT via virtual dispatch.
 *
 * The inheritance chain is flat and simple:
 * @code
 *   BaseWindow<HannWindow>     ← only logic, no vtable
 *        ↑
 *   HannWindow                 ← only the math formula
 * @endcode
 *
 * ─── What CRTP does here ──────────────────────────────────────────────────────
 *
 * `BaseWindow<Derived>` uses the Curiously Recurring Template Pattern.
 * The trick: in the constructor, we cast `this` (which is a BaseWindow*)
 * to `Derived*` and call `Derived::generate()`. The compiler resolves
 * this call at compile-time — zero runtime overhead, inlining possible.
 *
 * @code
 *   // Simplified view of what the constructor does:
 *   coeffs_ = static_cast<Derived*>(this)->generate(N);
 *   //         ^^^^^^^^^^^^^^^^^^^^^^^^
 *   //         "I know this is really a HannWindow, trust me compiler"
 * @endcode
 *
 * ─── What every derived class gets for free ───────────────────────────────────
 *   - coeffs_         : the precomputed coefficient vector
 *   - apply()         : element-wise multiply (the hot-path method)
 *   - operator()      : functor call syntax — `window(frame)` (delegates to apply)
 *   - size()          : returns N
 *   - coefficients()  : read-only access to coeffs_
 *   - coherentGain()  : spectral normalization metric
 *   - powerBandwidth(): equivalent noise bandwidth
 *
 * Every derived class only needs to implement ONE method: generate().
 *
 * @author  AMSC_STFT Project
 * @version 2.0  (removed IWindow, variant-ready)
 */

#pragma once

#include <cstddef>    // std::size_t
#include <vector>     // std::vector
#include <stdexcept>  // std::invalid_argument, std::logic_error
#include <string>     // std::to_string
#include <numeric>    // std::accumulate, std::inner_product
#include <concepts>   // std::same_as, std::constructible_from

namespace stft {

// ─────────────────────────────────────────────────────────────────────────────
// C++20 CONCEPT: WindowFunction
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @concept WindowFunction
 * @brief Constrains a type W to be a valid window function for STFT use.
 *
 * @details
 * A type W satisfies `WindowFunction` if and only if:
 *
 * | Requirement                   | Why it's needed                          |
 * |-------------------------------|------------------------------------------|
 * | W(std::size_t)                | Must be constructible with a frame size  |
 * | w.size() → std::size_t        | STFTAnalyzer needs to know the frame size |
 * | w.apply(vector<double>&)      | Core operation: window a signal frame    |
 * | w(vector<double>&)            | Functor call syntax (delegates to apply) |
 * | w.coefficients() →            | Needed for normalization / export        |
 * |   const vector<double>&       |                                          |
 * | w.coherentGain() → double     | Needed for magnitude normalization       |
 * | w.powerBandwidth() → double   | Needed for energy analysis               |
 *
 * @tparam W  The type to check against this concept.
 *
 * @par Example usage in STFTAnalyzer:
 * @code
 * template<typename FFT, stft::WindowFunction Window>
 * class STFTAnalyzer {
 *     // Window is guaranteed to have apply(), size(), etc.
 * };
 * @endcode
 *
 * @par Example: verifying your own class satisfies the concept:
 * @code
 * static_assert(stft::WindowFunction<HannWindow>,
 *               "HannWindow must satisfy WindowFunction!");
 * @endcode
 */
template<typename W>
concept WindowFunction =
    std::constructible_from<W, std::size_t>
    &&
    requires(
        W w,
        const W cw,
        std::vector<double>& sig
    )
    {
        { cw.size() } -> std::same_as<std::size_t>;
        { w.apply(sig) } -> std::same_as<void>;
        { w(sig) } -> std::same_as<void>;
        { cw.coefficients() } -> std::same_as<const std::vector<double>&>;
        { cw.coherentGain() } -> std::same_as<double>;
        { cw.powerBandwidth() } -> std::same_as<double>;
    };

/**
 * @def ASSERT_WINDOW_FUNCTION
 * @brief Helper macro to verify at compile-time that a class satisfies WindowFunction.
 *
 * @details
 * Usage: `ASSERT_WINDOW_FUNCTION(HannWindow);` at the bottom of a window header.
 *
 * @code
 * ASSERT_WINDOW_FUNCTION(HannWindow);
 * // Equivalent to:
 * // static_assert(stft::WindowFunction<HannWindow>, "...");
 * @endcode
 */
#define ASSERT_WINDOW_FUNCTION(Type)                                   \
    static_assert(                                                     \
        ::stft::WindowFunction<Type>,                                  \
        #Type " does not satisfy the WindowFunction concept. "           \
        "Check that it has: size(), apply(), coefficients(), "           \
        "coherentGain(), powerBandwidth()."                              \
    )

// ─────────────────────────────────────────────────────────────────────────────
// CRTP BASE CLASS: BaseWindow
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @class BaseWindow
 * @brief CRTP base providing precomputed storage and shared DSP utilities.
 *
 * @tparam Derived  The concrete window class (e.g. HannWindow).
 *                  Must implement:
 *                  @code
 *                    std::vector<double> generate(std::size_t N) const;
 *                  @endcode
 */
template<typename Derived>
class BaseWindow {

    // ── Compile-time guard ───────────────────────────────────────────────────
    // Prevents BaseWindow<BaseWindow<...>> (accidental infinite CRTP cycle).
    // We use static_assert here because the template parameter `Derived`
    // self-references `BaseWindow` at the point where the template head is
    // parsed, so a `requires` constraint in the template declaration would
    // create a circular dependency that the compiler cannot resolve.
    static_assert(
        !std::same_as<Derived, BaseWindow>,
        "Do not instantiate BaseWindow directly. "
        "Use HannWindow, HammingWindow, or BlackmanWindow."
    );

public:

    // ── Constructor ──────────────────────────────────────────────────────────
    /**
     * @brief Constructs the window and precomputes all N coefficients.
     *
     * @details
     * Calls Derived::generate(N) via CRTP (compile-time dispatch).
     * After this constructor returns, coeffs_ is fully populated and
     * will never change — generate() is called exactly once.
     *
     * The keyword `explicit` prevents implicit conversions:
     * @code
     *   HannWindow w = 1024; // compile error: implicit conversion blocked
     *   HannWindow w(1024);  // correct
     *   HannWindow w{1024};  // also correct (brace initialization)
     * @endcode
     *
     * @param N  Frame size (number of coefficients). Must be >= 2.
     * @throws std::invalid_argument if N < 2.
     * @throws std::logic_error if generate() returns wrong number of values.
     */
    explicit BaseWindow(std::size_t N) : N_(N) {

        if (N_ < 2)
            throw std::invalid_argument(
                "Window size must be >= 2, got " + std::to_string(N_) + "."
            );

        // ── CRTP dispatch ────────────────────────────────────────────────────
        // At this point in the program, `this` genuinely IS a Derived object
        // (you can only construct HannWindow, never BaseWindow directly).
        // So the cast is always safe.
        //
        // The compiler resolves this to Derived::generate() at compile-time.
        // No vtable lookup. The function can even be inlined.
        coeffs_ = static_cast<Derived*>(this)->generate(N_);

        if (coeffs_.size() != N_) {
            throw std::logic_error(
                "generate() returned " + std::to_string(coeffs_.size()) +
                " values, expected " + std::to_string(N_) + "."
            );
        }
    }

    // ── Rule of Five ─────────────────────────────────────────────────────────
    /**
     * @details
     * We explicitly declare all five special members and ask the compiler
     * to auto-generate them with `= default`.
     *
     * - Copy constructor/assignment : deep-copies the coefficient vector.
     * - Move constructor/assignment : transfers the internal buffer in O(1).
     * - Destructor                  : releases the vector's memory.
     *
     * Since there is no virtual base, the destructor does NOT need to be
     * virtual — this removes vtable overhead intentionally.
     */
    BaseWindow(const BaseWindow&)            = default;
    BaseWindow& operator=(const BaseWindow&) = default;
    BaseWindow(BaseWindow&&)                 = default;
    BaseWindow& operator=(BaseWindow&&)      = default;
    ~BaseWindow()                            = default;

    // ── Core DSP method ──────────────────────────────────────────────────────
    /**
     * @brief Applies the window to a signal frame in-place.
     *
     * @details
     * Performs element-wise multiplication: signal[i] *= coeffs_[i]
     *
     * "In-place" means we modify the input vector directly instead of
     * allocating a new one. This is critical for STFT performance: this
     * method is called thousands of times (once per frame), so every
     * unnecessary allocation wastes real time.
     *
     * The `const` at the end of the signature means: "this method does not
     * modify the window object itself". It only modifies the signal.
     *
     * @param signal  Reference to the frame buffer. Modified in place.
     * @throws std::invalid_argument if signal.size() != size().
     */
    void apply(std::vector<double>& signal) const {
        if (signal.size() != N_) {
            throw std::invalid_argument(
                "Frame size (" + std::to_string(signal.size()) +
                ") does not match window size (" + std::to_string(N_) + ")."
            );
        }
        for (std::size_t i = 0; i < N_; ++i)
            signal[i] *= coeffs_[i];
    }

    // ── Functor call operator ─────────────────────────────────────────────────
    /**
     * @brief Functor call syntax: windows a frame in place.
     *
     * @details
     * Makes every window a callable object (functor). Equivalent to apply(),
     * but allows a window to be invoked as `window(frame)` and used
     * interchangeably with free functions and lambdas in generic code.
     * Delegates to apply() so the windowing logic lives in exactly one place.
     *
     * Example:
     * @code
     *   HannWindow w(1024);
     *   w(frame);          // functor syntax  — same as w.apply(frame)
     *   w.apply(frame);    // explicit syntax  — still works unchanged
     * @endcode
     *
     * @param signal Frame buffer, modified in place.
     * @throws std::invalid_argument if signal.size() != size().
     */
    void operator()(std::vector<double>& signal) const { apply(signal); }

    // ── Accessors ────────────────────────────────────────────────────────────

    /**
     * @brief Returns the number of coefficients.
     *
     * @details
     * [[nodiscard]] : compiler warns if you call this and discard the result.
     * noexcept      : promises no exception is thrown; enables move
     *                 optimizations in standard containers.
     *
     * @return N (the value passed to the constructor).
     */
    [[nodiscard]] std::size_t size() const noexcept { return N_; }

    /**
     * @brief Returns a read-only reference to the precomputed coefficients.
     *
     * @details
     * Returning `const std::vector<double>&` instead of `std::vector<double>`:
     *   - The `&` means reference — no copy of the vector is made.
     *   - The `const` means the caller cannot modify the returned data.
     *
     * Together: zero-copy, read-only access.
     *
     * @return Const reference to the internal coefficient vector.
     */
    [[nodiscard]] const std::vector<double>& coefficients() const noexcept {
        return coeffs_;
    }

    // ── Spectral analysis utilities ───────────────────────────────────────────

    /**
     * @brief Computes the Coherent Gain (mean value of the window).
     *
     * @details
     * Formula: CG = (1/N) * sum_{n=0}^{N-1} w[n]
     *
     * Measures how much the window attenuates a constant (DC) signal.
     * Divide your FFT magnitudes by CG to recover the original amplitudes.
     *
     * std::accumulate(begin, end, init) sums a range starting from `init`.
     * We pass 0.0 (double) to force floating-point arithmetic.
     *
     * @return Coherent gain in (0, 1].
     */
    [[nodiscard]] double coherentGain() const noexcept {
        double sum = std::accumulate(coeffs_.begin(), coeffs_.end(), 0.0);
        return sum / static_cast<double>(N_);
    }

    /**
     * @brief Computes the Equivalent Noise Bandwidth (ENBW) in bins.
     *
     * @details
     * Formula: ENBW = N * sum(w[n]^2) / (sum w[n])^2
     *
     * std::inner_product(a.begin(), a.end(), b.begin(), init) computes
     * the dot product: sum(a[i] * b[i]). Passing the same vector twice
     * gives the sum of squares: sum(w[i]^2).
     *
     * @return ENBW in frequency bins (dimensionless).
     */
    [[nodiscard]] double powerBandwidth() const noexcept {
        double sumSq = std::inner_product(
            coeffs_.begin(), coeffs_.end(),
            coeffs_.begin(),
            0.0
        );
        double cg = coherentGain();
        if (cg == 0.0) return 0.0;
        return sumSq / (static_cast<double>(N_) * cg * cg);
    }

// ── Protected section ────────────────────────────────────────────────────────
// `protected`: visible to this class AND derived classes, not to external
// code. Derived class constructors need `this->coeffs_` to populate it
// directly if needed (though normally generate() handles it).
protected:

    std::size_t         N_;       ///< Window length (number of coefficients)
    std::vector<double> coeffs_;  ///< Precomputed coefficients w[0..N-1]
};

} // namespace stft