/**
 * @file HannWindow.hpp
 * @brief Hann window function implementation.
 *
 * @details
 * The Hann window (often incorrectly called "Hanning") was developed by
 * Julius von Hann. It is the most widely used window in audio FFT analysis
 * because it offers a good trade-off between:
 *   - **Frequency resolution** (how well it separates nearby frequencies)
 *   - **Spectral leakage** (how much energy "bleeds" into neighboring bins)
 *
 * Mathematical formula:
 * @code
 *   w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))    for n = 0, 1, ..., N-1
 * @endcode
 *
 * This is a raised cosine that starts at 0, rises smoothly to 1 at the
 * center, and falls back to 0. The smooth tapering at the edges is what
 * reduces spectral leakage.
 *
 * Properties:
 *   - Coherent gain:      0.5
 *   - ENBW:               1.5 bins
 *   - Peak sidelobe:      -31.5 dB
 *   - Main lobe width:    4 bins (at -3 dB)
 *
 * @author  AMSC_STFT Project
 * @version 2.0
 */

#pragma once

#include "BaseWindow.hpp"
#include <cmath>     // std::cos
#include <numbers>   // std::numbers::pi (C++20 mathematical constants)

namespace window {

/**
 * @class HannWindow
 * @brief Hann window: smooth raised cosine, best general-purpose choice.
 *
 * @details
 * This class only implements `generate()` — the mathematical formula.
 * Everything else (apply, size, coefficients, coherentGain, powerBandwidth,
 * the coefficient vector, error checking) is inherited from BaseWindow<HannWindow>.
 *
 * @code
 *   // Creating and using a Hann window:
 *   window::HannWindow w(1024);
 *   std::vector<double> frame = getFrame();  // 1024 samples
 *   w.apply(frame);                          // window applied in place
 *   // frame is now ready for FFT
 * @endcode
 */
class HannWindow : public BaseWindow<HannWindow> {
public:

    /**
     * @brief Constructs a Hann window of size N.
     *
     * @details
     * We just forward N to the base class constructor.
     * The base constructor will call generate(N) via CRTP to fill coeffs_.
     *
     * The syntax `: BaseWindow<HannWindow>(N)` is a **member initializer list**.
     * It runs BEFORE the body of the constructor `{}`.
     * Here the body is empty `{}` because all the work is done in BaseWindow.
     *
     * @param N  Number of coefficients. Must be >= 2.
     */
    explicit HannWindow(std::size_t N) : BaseWindow<HannWindow>(N) {}

    /**
     * @brief Computes the Hann window coefficients.
     *
     * @details
     * Called exactly once by BaseWindow's constructor via CRTP.
     * Returns a newly allocated vector — BaseWindow stores it in coeffs_.
     *
     * `std::numbers::pi` is the C++20 compile-time constant for π.
     * It replaces the old non-standard `M_PI` macro.
     *
     * Floating-point rounding can cause values to slightly exceed [0, 1].
     * We clamp them for numerical stability.
     *
     * @param N  Number of coefficients to generate.
     * @return   Vector of N Hann coefficients in [0, 1].
     */
    [[nodiscard]] std::vector<double> generate(std::size_t N) const {

        // Allocate a vector of N doubles, all initialized to 0.0
        std::vector<double> w(N);

        for (std::size_t n = 0; n < N; ++n) {
            // std::numbers::pi is a constexpr double = 3.14159265358979...
            // `2.0 * std::numbers::pi * n` computes the angle in radians.
            // `(N - 1)` is the denominator that normalizes the range to [0, 2pi].
            double coeff = 0.5 * (1.0 - std::cos(
                       2.0 * std::numbers::pi * static_cast<double>(n)
                       / static_cast<double>(N - 1)
                   ));
            // Clamp to [0, 1] to eliminate floating-point rounding artifacts
            w[n] = std::clamp(coeff, 0.0, 1.0);
        }
        // `return w` triggers move semantics automatically (NRVO / move):
        // the vector's internal buffer is transferred to the caller without
        // any data being copied. This is O(1) regardless of N.
        return w;
    }
};

} // namespace window