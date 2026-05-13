/**
 * @file BlackmanWindow.hpp
 * @brief Blackman window function implementation.
 *
 * @details
 * The Blackman window was developed by Ralph Beebe Blackman. It uses
 * THREE cosine terms instead of two, which gives it a much wider main
 * lobe but dramatically lower sidelobes than both Hann and Hamming.
 *
 * Mathematical formula:
 * @code
 *   w[n] = 0.42
 *          - 0.50 * cos(2*pi*n / (N-1))
 *          + 0.08 * cos(4*pi*n / (N-1))    for n = 0, 1, ..., N-1
 * @endcode
 *
 * The coefficients (0.42, 0.50, 0.08) come from the general Blackman
 * formula with alpha = 0.16: a0 = (1-alpha)/2, a1 = 0.5, a2 = alpha/2.
 * They satisfy: a0 - a1 + a2 = 0 (window tapers to exactly 0 at endpoints).
 *
 * Properties:
 *   - Coherent gain:      0.42
 *   - ENBW:               1.73 bins   (wider than Hann and Hamming)
 *   - Peak sidelobe:      -58 dB      (much better than Hann/Hamming)
 *   - Main lobe width:    6 bins      (wider: lower frequency resolution)
 *
 * Use when: you need to suppress spectral leakage as much as possible
 * and can sacrifice some frequency resolution. Common in audio analysis
 * where you need to detect weak signals near strong ones.
 *
 * Hann vs Hamming vs Blackman summary:
 * @code
 *   Window   | Sidelobe | Resolution | Use case
 *   ---------+----------+------------+---------------------------
 *   Hann     |  -31 dB  | Good       | General purpose (default)
 *   Hamming  |  -43 dB  | Better     | Speech processing
 *   Blackman |  -58 dB  | Worse      | Precision spectral analysis
 * @endcode
 *
 * @author  AMSC_STFT Project
 * @version 2.0
 */

#pragma once

#include "BaseWindow.hpp"
#include <cmath>
#include <numbers>

namespace window {

/**
 * @class BlackmanWindow
 * @brief Blackman window: three-term cosine, lowest sidelobes of the three.
 *
 * @details
 * Only implements generate(). All shared logic lives in BaseWindow<BlackmanWindow>.
 *
 * @code
 *   window::BlackmanWindow w(2048);
 *   std::vector<double> frame = getFrame();
 *   w.apply(frame);
 * @endcode
 */
class BlackmanWindow : public BaseWindow<BlackmanWindow> {
public:

    /**
     * @brief Constructs a Blackman window of size N.
     * @param N  Number of coefficients. Must be >= 2.
     */
    explicit BlackmanWindow(std::size_t N) : BaseWindow<BlackmanWindow>(N) {}

    /**
     * @brief Computes the Blackman window coefficients.
     *
     * @details
     * Called exactly once by BaseWindow's constructor via CRTP.
     *
     * We use `2.0 * std::numbers::pi` and `4.0 * std::numbers::pi` for
     * the two cosine frequencies. The second term (at 2x the frequency)
     * is what distinguishes Blackman from the two-term windows.
     *
     * A small optimization: precompute `2*pi*n/(N-1)` once per iteration
     * and reuse it for both cosine calls, avoiding redundant divisions.
     *
     * Floating-point rounding can cause endpoint values to be slightly
     * negative (e.g., -1e-17). We clamp them to [0, 1] for numerical stability.
     *
     * @param N  Number of coefficients to generate.
     * @return   Vector of N Blackman coefficients in [0, 1].
     */
    [[nodiscard]] std::vector<double> generate(std::size_t N) const {
        std::vector<double> w(N);

        // Precompute the constant part of the angle to avoid recomputing
        // the division `1.0 / (N-1)` at each loop iteration.
        // `const` marks this variable as non-modifiable — purely a hint
        // to the reader and the optimizer.
        const double twoPiOverNm1 =
            2.0 * std::numbers::pi / static_cast<double>(N - 1);

        for (std::size_t n = 0; n < N; ++n) {
            // angle = 2*pi*n / (N-1)
            const double angle = twoPiOverNm1 * static_cast<double>(n);

            double coeff = 0.42
                           - 0.50 * std::cos(angle)        // first harmonic
                           + 0.08 * std::cos(2.0 * angle); // second harmonic

            // Clamp to [0, 1] to eliminate floating-point rounding artifacts
            w[n] = std::clamp(coeff, 0.0, 1.0);
        }
        return w;
    }
};

} // namespace window