/**
 * @file HammingWindow.hpp
 * @brief Hamming window function implementation.
 *
 * @details
 * The Hamming window was developed by Richard Hamming. It is a modified
 * raised cosine that does NOT taper to zero at the edges (unlike Hann).
 * The non-zero endpoints give it a narrower main lobe, which means better
 * frequency resolution, but slightly higher sidelobes than Hann.
 *
 * Mathematical formula:
 * @code
 *   w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))    for n = 0, 1, ..., N-1
 * @endcode
 *
 * Compare with Hann: `0.5 - 0.5 * cos(...)`. The 0.54/0.46 split is
 * chosen to minimize the first sidelobe, at the cost of higher far sidelobes.
 *
 * Properties:
 *   - Coherent gain:      0.54
 *   - ENBW:               1.36 bins
 *   - Peak sidelobe:      -43 dB      (better than Hann's -31.5 dB)
 *   - Main lobe width:    4 bins (narrower effective resolution than Hann)
 *
 * Typical use: speech processing, where precise frequency localization
 * matters more than complete sidelobe suppression.
 *
 * @author  AMSC_STFT Project
 * @version 2.0
 */

#pragma once

#include "BaseWindow.hpp"
#include <algorithm>  // std::clamp
#include <cmath>
#include <numbers>

namespace stft {

/**
 * @class HammingWindow
 * @brief Hamming window: non-zero endpoints, better first-sidelobe suppression.
 *
 * @details
 * Only implements generate() — all shared logic lives in BaseWindow<HammingWindow>.
 *
 * @code
 *   stft::HammingWindow w(512);
 *   std::vector<double> frame = getFrame();
 *   w.apply(frame);
 * @endcode
 */
class HammingWindow : public BaseWindow<HammingWindow> {
public:

    /**
     * @brief Constructs a Hamming window of size N.
     * @param N  Number of coefficients. Must be >= 2.
     */
    explicit HammingWindow(std::size_t N) : BaseWindow<HammingWindow>(N) {}

    /**
     * @brief Computes the Hamming window coefficients.
     *
     * @details
     * Called exactly once by BaseWindow's constructor via CRTP.
     *
     * Note the values 0.54 and 0.46. These are the "optimal" coefficients
     * found by Hamming to minimize the height of the first sidelobe.
     * They satisfy: a0 + a1 = 1 (so the window sums to non-zero at edges).
     *
     * Floating-point rounding can cause values to slightly exceed [0.08, 1.0].
     * We clamp them for numerical stability.
     *
     * @param N  Number of coefficients to generate.
     * @return   Vector of N Hamming coefficients in [0.08, 1.0].
     */
    [[nodiscard]] std::vector<double> generate(std::size_t N) const {
        std::vector<double> w(N);
        for (std::size_t n = 0; n < N; ++n) {
            double coeff = 0.54 - 0.46 * std::cos(
                       2.0 * std::numbers::pi * static_cast<double>(n)
                       / static_cast<double>(N - 1)
                   );
            // Clamp to [0.08, 1.0] to eliminate floating-point rounding artifacts
            w[n] = std::clamp(coeff, 0.08, 1.0);
        }
        return w;
    }
};

} // namespace stft