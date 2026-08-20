/**
 * @file HammingWindow.hpp
 * @brief Hamming window: raised cosine with non-zero endpoints.
 *
 * @details
 * @code
 *   w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))    for n = 0, 1, ..., N-1
 * @endcode
 *
 * The 0.54 / 0.46 split (vs. Hann's 0.5 / 0.5) is chosen to minimise the
 * first sidelobe. Unlike Hann and Blackman, it does not taper to zero at
 * the edges, which narrows the main lobe at the cost of higher far
 * sidelobes — useful when precise frequency localisation (e.g. speech)
 * matters more than complete sidelobe suppression.
 *
 * | Property           | Value      |
 * |---------------------|-----------|
 * | Coherent gain        | 0.54      |
 * | ENBW                 | 1.36 bins |
 * | Peak sidelobe         | -43 dB   |
 * | Main lobe width (-3 dB) | 4 bins |
 */

#pragma once

#include "BaseWindow.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace stft {

/**
 * @brief Hamming window: non-zero endpoints, strong first-sidelobe suppression.
 *
 * Implements only `generate()`; everything else comes from
 * BaseWindow<HammingWindow>.
 */
class HammingWindow : public BaseWindow<HammingWindow> {
public:
    /// @param N  Number of coefficients. Must be >= 2.
    explicit HammingWindow(std::size_t N) : BaseWindow<HammingWindow>(N) {}

    /**
     * @brief Computes the Hamming coefficients. Called once, via CRTP, by
     *        BaseWindow's constructor.
     * @return N coefficients in [0.08, 1.0] (clamped against floating-point
     *         rounding).
     */
    [[nodiscard]] std::vector<double> generate(std::size_t N) const {
        std::vector<double> w(N);
        for (std::size_t n = 0; n < N; ++n) {
            const double coeff = 0.54 - 0.46 * std::cos(
                2.0 * std::numbers::pi * static_cast<double>(n)
                / static_cast<double>(N - 1));
            w[n] = std::clamp(coeff, 0.08, 1.0);
        }
        return w;
    }
};

} // namespace stft
