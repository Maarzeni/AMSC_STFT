/**
 * @file HannWindow.hpp
 * @brief Hann window: raised cosine tapering to zero at both endpoints.
 *
 * @details
 * @code
 *   w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))    for n = 0, 1, ..., N-1
 * @endcode
 *
 * The most widely used general-purpose window: a good trade-off between
 * frequency resolution and spectral leakage.
 *
 * | Property           | Value      |
 * |---------------------|-----------|
 * | Coherent gain        | 0.5       |
 * | ENBW                 | 1.5 bins  |
 * | Peak sidelobe         | -31.5 dB |
 * | Main lobe width (-3 dB) | 4 bins |
 */

#pragma once

#include "BaseWindow.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace stft {

/**
 * @brief Hann window: smooth raised cosine, general-purpose default.
 *
 * Implements only `generate()`; everything else — `apply()`, `size()`,
 * `coefficients()`, `coherentGain()`, `powerBandwidth()` — comes from
 * BaseWindow<HannWindow>.
 */
class HannWindow : public BaseWindow<HannWindow> {
public:
    /// @param N  Number of coefficients. Must be >= 2.
    explicit HannWindow(std::size_t N) : BaseWindow<HannWindow>(N) {}

    /**
     * @brief Computes the Hann coefficients. Called once, via CRTP, by
     *        BaseWindow's constructor.
     * @return N coefficients in [0, 1] (clamped against floating-point
     *         rounding at the endpoints).
     */
    [[nodiscard]] std::vector<double> generate(std::size_t N) const {
        std::vector<double> w(N);
        for (std::size_t n = 0; n < N; ++n) {
            const double coeff = 0.5 * (1.0 - std::cos(
                2.0 * std::numbers::pi * static_cast<double>(n)
                / static_cast<double>(N - 1)));
            w[n] = std::clamp(coeff, 0.0, 1.0);
        }
        return w;
    }
};

} // namespace stft
