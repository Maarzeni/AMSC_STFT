/**
 * @file BlackmanWindow.hpp
 * @brief Blackman window: three-term cosine, lowest sidelobes of the three.
 *
 * @details
 * @code
 *   w[n] = 0.42 - 0.50*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
 *                                                for n = 0, 1, ..., N-1
 * @endcode
 *
 * The coefficients come from the general Blackman formula with alpha =
 * 0.16: a0 = (1-alpha)/2, a1 = 0.5, a2 = alpha/2; they satisfy
 * a0 - a1 + a2 = 0, so the window tapers to exactly zero at both endpoints.
 * The second cosine term (at twice the frequency of the first) is what
 * distinguishes it from the two-term Hann/Hamming windows: a wider main
 * lobe in exchange for much lower sidelobes. Useful when suppressing
 * spectral leakage matters more than frequency resolution — e.g.
 * detecting a weak signal near a strong one.
 *
 * | Property           | Value      |
 * |---------------------|-----------|
 * | Coherent gain        | 0.42      |
 * | ENBW                 | 1.73 bins |
 * | Peak sidelobe         | -58 dB   |
 * | Main lobe width (-3 dB) | 6 bins |
 *
 * | Window   | Sidelobe | Resolution | Typical use               |
 * |----------|----------|------------|----------------------------|
 * | Hann     | -31 dB   | good       | general purpose (default)  |
 * | Hamming  | -43 dB   | better     | speech processing          |
 * | Blackman | -58 dB   | worse      | precision spectral analysis|
 */

#pragma once

#include "BaseWindow.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace stft {

/**
 * @brief Blackman window: three-term cosine, lowest sidelobes of the three.
 *
 * Implements only `generate()`; everything else comes from
 * BaseWindow<BlackmanWindow>.
 */
class BlackmanWindow : public BaseWindow<BlackmanWindow> {
public:
    /// @param N  Number of coefficients. Must be >= 2.
    explicit BlackmanWindow(std::size_t N) : BaseWindow<BlackmanWindow>(N) {}

    /**
     * @brief Computes the Blackman coefficients. Called once, via CRTP, by
     *        BaseWindow's constructor.
     * @return N coefficients in [0, 1] (clamped against floating-point
     *         rounding at the endpoints).
     */
    [[nodiscard]] std::vector<double> generate(std::size_t N) const {
        std::vector<double> w(N);
        const double twoPiOverNm1 = 2.0 * std::numbers::pi / static_cast<double>(N - 1);

        for (std::size_t n = 0; n < N; ++n) {
            const double angle = twoPiOverNm1 * static_cast<double>(n);
            const double coeff = 0.42
                                - 0.50 * std::cos(angle)
                                + 0.08 * std::cos(2.0 * angle);
            w[n] = std::clamp(coeff, 0.0, 1.0);
        }
        return w;
    }
};

} // namespace stft
