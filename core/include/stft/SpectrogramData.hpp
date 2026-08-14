/**
 * @file SpectrogramData.hpp
 * @brief Data structure holding the time-frequency magnitude matrix produced
 *        by the STFT (both serial and MPI distributed variants).
 *
 * @details
 * Magnitudes are stored in a single contiguous row-major flat buffer:
 *
 *   magnitudes[f * numBins + b]  =  magnitude at frame f, frequency bin b
 *
 * This layout lets MPI_Gatherv assemble the full spectrogram from per-rank
 * blocks without extra copying: each rank's local slice is already a
 * contiguous chunk of the final buffer.
 *
 * One-sided spectrum: only bins 0..numBins-1 are stored, where
 *   numBins = frameSize / 2 + 1
 * (DC bin + positive-frequency bins up to Nyquist, inclusive).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>

namespace stft {

struct SpectrogramData {
    std::size_t   numFrames  = 0;  ///< Number of time frames
    std::size_t   numBins    = 0;  ///< One-sided frequency bins: frameSize/2 + 1
    std::size_t   frameSize  = 0;  ///< STFT frame size (power of 2)
    std::size_t   hopSize    = 0;  ///< Hop between successive frames
    std::uint32_t sampleRate = 0;  ///< Original audio sample rate (Hz)

    /// Row-major magnitude buffer: size == numFrames * numBins
    std::vector<double> magnitudes;

    // ── Accessors ─────────────────────────────────────────────────────────────

    [[nodiscard]] double& at(std::size_t frame, std::size_t bin) {
        if (frame >= numFrames || bin >= numBins)
            throw std::out_of_range(
                "SpectrogramData::at(" + std::to_string(frame) + "," +
                std::to_string(bin) + ") out of range (" +
                std::to_string(numFrames) + "x" + std::to_string(numBins) + ").");
        return magnitudes[frame * numBins + bin];
    }

    [[nodiscard]] double at(std::size_t frame, std::size_t bin) const {
        if (frame >= numFrames || bin >= numBins)
            throw std::out_of_range(
                "SpectrogramData::at(" + std::to_string(frame) + "," +
                std::to_string(bin) + ") out of range (" +
                std::to_string(numFrames) + "x" + std::to_string(numBins) + ").");
        return magnitudes[frame * numBins + bin];
    }

    [[nodiscard]] bool empty() const noexcept { return magnitudes.empty(); }

    /// Frequency (Hz) for bin index b.  Requires sampleRate > 0 and frameSize > 0.
    [[nodiscard]] double binFrequency(std::size_t bin) const noexcept {
        if (frameSize == 0 || sampleRate == 0) return 0.0;
        return static_cast<double>(bin) * static_cast<double>(sampleRate)
               / static_cast<double>(frameSize);
    }

    /// Time (seconds) at the start of frame f.  Requires sampleRate > 0.
    [[nodiscard]] double frameTime(std::size_t frame) const noexcept {
        if (hopSize == 0 || sampleRate == 0) return 0.0;
        return static_cast<double>(frame) * static_cast<double>(hopSize)
               / static_cast<double>(sampleRate);
    }
};

} // namespace stft
