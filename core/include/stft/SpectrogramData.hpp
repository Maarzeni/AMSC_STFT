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
 *
 * ─── Why the matrix is float and the arithmetic is not ──────────────────────
 * Every magnitude is computed in double (see STFTAnalyzer::computeFrame) and
 * narrowed once, on the way into this buffer.  Nothing downstream needs more:
 * the magnitudes exist to be turned into dB over a bounded dynamic range and
 * mapped to 8-bit colour, and float's 24-bit mantissa resolves ~7 decimal
 * digits against the ~4 decades an 80 dB range spans.
 *
 * The reason to care is MPI_Gatherv.  The distributed analyzer assembles this
 * whole matrix on the root rank, and at the usual geometry (frameSize 1024,
 * hop 512) it is as large as the input signal itself — the single largest
 * transfer in the program, and the one that bounds its scaling.  Storing it as
 * float halves that transfer outright.
 *
 * Magnitude is a named alias rather than a bare `float` so the trade is one
 * edit away from being reversed, and so call sites that need the element type
 * (spans over a frame, MPI datatypes) can say what they mean.  The matching
 * MPI datatype lives in MPI_STFTAnalyzer.hpp, next to the only collective that
 * uses it, so this header stays free of <mpi.h>.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace stft {

/**
 * @brief Element type of the magnitude matrix.
 *
 * float, deliberately: see the note at the top of this file.  Change it here
 * and the analyzers, the exporter and the MPI datatype below follow.
 */
using Magnitude = float;

/**
 * @brief Allocator that DEFAULT-initialises instead of value-initialising.
 *
 * std::vector<T>::resize(n) value-initialises the new elements, which for a
 * numeric type means writing zeros over every one of them.  For the magnitude
 * matrix those zeros are pure waste: both writers fill the buffer completely
 * before anything reads it, so every zero is overwritten without ever having
 * been looked at.  It is not a small waste — at the 300 s benchmark geometry
 * the fill measures 11 ms against 5 ms for the transfer that follows it, i.e.
 * roughly two thirds of what the gather phase appeared to cost.
 *
 * This allocator differs from std::allocator in one respect: its construct()
 * default-initialises, which for a trivial type does nothing at all.  resize()
 * then reserves memory and stops.
 *
 * @warning The consequence is that resize() leaves the new elements holding
 *          UNSPECIFIED values, not zeros.  A buffer of this type must be
 *          written in full before it is read.  Both of the places that grow one
 *          satisfy that by construction, and both are covered by tests:
 *
 *          - STFTAnalyzer::analyzeRange resizes to count * numBins and its
 *            frame loop writes every one of the count rows, numBins values at
 *            a time (see computeFrame's outRow).
 *          - MPI_STFTAnalyzer::analyze resizes root's buffer to
 *            totalFrames * numBins, and the MPI_Gatherv that follows covers it
 *            exactly: the per-rank frame counts sum to totalFrames and their
 *            displacements are contiguous.
 *
 *          test_MPI_STFTAnalyzer compares EVERY element of the gathered matrix
 *          against the serial reference at 1 to 4 ranks under both distribution
 *          strategies, so a byte left uncovered by either writer shows up as a
 *          mismatch rather than as a plausible-looking number.  Anything new
 *          that resizes a MagnitudeBuffer inherits this obligation.
 */
template <typename T>
class UninitialisedAllocator : public std::allocator<T> {
public:
    using std::allocator<T>::allocator;

    /// Default-initialise: a no-op for trivial T, which is the whole point.
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }

    /// Every other construction is forwarded unchanged, so resize(n, value),
    /// assignment from an initializer_list and the rest behave normally.
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    [[nodiscard]] bool operator==(const UninitialisedAllocator<U>&) const noexcept {
        return true;   // stateless: any instance can free any other's memory
    }
};

/// The magnitude matrix's storage.  Named because its resize() semantics are
/// not std::vector's — see UninitialisedAllocator above before growing one.
using MagnitudeBuffer = std::vector<Magnitude, UninitialisedAllocator<Magnitude>>;

/// The time-frequency magnitude matrix produced by the STFT (serial and MPI).
struct SpectrogramData {
    std::size_t   numFrames  = 0;  ///< Number of time frames
    std::size_t   numBins    = 0;  ///< One-sided frequency bins: frameSize/2 + 1
    std::size_t   frameSize  = 0;  ///< STFT frame size (power of 2)
    std::size_t   hopSize    = 0;  ///< Hop between successive frames
    std::uint32_t sampleRate = 0;  ///< Original audio sample rate (Hz)

    /// Row-major magnitude buffer: size == numFrames * numBins.
    /// resize() does NOT zero the new elements — see MagnitudeBuffer.
    MagnitudeBuffer magnitudes;

    // ── Accessors ─────────────────────────────────────────────────────────────

    /**
     * @brief Mutable access to the magnitude at (frame, bin).
     * @throws std::out_of_range if `frame >= numFrames` or `bin >= numBins`.
     */
    [[nodiscard]] Magnitude& at(std::size_t frame, std::size_t bin) {
        if (frame >= numFrames || bin >= numBins)
            throw std::out_of_range(
                "SpectrogramData::at(" + std::to_string(frame) + "," +
                std::to_string(bin) + ") out of range (" +
                std::to_string(numFrames) + "x" + std::to_string(numBins) + ").");
        return magnitudes[frame * numBins + bin];
    }

    /**
     * @brief Read-only access to the magnitude at (frame, bin).
     * @throws std::out_of_range if `frame >= numFrames` or `bin >= numBins`.
     */
    [[nodiscard]] Magnitude at(std::size_t frame, std::size_t bin) const {
        if (frame >= numFrames || bin >= numBins)
            throw std::out_of_range(
                "SpectrogramData::at(" + std::to_string(frame) + "," +
                std::to_string(bin) + ") out of range (" +
                std::to_string(numFrames) + "x" + std::to_string(numBins) + ").");
        return magnitudes[frame * numBins + bin];
    }

    /// @return True for a default-constructed or otherwise empty spectrogram.
    [[nodiscard]] bool empty() const noexcept { return magnitudes.empty(); }

    /**
     * @brief Frequency (Hz) of bin `bin`, given this spectrogram's own
     *        frameSize and sampleRate.
     * @return 0.0 if frameSize or sampleRate is unset (zero), rather than
     *         dividing by zero.
     */
    [[nodiscard]] double binFrequency(std::size_t bin) const noexcept {
        if (frameSize == 0 || sampleRate == 0) return 0.0;
        return static_cast<double>(bin) * static_cast<double>(sampleRate)
               / static_cast<double>(frameSize);
    }

    /**
     * @brief Time (seconds) at the start of frame `frame`, given this
     *        spectrogram's own hopSize and sampleRate.
     * @return 0.0 if hopSize or sampleRate is unset (zero), rather than
     *         dividing by zero.
     */
    [[nodiscard]] double frameTime(std::size_t frame) const noexcept {
        if (hopSize == 0 || sampleRate == 0) return 0.0;
        return static_cast<double>(frame) * static_cast<double>(hopSize)
               / static_cast<double>(sampleRate);
    }
};

} // namespace stft
