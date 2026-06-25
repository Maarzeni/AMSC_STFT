/**
 * @file STFTAnalyzer.hpp
 * @brief OpenMP-parallel Short-Time Fourier Transform engine.
 *
 * @details
 * STFTAnalyzer<FFT, Window> slices a 1-D audio signal into overlapping frames,
 * applies a window function to each frame, runs a forward FFT, and accumulates
 * the one-sided magnitude spectrum into a SpectrogramData matrix.
 *
 * Template parameters are constrained by C++20 concepts (IsFFT, WindowFunction)
 * so that type errors are reported at the point of instantiation, not buried
 * inside the implementation.
 *
 * ─── Parallelism strategy ────────────────────────────────────────────────────
 * The outer frame loop is parallelised with OpenMP. Because both Window and
 * FFT modify their data in-place, each OpenMP thread constructs its own private
 * Window and FFT instances (declared inside the loop body). This approach
 * requires no synchronisation and scales linearly with the number of cores.
 *
 * Use IterativeFFT as the FFT parameter when running with OpenMP; ParallelFFT
 * inside an OpenMP region would create nested parallelism and is likely slower.
 *
 * ─── MPI integration point ───────────────────────────────────────────────────
 * MPI_STFTAnalyzer holds an STFTAnalyzer<FFT,Window> instance and calls
 * analyzeRange() on a contiguous block of frames assigned to each rank.
 *
 * @tparam FFT     Any type satisfying IsFFT<FFT> (e.g. IterativeFFT)
 * @tparam Window  Any type satisfying WindowFunction<Window> (e.g. HannWindow)
 */

#pragma once

#include "fft/BaseFFT.hpp"
#include "window/BaseWindow.hpp"
#include "stft/SpectrogramData.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stft {

template<typename FFT, typename Window>
    requires IsFFT<FFT> && WindowFunction<Window>
class STFTAnalyzer {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    /**
     * @param frameSize  Number of samples per STFT frame.  Must be a power of
     *                   two and >= 2 (enforced by the FFT layer).
     * @param hopSize    Step between successive frame starts.  Must be >= 1.
     * @param sampleRate Audio sample rate in Hz (stored in SpectrogramData for
     *                   convenience; does not affect computation).
     */
    explicit STFTAnalyzer(std::size_t frameSize,
                          std::size_t hopSize,
                          std::uint32_t sampleRate = 0)
        : frameSize_(frameSize), hopSize_(hopSize), sampleRate_(sampleRate)
    {
        if (frameSize_ < 2 || (frameSize_ & (frameSize_ - 1)) != 0)
            throw std::invalid_argument(
                "STFTAnalyzer: frameSize must be a power of two >= 2, got " +
                std::to_string(frameSize_) + ".");
        if (hopSize_ < 1)
            throw std::invalid_argument(
                "STFTAnalyzer: hopSize must be >= 1.");
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    /**
     * @brief Returns how many complete frames fit in a signal of length n.
     *
     * Defined as: n < frameSize ? 0 : 1 + (n - frameSize) / hopSize
     *
     * The last frame may be zero-padded (see computeFrame) if the signal does
     * not fill it exactly.
     */
    [[nodiscard]] static std::size_t numFrames(std::size_t signalLen,
                                               std::size_t frameSize,
                                               std::size_t hopSize) noexcept {
        if (signalLen < frameSize) return 0;
        return 1 + (signalLen - frameSize) / hopSize;
    }

    [[nodiscard]] std::size_t frameSize()  const noexcept { return frameSize_;  }
    [[nodiscard]] std::size_t hopSize()    const noexcept { return hopSize_;    }
    [[nodiscard]] std::uint32_t sampleRate() const noexcept { return sampleRate_; }

    // ── Analysis ──────────────────────────────────────────────────────────────

    /**
     * @brief Compute the full STFT spectrogram for a signal.
     *
     * @param signal  Input audio samples (any length).
     * @return SpectrogramData with numFrames rows and numBins = frameSize/2+1 cols.
     */
    [[nodiscard]] SpectrogramData
    analyze(const std::vector<double>& signal) const {
        const std::size_t total = numFrames(signal.size(), frameSize_, hopSize_);
        return analyzeRange(signal, 0, total);
    }

    /**
     * @brief Compute the STFT for a contiguous range of frames.
     *
     * Used by MPI_STFTAnalyzer: each rank calls this with its assigned slice
     * [startFrame, startFrame + count).
     *
     * @param signal      Input audio samples.
     * @param startFrame  Index of the first frame to process.
     * @param count       Number of frames to process.
     * @return SpectrogramData with `count` rows, numBins cols.
     *         If count == 0, returns an empty SpectrogramData.
     */
    [[nodiscard]] SpectrogramData
    analyzeRange(const std::vector<double>& signal,
                 std::size_t startFrame,
                 std::size_t count) const
    {
        const std::size_t numBins = frameSize_ / 2 + 1;

        SpectrogramData result;
        result.numFrames  = count;
        result.numBins    = numBins;
        result.frameSize  = frameSize_;
        result.hopSize    = hopSize_;
        result.sampleRate = sampleRate_;
        result.magnitudes.resize(count * numBins, 0.0);

        if (count == 0) return result;

        // Each OpenMP thread gets private Window + FFT + work buffers to avoid
        // data races on in-place mutating objects.
#ifdef _OPENMP
        #pragma omp parallel
        {
            Window win(frameSize_);
            FFT    fft;
            std::vector<double>               buf(frameSize_);
            std::vector<std::complex<double>> spec(frameSize_);

            #pragma omp for schedule(static)
            for (std::ptrdiff_t fi = 0;
                 fi < static_cast<std::ptrdiff_t>(count);
                 ++fi)
            {
                computeFrame(signal, startFrame + static_cast<std::size_t>(fi),
                             win, fft, buf, spec,
                             result.magnitudes.data() +
                             static_cast<std::size_t>(fi) * numBins);
            }
        }
#else
        Window win(frameSize_);
        FFT    fft;
        std::vector<double>               buf(frameSize_);
        std::vector<std::complex<double>> spec(frameSize_);

        for (std::size_t fi = 0; fi < count; ++fi) {
            computeFrame(signal, startFrame + fi,
                         win, fft, buf, spec,
                         result.magnitudes.data() + fi * numBins);
        }
#endif
        return result;
    }

private:
    std::size_t    frameSize_;
    std::size_t    hopSize_;
    std::uint32_t  sampleRate_;

    /**
     * @brief Process a single frame: copy + zero-pad → window → FFT → magnitude.
     *
     * Writes numBins = frameSize/2+1 normalized magnitudes into outRow.
     * Normalization: divide by (frameSize * coherentGain) to recover physical
     * amplitude units (independent of window choice and frame size).
     */
    void computeFrame(const std::vector<double>& signal,
                      std::size_t frameIdx,
                      Window& win,
                      FFT&    fft,
                      std::vector<double>&               buf,
                      std::vector<std::complex<double>>& spec,
                      double* outRow) const
    {
        const std::size_t offset = frameIdx * hopSize_;
        const std::size_t numBins = frameSize_ / 2 + 1;

        // Copy samples into buf; zero-pad tail if the frame extends past signal end.
        for (std::size_t i = 0; i < frameSize_; ++i)
            buf[i] = (offset + i < signal.size()) ? signal[offset + i] : 0.0;

        // Apply window in-place.
        win(buf);

        // Pack into complex input (imag = 0).
        for (std::size_t i = 0; i < frameSize_; ++i)
            spec[i] = {buf[i], 0.0};

        // In-place forward FFT.
        fft.forward(spec);

        // One-sided magnitude, normalized so that a full-scale sinusoid gives 1.0.
        const double norm = static_cast<double>(frameSize_) * win.coherentGain();
        for (std::size_t k = 0; k < numBins; ++k)
            outRow[k] = std::abs(spec[k]) / norm;
    }
};

} // namespace stft
