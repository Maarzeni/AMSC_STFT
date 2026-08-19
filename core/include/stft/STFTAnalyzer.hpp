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
 * Where the parallelism lives is a runtime choice (see Parallelism below), and
 * it is orthogonal to how the FFT engine is built (see the FFT factory below):
 *
 *   Parallelism::Frames     The outer frame loop runs inside an OpenMP region.
 *                           Because both Window and FFT mutate their data in
 *                           place, each thread builds its own private Window,
 *                           FFT and work buffers, so no synchronisation is
 *                           needed.  Pair it with a serial engine such as
 *                           IterativeFFT.  This is the default and matches the
 *                           behaviour of every earlier release.
 *
 *   Parallelism::Transform  The frame loop is plain sequential and one engine
 *                           serves every frame; the parallelism, if any, comes
 *                           from inside the FFT itself.  Pair it with an engine
 *                           that parallelises a single transform, such as
 *                           ParallelFFT.
 *
 * The two modes spend the same thread budget in different places, which is what
 * makes them directly comparable: across frames versus within each transform.
 *
 * ─── Building the FFT engine ────────────────────────────────────────────────
 * The engine is obtained from an FFTFactory (std::function<FFT()>), which
 * defaults to default-construction — so the three-argument constructor behaves
 * exactly as before.  A factory, or an already-constructed instance to copy
 * from, can be injected instead; this is the only way to configure an engine
 * that takes constructor arguments, such as ParallelFFT's thread count.
 *
 * Under Frames the factory is invoked once per thread, inside the OpenMP
 * region, so the engines are never shared.  A factory returning ParallelFFT(0)
 * would therefore resolve "auto-detect" against the nested level rather than
 * the machine: factories must fix the thread count explicitly, capturing it
 * before any omp_set_num_threads() call.  With that caveat honoured, Frames
 * plus a parallel engine is deliberate nested parallelism, with the inner width
 * pinned by the factory instead of being read from the ambient environment.
 *
 * The type-erasure cost of std::function is paid once per thread (Frames) or
 * once per call (Transform), never per frame, and no polymorphic base pointer
 * is introduced: the CRTP static dispatch of BaseFFT is untouched.
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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stft {

/**
 * @brief Where the thread budget is spent during an analysis pass.
 *
 * Frames:    OpenMP across the frame loop, one private engine per thread.  The
 *            FFT itself is expected to be serial (IterativeFFT).
 * Transform: sequential frame loop over a single engine, so the FFT's own
 *            parallel region is the only one (ParallelFFT).
 *
 * Declared at namespace scope so call sites can name it without repeating the
 * analyzer's template arguments — the same choice made for Distribution in
 * MPI_STFTAnalyzer.hpp.
 */
enum class Parallelism {
    Frames,     ///< #pragma omp parallel over the frames (default)
    Transform   ///< sequential frames; parallelism comes from within the FFT
};

template<typename FFT, typename Window>
    requires IsFFT<FFT> && WindowFunction<Window>
class STFTAnalyzer {
public:
    /// Builds one FFT engine.  Invoked once per thread under Parallelism::Frames,
    /// once per analysis call under Parallelism::Transform.
    using FFTFactory = std::function<FFT()>;

    // ── Construction ──────────────────────────────────────────────────────────

    /**
     * @brief Default-constructed engines.
     *
     * @param frameSize  Number of samples per STFT frame.  Must be a power of
     *                   two and >= 2 (enforced by the FFT layer).
     * @param hopSize    Step between successive frame starts.  Must be >= 1.
     * @param sampleRate Audio sample rate in Hz (stored in SpectrogramData for
     *                   convenience; does not affect computation).
     * @param mode       Where the thread budget is spent (see Parallelism).
     */
    explicit STFTAnalyzer(std::size_t frameSize,
                          std::size_t hopSize,
                          std::uint32_t sampleRate = 0,
                          Parallelism mode = Parallelism::Frames)
        // constructible_from, not default_initializable: BaseFFT's default
        // constructor is protected, so FFT{} is ill-formed for the CRTP
        // derivatives (they are aggregates) while FFT() is fine.
        requires std::constructible_from<FFT>
        : frameSize_(frameSize), hopSize_(hopSize), sampleRate_(sampleRate),
          fftFactory_([] { return FFT(); }), mode_(mode)
    {
        validate();
    }

    /**
     * @brief Engines built on demand by a caller-supplied factory.
     *
     * The only way to configure an engine whose constructor takes arguments:
     *
     *   STFTAnalyzer<ParallelFFT, HannWindow> a(
     *       1024, 512, 44100,
     *       [n = threads] { return ParallelFFT(n); },
     *       Parallelism::Transform);
     *
     * @param factory  Must not be empty.  See the "Building the FFT engine"
     *                 note at the top of this file for the ParallelFFT(0)
     *                 caveat under Parallelism::Frames.
     */
    STFTAnalyzer(std::size_t frameSize,
                 std::size_t hopSize,
                 std::uint32_t sampleRate,
                 FFTFactory factory,
                 Parallelism mode = Parallelism::Frames)
        : frameSize_(frameSize), hopSize_(hopSize), sampleRate_(sampleRate),
          fftFactory_(std::move(factory)), mode_(mode)
    {
        if (!fftFactory_)
            throw std::invalid_argument(
                "STFTAnalyzer: the FFT factory must not be empty.");
        validate();
    }

    /**
     * @brief An already-constructed engine, used as a prototype.
     *
     * The prototype is never used directly: every analysis pass copies it, so
     * under Parallelism::Frames each thread still gets its own engine and the
     * const-qualified analyze() never mutates the stored instance.  Defaults to
     * Parallelism::Transform, the reason one usually configures an engine by
     * hand in the first place.
     *
     * Unambiguous against the factory overload: a lambda converts to FFTFactory
     * but not to FFT, and an FFT converts to FFT but not to FFTFactory.
     */
    STFTAnalyzer(std::size_t frameSize,
                 std::size_t hopSize,
                 std::uint32_t sampleRate,
                 FFT prototype,
                 Parallelism mode = Parallelism::Transform)
        requires std::copy_constructible<FFT>
        : STFTAnalyzer(frameSize, hopSize, sampleRate,
                       FFTFactory([p = std::move(prototype)] { return p; }),
                       mode)
    {}

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
    [[nodiscard]] Parallelism parallelism() const noexcept { return mode_;      }

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

#ifdef _OPENMP
        // Each OpenMP thread gets private Window + FFT + work buffers to avoid
        // data races on in-place mutating objects.  The factory runs once per
        // thread, inside the region, so no engine is ever shared.
        if (mode_ == Parallelism::Frames) {
            #pragma omp parallel
            {
                Window win(frameSize_);
                FFT    fft = fftFactory_();
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
            return result;
        }
#endif
        // Parallelism::Transform, and every build without OpenMP: no region of
        // our own, one engine for the whole pass.  Any parallelism here comes
        // from inside fft itself.  The engine is a local, not a member, so the
        // const-qualified analysis can call its non-const forward().
        Window win(frameSize_);
        FFT    fft = fftFactory_();
        std::vector<double>               buf(frameSize_);
        std::vector<std::complex<double>> spec(frameSize_);

        for (std::size_t fi = 0; fi < count; ++fi) {
            computeFrame(signal, startFrame + fi,
                         win, fft, buf, spec,
                         result.magnitudes.data() + fi * numBins);
        }
        return result;
    }

private:
    std::size_t    frameSize_;
    std::size_t    hopSize_;
    std::uint32_t  sampleRate_;
    FFTFactory     fftFactory_;   ///< never empty: checked at construction
    Parallelism    mode_;

    /// Geometry checks shared by every constructor.
    void validate() const {
        if (frameSize_ < 2 || (frameSize_ & (frameSize_ - 1)) != 0)
            throw std::invalid_argument(
                "STFTAnalyzer: frameSize must be a power of two >= 2, got " +
                std::to_string(frameSize_) + ".");
        if (hopSize_ < 1)
            throw std::invalid_argument(
                "STFTAnalyzer: hopSize must be >= 1.");
    }

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
