/**
 * @file benchmark_Suite.hpp
 * @brief Reproducible micro-benchmark utilities for the AMSC_STFT project.
 *
 * @details
 * Header-only, dependency-free (STL + project code only).  Provides:
 *   - high-resolution timing built on std::chrono
 *   - simple statistics (min / mean / median / stddev) over repeated runs
 *   - deterministic signal generators (reproducible across runs and ranks)
 *   - aligned table + speedup printing
 *   - generic, concept-constrained benchmark drivers for FFT and STFT
 *
 * This header is intentionally MPI-free so that the serial/OpenMP benchmark
 * (benchmark_Main.cpp) can use it without linking MPI.  The MPI timing logic
 * lives in benchmark_MPI_Main.cpp.
 *
 * The benchmark logic NEVER modifies FFT/STFT behaviour — it only times the
 * existing public APIs (forward(), analyze()).
 */

#pragma once

#include "fft/BaseFFT.hpp"
#include "window/BaseWindow.hpp"
#include "stft/STFTAnalyzer.hpp"
#include "stft/SpectrogramData.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <numeric>
#include <string>
#include <vector>

namespace stft::bench {

// ══════════════════════════════════════════════════════════════════════════════
// Statistics
// ══════════════════════════════════════════════════════════════════════════════

struct Stats {
    double min    = 0.0;   ///< fastest run  (ms)
    double mean   = 0.0;   ///< arithmetic mean (ms)
    double median = 0.0;   ///< median (ms)
    double stddev = 0.0;   ///< sample standard deviation (ms)
    int    reps   = 0;     ///< number of timed repetitions
};

inline Stats computeStats(std::vector<double> samplesMs) {
    Stats s;
    s.reps = static_cast<int>(samplesMs.size());
    if (samplesMs.empty()) return s;

    std::sort(samplesMs.begin(), samplesMs.end());
    s.min = samplesMs.front();

    const double sum = std::accumulate(samplesMs.begin(), samplesMs.end(), 0.0);
    s.mean = sum / static_cast<double>(samplesMs.size());

    const std::size_t mid = samplesMs.size() / 2;
    s.median = (samplesMs.size() % 2 == 0)
             ? 0.5 * (samplesMs[mid - 1] + samplesMs[mid])
             : samplesMs[mid];

    double acc = 0.0;
    for (double v : samplesMs) acc += (v - s.mean) * (v - s.mean);
    s.stddev = (samplesMs.size() > 1)
             ? std::sqrt(acc / static_cast<double>(samplesMs.size() - 1))
             : 0.0;
    return s;
}

// ══════════════════════════════════════════════════════════════════════════════
// Prevent the optimiser from discarding benchmarked results
// ══════════════════════════════════════════════════════════════════════════════

template<typename T>
inline void doNotOptimize(const T& value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(&value) : "memory");
#else
    volatile const void* sink = &value;
    (void)sink;
#endif
}

// ══════════════════════════════════════════════════════════════════════════════
// Timing
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Time `run` over `reps` repetitions, calling `prepare` (untimed) before
 *        each one (e.g. to reset an in-place buffer).
 *
 * @param warmup  number of untimed warm-up iterations (cache/thread warm-up)
 * @param reps    number of timed iterations
 * @param prepare callable run before each timed iteration (not measured)
 * @param run     callable that performs the work being measured
 */
template<typename Prepare, typename Run>
inline Stats measure(int warmup, int reps, Prepare&& prepare, Run&& run) {
    using clock = std::chrono::steady_clock;

    for (int i = 0; i < warmup; ++i) { prepare(); run(); }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(std::max(reps, 0)));

    for (int i = 0; i < reps; ++i) {
        prepare();
        const auto t0 = clock::now();
        run();
        const auto t1 = clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return computeStats(std::move(samples));
}

/// Overload with no per-iteration preparation step.
template<typename Run>
inline Stats measure(int warmup, int reps, Run&& run) {
    return measure(warmup, reps, []{}, std::forward<Run>(run));
}

// ══════════════════════════════════════════════════════════════════════════════
// Deterministic, reproducible signal generators
// ══════════════════════════════════════════════════════════════════════════════

/// Composite sine signal (440 + 880 + 1760 Hz).  Identical on every call/rank.
inline std::vector<double> makeSignal(std::size_t n, std::uint32_t sampleRate = 44100) {
    std::vector<double> sig(n);
    const double fs = static_cast<double>(sampleRate);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / fs;
        sig[i] = 0.5 * std::sin(2.0 * std::numbers::pi * 440.0  * t)
               + 0.3 * std::sin(2.0 * std::numbers::pi * 880.0  * t)
               + 0.2 * std::sin(2.0 * std::numbers::pi * 1760.0 * t);
    }
    return sig;
}

/// Complex input of length `n` (power of two) built from makeSignal (imag = 0).
inline std::vector<std::complex<double>> makeComplexInput(std::size_t n) {
    const auto sig = makeSignal(n);
    std::vector<std::complex<double>> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = {sig[i], 0.0};
    return data;
}

// ══════════════════════════════════════════════════════════════════════════════
// Printing (aligned table + machine-readable note)
// ══════════════════════════════════════════════════════════════════════════════

inline void printSectionTitle(const std::string& title) {
    std::cout << "\n" << title << "\n"
              << std::string(title.size(), '=') << "\n";
}

inline void printTableHeader() {
    std::cout << std::left  << std::setw(26) << "Variant"
              << std::right << std::setw(12) << "Size/Frames"
              << std::setw(12) << "min(ms)"
              << std::setw(12) << "mean(ms)"
              << std::setw(12) << "median(ms)"
              << std::setw(11) << "stddev"
              << std::setw(11) << "speedup" << "\n"
              << std::string(94, '-') << "\n";
}

/**
 * @brief Print one result row.
 * @param baselineMeanMs  if > 0, prints speedup = baselineMeanMs / s.mean,
 *                        otherwise prints "-" (use for the baseline row itself).
 */
inline void printRow(const std::string& label,
                     std::size_t size,
                     const Stats& s,
                     double baselineMeanMs = 0.0) {
    std::cout << std::left  << std::setw(26) << label
              << std::right << std::setw(12) << size
              << std::fixed << std::setprecision(3)
              << std::setw(12) << s.min
              << std::setw(12) << s.mean
              << std::setw(12) << s.median
              << std::setw(11) << s.stddev;
    if (baselineMeanMs > 0.0 && s.mean > 0.0)
        std::cout << std::setw(10) << (baselineMeanMs / s.mean) << "x";
    else
        std::cout << std::setw(11) << "-";
    std::cout << "\n";
}

// ══════════════════════════════════════════════════════════════════════════════
// Generic, concept-constrained benchmark drivers
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Benchmark a forward FFT (in-place).  The input is copied (untimed)
 *        before each run so every repetition transforms identical data.
 */
template<typename FFT>
    requires IsFFT<FFT>
inline Stats benchFFT(FFT& fft,
                      const std::vector<std::complex<double>>& input,
                      int warmup, int reps) {
    std::vector<std::complex<double>> work;
    return measure(warmup, reps,
        [&]{ work = input; },               // untimed: reset buffer
        [&]{ fft.forward(work); doNotOptimize(work); });
}

/**
 * @brief Benchmark an STFT analyze() pass over a whole signal.
 */
template<typename FFT, typename Window>
    requires IsFFT<FFT> && WindowFunction<Window>
inline Stats benchSTFT(STFTAnalyzer<FFT, Window>& analyzer,
                       const std::vector<double>& signal,
                       int warmup, int reps) {
    SpectrogramData out;
    return measure(warmup, reps,
        [&]{ out = analyzer.analyze(signal); doNotOptimize(out); });
}

} // namespace stft::bench
