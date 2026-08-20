/**
 * @file benchmark_Suite.hpp
 * @brief Reproducible micro-benchmark utilities for the AMSC_STFT project.
 *
 * @details
 * Header-only, dependency-free (STL + project code only).  Provides:
 *   - high-resolution timing built on std::chrono
 *   - simple statistics (min / mean / median / stddev) over repeated runs
 *   - deterministic signal generators (reproducible across runs and ranks)
 *   - a single CSV row schema, shared by every benchmark this project has
 *     (benchmark_Main.cpp and benchmark_MPI_Main.cpp alike), and the writer
 *     that emits it
 *   - generic, concept-constrained benchmark drivers for FFT and STFT
 *
 * This header is intentionally MPI-free so that the serial/OpenMP benchmark
 * (benchmark_Main.cpp) can use it without linking MPI.  The MPI timing logic
 * lives in benchmark_MPI_Main.cpp.
 *
 * The benchmark logic NEVER modifies FFT/STFT behaviour — it only times the
 * existing public APIs (forward(), analyze()).
 *
 * ─── Output is data, not a report ────────────────────────────────────────────
 * Every benchmark writes CSV rows to stdout and nothing else: no titles, no
 * aligned columns, no explanatory notes. That text was for a human reading a
 * terminal, but the actual reader is analysis/parse_results.py, and a table
 * meant to be pretty is exactly what makes a parser complicated — column
 * widths to split on, section titles to recognise, a label like "MPI pure (1
 * thr/rank)" to re-derive ranks and threads from by regex. A CSV row carries
 * that same ranks/threads distinction as two plain integer columns, written by
 * the code that already knows them with certainty, so nothing downstream has
 * to guess. See the Row documentation below for what each column means.
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
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace stft::bench {

// ─── Statistics ───────────────────────────────────────────────────────────────

struct Stats {
    double min    = 0.0;   ///< fastest run  (ms)
    double mean   = 0.0;   ///< arithmetic mean (ms)
    double median = 0.0;   ///< median (ms)
    double stddev = 0.0;   ///< sample standard deviation (ms)
    int    reps   = 0;     ///< number of timed repetitions
};

/// @brief Min / mean / median / sample stddev over a set of timings (ms).
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

// ─── Prevent the optimiser from discarding benchmarked results ────────────────

/// @brief Forces `value` to look "used" to the compiler, so a benchmarked
///        computation whose result is otherwise unread is not optimised away.
template<typename T>
inline void doNotOptimize(const T& value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(&value) : "memory");
#else
    volatile const void* sink = &value;
    (void)sink;
#endif
}

// ─── Timing ─────────────────────────────────────────────────────────────────

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

// ─── Deterministic, reproducible signal generators ─────────────────────────────

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

// ─── CSV output ────────────────────────────────────────────────────────────────

/**
 * @brief One CSV row: the schema shared by every measurement either binary
 *        takes, a timed run or a memory footprint alike.
 *
 * Only the fields relevant to `kind` are filled; the rest are left blank
 * (std::nullopt) rather than forcing unrelated columns — "strategy" — on a
 * row that has nothing to put there — such as an FFT timing row.
 *
 * @par kind = "timing"
 * One row per (variant, workload). `speedup` is `baseline mean / mean`
 * against the serial row of the same workload — divide it by ranks*threads
 * for the parallel efficiency. `ranks` and `threads` are what THAT VARIANT
 * used (a serial row is 1x1 even if the process had eight cores available);
 * `omp_threads` is what the RUN was given, which differs exactly where it
 * matters: a serial engine handed eight threads still reports 1, and
 * comparing the two is how "this engine does not scale" becomes visible in a
 * plot instead of collapsing to a single point.
 *
 * @par kind = "memory_analytic"
 * How many input samples one rank holds under a given distribution strategy,
 * at a given workload — exact arithmetic on the frame layout, not a
 * measurement, so `bcast` and `scatter` are both reported from a single run
 * without either disturbing the other.
 *
 * @par kind = "memory_rss"
 * Measured peak resident memory (Linux VmHWM) for the whole process
 * lifetime, hence one row per run rather than per workload — the mark left
 * by the largest workload covers every smaller one measured before it.
 * Belongs to the single strategy the process was launched with; comparing
 * bcast against scatter needs two separate runs. Blank on a platform where
 * VmHWM is not readable, rather than a wrong number.
 */
struct Row {
    std::string kind;                  ///< "timing" | "memory_analytic" | "memory_rss"
    std::string section;               ///< "fft" | "stft" | "mpi"      (timing only)
    std::string variant;               ///< e.g. "IterativeFFT", "hybrid (2 thr/rank)"
    std::string strategy;              ///< "bcast" | "scatter"          (memory only)

    std::optional<int> ranks, threads, ompThreads;
    std::optional<long long> size;     ///< FFT transform length         (fft only)
    std::optional<long long> frames;   ///< STFT frame count       (stft/mpi/memory)
    std::optional<double> workloadS;   ///< seconds of audio            (not fft)

    std::optional<double> minMs, meanMs, medianMs, stddevMs, speedup;
    std::optional<double> minMib, maxMib, avgMib;
};

/// @brief Writes the CSV column header line matching the Row schema below.
inline void writeHeader(std::ostream& os) {
    os << "kind,section,variant,strategy,ranks,threads,omp_threads,size,frames,"
          "workload_s,min_ms,mean_ms,median_ms,stddev_ms,speedup,"
          "min_mib,max_mib,avg_mib\n";
}

namespace detail {

/// Bare field: no quoting needed, none of these values can contain a comma.
inline void put(std::ostream& os, const std::optional<int>& v) { if (v) os << *v; }
inline void put(std::ostream& os, const std::optional<long long>& v) { if (v) os << *v; }
inline void put(std::ostream& os, const std::optional<double>& v) {
    if (v) os << std::fixed << std::setprecision(6) << *v;
}

/// Quoted: labels are free text ("STFT OpenMP (8 thr)") and, while none
/// happens to contain a comma today, nothing guarantees a future one won't.
inline void putQuoted(std::ostream& os, const std::string& s) {
    os << '"' << s << '"';
}

} // namespace detail

/// @brief Writes one CSV row (leaving unset optional fields blank, per Row).
inline void writeRow(std::ostream& os, const Row& r) {
    using detail::put;
    using detail::putQuoted;
    putQuoted(os, r.kind);        os << ',';
    putQuoted(os, r.section);     os << ',';
    putQuoted(os, r.variant);     os << ',';
    putQuoted(os, r.strategy);    os << ',';
    put(os, r.ranks);             os << ',';
    put(os, r.threads);           os << ',';
    put(os, r.ompThreads);        os << ',';
    put(os, r.size);              os << ',';
    put(os, r.frames);            os << ',';
    put(os, r.workloadS);         os << ',';
    put(os, r.minMs);             os << ',';
    put(os, r.meanMs);            os << ',';
    put(os, r.medianMs);          os << ',';
    put(os, r.stddevMs);          os << ',';
    put(os, r.speedup);           os << ',';
    put(os, r.minMib);            os << ',';
    put(os, r.maxMib);            os << ',';
    put(os, r.avgMib);            os << '\n';
}

/**
 * @brief Build and write a `kind = "timing"` row.
 * @param baselineMeanMs  if set, `speedup` is baselineMeanMs / s.mean;
 *                        left blank for the baseline row itself.
 */
inline void writeTiming(std::ostream& os, const std::string& section,
                        const std::string& variant, int ranks, int threads,
                        int ompThreads, std::optional<long long> size,
                        std::optional<long long> frames,
                        std::optional<double> workloadS, const Stats& s,
                        std::optional<double> baselineMeanMs = std::nullopt) {
    Row r;
    r.kind = "timing";
    r.section = section;
    r.variant = variant;
    r.ranks = ranks;
    r.threads = threads;
    r.ompThreads = ompThreads;
    r.size = size;
    r.frames = frames;
    r.workloadS = workloadS;
    r.minMs = s.min;
    r.meanMs = s.mean;
    r.medianMs = s.median;
    r.stddevMs = s.stddev;
    if (baselineMeanMs && s.mean > 0.0)
        r.speedup = *baselineMeanMs / s.mean;
    writeRow(os, r);
}

// ─── Generic, concept-constrained benchmark drivers ────────────────────────────

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
