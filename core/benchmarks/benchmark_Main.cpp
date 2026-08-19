/**
 * @file benchmark_Main.cpp
 * @brief Serial / OpenMP benchmark entry point (no MPI).
 *
 * @details
 * Three benchmark groups, each writing `kind = "timing"` CSV rows (see the
 * schema documented in benchmark_Suite.hpp):
 *
 *   1. FFT engines (section "fft") on a single transform of increasing size:
 *      RecursiveFFT, IterativeFFT, ParallelFFT (OpenMP, auto-detected thread
 *      count). Speedup is reported relative to RecursiveFFT.
 *
 *   2. STFT (section "stft") over signals of increasing length, comparing
 *      STFTAnalyzer<IterativeFFT, HannWindow> serial (1 thread) against the
 *      same analyzer with all available threads. The OpenMP parallelism is
 *      the frame loop inside STFTAnalyzer. Speedup is relative to the serial
 *      run at the same size.
 *
 *   3. Parallelism granularity (section "stft_granularity"): the same thread
 *      budget spent two different ways — parallel ACROSS frames (STFT
 *      OpenMP + IterativeFFT, reusing group 2's measurement) versus parallel
 *      WITHIN each frame's transform (a plain sequential frame loop calling
 *      ParallelFFT). Both are reported against the same serial baseline as
 *      group 2, so the two strategies are directly comparable.
 *
 *      Both rows run through STFTAnalyzer itself: Parallelism::Frames for
 *      the first, Parallelism::Transform plus a ParallelFFT factory for the
 *      second. The factory fixes the engine's thread count at capture time,
 *      which is what keeps the transform parallel while the frame loop is
 *      sequential — the two used to read the same ambient count, and this
 *      row was a hand-written replica of the analyzer's frame loop to work
 *      around it.
 *
 * Usage:
 *   ./benchmark_Main [reps] [warmup] [frame] [hop] [seconds]
 *   OMP_NUM_THREADS=8 ./benchmark_Main
 *   ./benchmark_Main 7 2 8192 4096 20      # coarser frames, 20 s of audio
 *
 * `hop` may be given as 0, meaning frame/2.
 *
 * Output is CSV on stdout, nothing else — see benchmark_Suite.hpp for the
 * column schema and how to read it. MPI / hybrid STFT benchmarks: see
 * benchmark_MPI_Main.
 *
 * Only std::chrono + STL + project code are used.
 */

#include "benchmark_Suite.hpp"

#include "fft/RecursiveFFT.hpp"
#include "fft/IterativeFFT.hpp"
#include "fft/ParallelFFT.hpp"
#include "stft/STFTAnalyzer.hpp"
#include "window/HannWindow.hpp"

#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace stft;

namespace {

int maxThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void setThreads([[maybe_unused]] int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}

} // namespace

int main(int argc, char** argv) {
    int reps   = 7;
    int warmup = 2;
    if (argc >= 2) reps   = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) warmup = std::max(0, std::atoi(argv[2]));

    // Geometry of the STFT sections (the FFT section uses its own size sweep).
    std::size_t frame = 1024;
    std::size_t hop   = 0;        // 0 → frame / 2
    if (argc >= 4) frame = static_cast<std::size_t>(std::max(2, std::atoi(argv[3])));
    if (argc >= 5) hop   = static_cast<std::size_t>(std::max(0, std::atoi(argv[4])));
    if (hop == 0) hop = frame / 2;

    // Workload sweep, expressed as seconds of audio. Passing a duration on
    // the command line collapses the sweep to that single point.
    std::vector<double> durations = {1.0, 5.0, 10.0, 30.0};
    if (argc >= 6) durations = { std::max(0.1, std::atof(argv[5])) };

    if ((frame & (frame - 1)) != 0) {
        std::cerr << "Frame size must be a power of two (got " << frame << ").\n";
        return 1;
    }

    bench::writeHeader(std::cout);

    // ─── FFT engines ────────────────────────────────────────────────────────
    const std::vector<std::size_t> fftSizes = {
        1u << 10, 1u << 12, 1u << 14, 1u << 16, 1u << 18
    };

    for (std::size_t n : fftSizes) {
        const auto input = bench::makeComplexInput(n);

        RecursiveFFT rec;
        IterativeFFT itr;
        ParallelFFT  par(0);  // auto-detect threads

        const bench::Stats sRec = bench::benchFFT(rec, input, warmup, reps);
        const bench::Stats sItr = bench::benchFFT(itr, input, warmup, reps);
        const bench::Stats sPar = bench::benchFFT(par, input, warmup, reps);

        const int threads = maxThreads();
        const double base = sRec.mean;  // baseline = recursive
        bench::writeTiming(std::cout, "fft", "RecursiveFFT", 1, 1, threads,
                           static_cast<long long>(n), std::nullopt, std::nullopt, sRec);
        bench::writeTiming(std::cout, "fft", "IterativeFFT", 1, 1, threads,
                           static_cast<long long>(n), std::nullopt, std::nullopt, sItr, base);
        bench::writeTiming(std::cout, "fft", "ParallelFFT", 1, threads, threads,
                           static_cast<long long>(n), std::nullopt, std::nullopt, sPar, base);
    }

    // ─── STFT: serial vs OpenMP, and parallelism granularity ───────────────
    constexpr std::uint32_t SR = 44100;

    // Frame geometry is fixed across the sweep, so one analyzer — and one
    // ParallelFFT engine — serves every workload; only the signal changes.
    STFTAnalyzer<IterativeFFT, HannWindow> analyzer(frame, hop, SR);

    // Captured before clamping: omp_set_num_threads() writes the same
    // variable omp_get_max_threads() reads back, so after setThreads(1) the
    // machine's real core count is no longer readable.
    const int nThreads = maxThreads();

    // The same analyzer, with the thread budget spent inside each transform
    // instead of across the frames. The factory captures nThreads by value:
    // ParallelFFT would otherwise read the ambient count when built, and the
    // setThreads(1) below would collapse it to serial along with the frame
    // loop rather than leaving it parallel independently of it.
    STFTAnalyzer<ParallelFFT, HannWindow> transformAnalyzer(
        frame, hop, SR,
        [n = static_cast<std::size_t>(nThreads)] { return ParallelFFT(n); },
        Parallelism::Transform);

    for (const double sec : durations) {
        const std::size_t signalLen =
            static_cast<std::size_t>(sec * static_cast<double>(SR));
        const std::size_t frames =
            STFTAnalyzer<IterativeFFT, HannWindow>::numFrames(signalLen, frame, hop);

        // A signal shorter than one frame produces no frames at all: there is
        // nothing to measure, and the row would report a meaningless 0.
        if (frames == 0) continue;

        const auto signal = bench::makeSignal(signalLen, SR);

        setThreads(1);
        const bench::Stats sSerial = bench::benchSTFT(analyzer, signal, warmup, reps);
        bench::writeTiming(std::cout, "stft", "STFT serial (1 thread)", 1, 1, nThreads,
                           std::nullopt, static_cast<long long>(frames), sec, sSerial);

        setThreads(nThreads);
        const bench::Stats sOmp = bench::benchSTFT(analyzer, signal, warmup, reps);
        bench::writeTiming(std::cout, "stft",
                           "STFT OpenMP (" + std::to_string(nThreads) + " thr)",
                           1, nThreads, nThreads, std::nullopt,
                           static_cast<long long>(frames), sec, sOmp, sSerial.mean);

        // setThreads() does not reach this measurement: under
        // Parallelism::Transform the frame loop has no OpenMP region, and the
        // engine's thread count was fixed when the factory captured it.
        const bench::Stats sGranular =
            bench::benchSTFT(transformAnalyzer, signal, warmup, reps);
        bench::writeTiming(std::cout, "stft_granularity",
                           "sequential frames + ParallelFFT (" +
                               std::to_string(nThreads) + " thr)",
                           1, nThreads, nThreads, std::nullopt,
                           static_cast<long long>(frames), sec, sGranular, sSerial.mean);
        bench::writeTiming(std::cout, "stft_granularity",
                           "OpenMP frames + IterativeFFT (" +
                               std::to_string(nThreads) + " thr)",
                           1, nThreads, nThreads, std::nullopt,
                           static_cast<long long>(frames), sec, sOmp, sSerial.mean);
    }

    return 0;
}
