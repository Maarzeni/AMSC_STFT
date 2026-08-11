/**
 * @file benchmark_Main.cpp
 * @brief Serial / OpenMP benchmark entry point (no MPI).
 *
 * @details
 * Two benchmark groups:
 *
 *   1. FFT engines on a single transform of increasing size:
 *        - RecursiveFFT
 *        - IterativeFFT
 *        - ParallelFFT (OpenMP, auto-detected thread count)
 *      Speedup is reported relative to RecursiveFFT.
 *
 *   2. STFT over signals of increasing length, comparing at each size:
 *        - serial  : STFTAnalyzer<IterativeFFT, HannWindow> with 1 thread
 *        - OpenMP  : same analyzer with all available threads
 *      (The OpenMP parallelism is the frame loop inside STFTAnalyzer.)
 *      Speedup is reported relative to the serial run at the SAME size.
 *
 * Usage:
 *   ./benchmark_Main [reps] [warmup] [frame] [hop] [seconds]
 *   OMP_NUM_THREADS=8 ./benchmark_Main
 *   ./benchmark_Main 7 2 8192 4096 20      # coarser frames, 20 s of audio
 *
 * `hop` may be given as 0, meaning frame/2.
 *
 * Only std::chrono + STL + project code are used.
 */

#include "benchmark_Suite.hpp"

#include "fft/RecursiveFFT.hpp"
#include "fft/IterativeFFT.hpp"
#include "fft/ParallelFFT.hpp"
#include "stft/STFTAnalyzer.hpp"
#include "window/HannWindow.hpp"

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

    // Geometry of the STFT section (the FFT section uses its own size sweep).
    // Larger frames raise the work per OpenMP iteration and thus amortise the
    // fixed synchronisation cost, at the price of fewer frames to distribute.
    std::size_t frame = 1024;
    std::size_t hop   = 0;        // 0 → frame / 2
    if (argc >= 4) frame = static_cast<std::size_t>(std::max(2, std::atoi(argv[3])));
    if (argc >= 5) hop   = static_cast<std::size_t>(std::max(0, std::atoi(argv[4])));
    if (hop == 0) hop = frame / 2;

    // Workload sweep for the STFT section, expressed as seconds of audio, so
    // that the table shows how the OpenMP gain grows with the number of frames
    // the way the FFT table shows it growing with transform size.  Passing a
    // duration on the command line collapses the sweep to that single point.
    std::vector<double> durations = {1.0, 5.0, 10.0, 30.0};
    if (argc >= 6) durations = { std::max(0.1, std::atof(argv[5])) };

    if ((frame & (frame - 1)) != 0) {
        std::cerr << "Frame size must be a power of two (got " << frame << ").\n";
        return 1;
    }

    std::cout << "AMSC_STFT Benchmark Suite (serial / OpenMP)\n"
              << "  repetitions : " << reps << "  (warmup " << warmup << ")\n"
              << "  max threads : " << maxThreads() << "\n"
              << "  frame / hop : " << frame << " / " << hop << " samples\n"
              << "  workloads   : ";
    for (std::size_t i = 0; i < durations.size(); ++i)
        std::cout << durations[i]
                  << (i + 1 < durations.size() ? ", " : " s of audio\n");

    // ══════════════════════════════════════════════════════════════════════════
    // 1. FFT engines
    // ══════════════════════════════════════════════════════════════════════════
    bench::printSectionTitle("FFT engines (single forward transform)");
    bench::printTableHeader();

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

        const double base = sRec.mean;  // baseline = recursive
        bench::printRow("RecursiveFFT", n, sRec);
        bench::printRow("IterativeFFT", n, sItr, base);
        bench::printRow("ParallelFFT",  n, sPar, base);
        std::cout << "\n";
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2. STFT: serial vs OpenMP
    // ══════════════════════════════════════════════════════════════════════════
    bench::printSectionTitle("STFT: serial vs OpenMP (IterativeFFT + HannWindow)");
    bench::printTableHeader();

    constexpr std::uint32_t SR = 44100;

    // Frame geometry is fixed across the sweep, so one analyzer serves every
    // workload: only the signal handed to analyze() changes.
    STFTAnalyzer<IterativeFFT, HannWindow> analyzer(frame, hop, SR);

    // Capture the thread count before clamping: omp_set_num_threads() writes
    // the same internal variable that omp_get_max_threads() reads back, so
    // after setThreads(1) the machine's real core count is no longer readable.
    const int nThreads = maxThreads();

    for (const double sec : durations) {
        const std::size_t signalLen =
            static_cast<std::size_t>(sec * static_cast<double>(SR));
        const std::size_t frames =
            STFTAnalyzer<IterativeFFT, HannWindow>::numFrames(signalLen, frame, hop);

        // A signal shorter than one frame produces no frames at all: there is
        // nothing to measure, and the row would report a meaningless 0.
        if (frames == 0) continue;

        const auto signal = bench::makeSignal(signalLen, SR);

        // Serial (1 thread)
        setThreads(1);
        const bench::Stats sSerial = bench::benchSTFT(analyzer, signal, warmup, reps);
        bench::printRow("STFT serial (1 thread)", frames, sSerial);

        // OpenMP (all threads)
        setThreads(nThreads);
        const bench::Stats sOmp = bench::benchSTFT(analyzer, signal, warmup, reps);
        bench::printRow("STFT OpenMP (" + std::to_string(nThreads) + " thr)",
                        frames, sOmp, sSerial.mean);
        std::cout << "\n";
    }

    std::cout << "Notes:\n"
              << "  - speedup is relative to the first row of each block.\n"
              << "  - each STFT block is one workload; the frame count grows\n"
              << "    down the table, so the OpenMP gain should improve with it.\n"
              << "  - set OMP_NUM_THREADS to control the OpenMP thread count.\n"
              << "  - args: [reps] [warmup] [frame] [hop] [seconds];\n"
              << "    giving [seconds] replaces the sweep with one workload.\n"
              << "  - MPI / hybrid STFT benchmarks: see benchmark_MPI_Main.\n";

    return 0;
}
