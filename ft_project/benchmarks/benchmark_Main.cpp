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
 *   2. STFT over a long signal, comparing:
 *        - serial  : STFTAnalyzer<IterativeFFT, HannWindow> with 1 thread
 *        - OpenMP  : same analyzer with all available threads
 *      (The OpenMP parallelism is the frame loop inside STFTAnalyzer.)
 *      Speedup is reported relative to the serial run.
 *
 * Usage:
 *   ./benchmark_Main [reps] [warmup]
 *   OMP_NUM_THREADS=8 ./benchmark_Main
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

    std::cout << "AMSC_STFT Benchmark Suite (serial / OpenMP)\n"
              << "  repetitions : " << reps << "  (warmup " << warmup << ")\n"
              << "  max threads : " << maxThreads() << "\n";

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

    constexpr std::size_t   FRAME = 1024;
    constexpr std::size_t   HOP   = 512;
    constexpr std::uint32_t SR    = 44100;

    // ~10 s of audio → plenty of frames to parallelise.
    const std::size_t signalLen = 10u * SR;
    const auto signal = bench::makeSignal(signalLen, SR);

    STFTAnalyzer<IterativeFFT, HannWindow> analyzer(FRAME, HOP, SR);
    const std::size_t frames =
        STFTAnalyzer<IterativeFFT, HannWindow>::numFrames(signalLen, FRAME, HOP);

    // Serial (1 thread)
    setThreads(1);
    const bench::Stats sSerial = bench::benchSTFT(analyzer, signal, warmup, reps);
    bench::printRow("STFT serial (1 thread)", frames, sSerial);

    // OpenMP (all threads)
    setThreads(maxThreads());
    const bench::Stats sOmp = bench::benchSTFT(analyzer, signal, warmup, reps);
    bench::printRow("STFT OpenMP (" + std::to_string(maxThreads()) + " thr)",
                    frames, sOmp, sSerial.mean);

    std::cout << "\nNotes:\n"
              << "  - speedup is relative to the first row of each group.\n"
              << "  - set OMP_NUM_THREADS to control the OpenMP thread count.\n"
              << "  - MPI / hybrid STFT benchmarks: see benchmark_MPI_Main.\n";

    return 0;
}
