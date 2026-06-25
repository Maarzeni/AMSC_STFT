/**
 * @file benchmark_MPI_Main.cpp
 * @brief Distributed STFT benchmark entry point (MPI and hybrid MPI+OpenMP).
 *
 * @details
 * Benchmarks MPI_STFTAnalyzer<IterativeFFT, HannWindow> on a long signal,
 * comparing two configurations of the SAME distributed run:
 *
 *   - MPI pure   : 1 OpenMP thread per rank
 *   - hybrid     : all available OpenMP threads per rank
 *
 * Wall-clock time per repetition is the time of the SLOWEST rank
 * (MPI_Reduce with MPI_MAX), since that is what bounds the collective.
 * Only the root rank prints results.
 *
 * Usage (compare across rank counts to read strong scaling):
 *   mpirun -np 1 ./benchmark_MPI_Main
 *   mpirun -np 2 ./benchmark_MPI_Main
 *   mpirun -np 4 ./benchmark_MPI_Main
 *   OMP_NUM_THREADS=4 mpirun -np 2 ./benchmark_MPI_Main   # hybrid
 *
 * Only std::chrono + STL + project code (+ MPI) are used.
 */

#include "benchmark_Suite.hpp"

#include "mpi/MPIContext.hpp"
#include "stft/MPI_STFTAnalyzer.hpp"
#include "stft/STFTAnalyzer.hpp"
#include "fft/IterativeFFT.hpp"
#include "window/HannWindow.hpp"

#include <mpi.h>

#include <chrono>
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

/**
 * @brief Collective timing: every rank runs `run` `reps` times; the per-rep
 *        wall time is the maximum across ranks.  Stats are computed on root.
 *
 * A barrier precedes each timed region so all ranks start together.
 */
template<typename Run>
bench::Stats measureMPI(const MPIContext& ctx, int warmup, int reps, Run&& run) {
    using clock = std::chrono::steady_clock;

    for (int i = 0; i < warmup; ++i) { ctx.barrier(); run(); }

    std::vector<double> local(static_cast<std::size_t>(reps), 0.0);
    for (int i = 0; i < reps; ++i) {
        ctx.barrier();
        const auto t0 = clock::now();
        run();
        const auto t1 = clock::now();
        local[static_cast<std::size_t>(i)] =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::vector<double> global(static_cast<std::size_t>(reps), 0.0);
    MPI_Reduce(local.data(), global.data(), reps, MPI_DOUBLE, MPI_MAX,
               MPIContext::root, MPI_COMM_WORLD);

    if (ctx.isRoot())
        return bench::computeStats(std::move(global));
    return bench::Stats{};
}

} // namespace

int main(int argc, char** argv) {
    MPIContext ctx(argc, argv);

    int reps   = 7;
    int warmup = 2;
    if (argc >= 2) reps   = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) warmup = std::max(0, std::atoi(argv[2]));

    constexpr std::size_t   FRAME = 1024;
    constexpr std::size_t   HOP   = 512;
    constexpr std::uint32_t SR    = 44100;

    const std::size_t signalLen = 10u * SR;   // ~10 s of audio
    const std::size_t frames =
        STFTAnalyzer<IterativeFFT, HannWindow>::numFrames(signalLen, FRAME, HOP);

    // Root generates the signal; non-root pass an empty vector (the analyzer
    // broadcasts internally).  Deterministic generator → reproducible.
    std::vector<double> signal;
    if (ctx.isRoot()) signal = bench::makeSignal(signalLen, SR);

    MPI_STFTAnalyzer<IterativeFFT, HannWindow> analyzer(ctx, FRAME, HOP, SR);

    if (ctx.isRoot()) {
        std::cout << "AMSC_STFT Benchmark Suite (MPI / hybrid)\n"
                  << "  ranks        : " << ctx.size() << "\n"
                  << "  threads/rank : up to " << maxThreads() << "\n"
                  << "  repetitions  : " << reps << "  (warmup " << warmup << ")\n"
                  << "  signal       : " << signalLen << " samples, "
                  << frames << " frames\n";
        bench::printSectionTitle(
            "Distributed STFT (MPI_STFTAnalyzer<IterativeFFT, HannWindow>)");
        bench::printTableHeader();
    }

    auto runOnce = [&]{
        const SpectrogramData out = analyzer.analyze(signal);
        bench::doNotOptimize(out);
    };

    // MPI pure: 1 thread per rank
    setThreads(1);
    const bench::Stats sMPI = measureMPI(ctx, warmup, reps, runOnce);
    if (ctx.isRoot())
        bench::printRow("MPI pure (1 thr/rank)", frames, sMPI);

    // Hybrid: all threads per rank
    setThreads(maxThreads());
    const bench::Stats sHybrid = measureMPI(ctx, warmup, reps, runOnce);
    if (ctx.isRoot())
        bench::printRow("hybrid (" + std::to_string(maxThreads()) + " thr/rank)",
                        frames, sHybrid, sMPI.mean);

    if (ctx.isRoot()) {
        std::cout << "\nNotes:\n"
                  << "  - time per rep = slowest rank (MPI_MAX).\n"
                  << "  - speedup is hybrid vs MPI-pure at the SAME rank count.\n"
                  << "  - vary -np across runs to read strong scaling;\n"
                  << "    -np 1 is the serial/OpenMP-equivalent baseline.\n";
    }

    return 0;
}
