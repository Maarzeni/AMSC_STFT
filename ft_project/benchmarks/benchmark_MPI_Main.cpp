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
 * A second table reports the per-rank memory footprint under both input
 * distribution strategies (see Distribution in MPI_STFTAnalyzer.hpp), which is
 * what the broadcast-vs-scatter change is actually about.
 *
 * Usage:
 *   mpirun -np <P> ./benchmark_MPI_Main [reps] [warmup] [frame] [hop] [seconds] [strategy]
 *
 *   mpirun -np 2 ./benchmark_MPI_Main
 *   OMP_NUM_THREADS=4 mpirun -np 2 ./benchmark_MPI_Main            # hybrid
 *   mpirun -np 2 ./benchmark_MPI_Main 7 2 8192 4096 20             # coarser frames
 *   mpirun -np 2 ./benchmark_MPI_Main 7 2 1024 512 30 bcast        # old strategy
 *
 * `hop` may be given as 0, meaning frame/2.  `strategy` is `scatter` (default)
 * or `bcast`.  The rank count cannot be varied from inside the program — it is
 * fixed by mpirun at launch — so the strong scaling sweep lives in the
 * companion script run_scaling.sh.
 *
 * ─── Why one strategy per process ───────────────────────────────────────────
 * VmHWM is a high-water mark over the whole process lifetime: it never falls,
 * so a process that ran the broadcast first would carry that peak into the
 * scatter's measurement and report the two as equal.  No ordering fixes this
 * (the sweep contaminates it a second time, since a later workload raises the
 * mark recorded for every earlier one).  The measured peak is therefore
 * attributed to the single strategy the process was launched with, and the
 * before/after comparison is two runs:
 *
 *   mpirun -np 4 ./benchmark_MPI_Main 7 2 1024 512 30 bcast
 *   mpirun -np 4 ./benchmark_MPI_Main 7 2 1024 512 30 scatter
 *
 * The analytic footprint has no such problem — it is arithmetic, not a
 * measurement — so that table shows both strategies at every workload in a
 * single run.
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace stft;

namespace {

// ── Memory instrumentation ───────────────────────────────────────────────────

/**
 * @brief Peak resident set size of this process in KiB, 0 if not measurable.
 *
 * VmHWM is the kernel's high-water mark for RSS: the largest the process has
 * ever been resident, not its size right now.  That is the figure a footprint
 * comparison needs (a transient peak is what makes a job run out of memory),
 * and it is also why it cannot be attributed to more than one strategy per
 * process — see the note at the top of the file.
 *
 * /proc/self/status is Linux-only, which is where the cluster and the CI run.
 * Elsewhere this returns 0 and the table prints "n/a" rather than a wrong
 * number: the analytic section above it stays valid on every platform.
 */
double peakRssKiB() {
#if defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string   line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {          // "VmHWM:\t  123456 kB"
            std::istringstream in(line.substr(6));
            double kib = 0.0;
            in >> kib;
            return kib;
        }
    }
#endif
    return 0.0;
}

/// Spread of a per-rank quantity across the communicator (valid on root).
struct Spread {
    double min = 0.0;
    double max = 0.0;
    double avg = 0.0;
};

/**
 * @brief Reduce one per-rank value to MIN / MAX / AVG on the root rank.
 *
 * Collective: every rank must call it, only root gets meaningful results —
 * the same convention as measureMPI() below.
 */
Spread reduceSpread(const MPIContext& ctx, double local) {
    Spread s;
    double sum = 0.0;
    MPI_Reduce(&local, &s.min, 1, MPI_DOUBLE, MPI_MIN, MPIContext::root, MPI_COMM_WORLD);
    MPI_Reduce(&local, &s.max, 1, MPI_DOUBLE, MPI_MAX, MPIContext::root, MPI_COMM_WORLD);
    MPI_Reduce(&local, &sum,   1, MPI_DOUBLE, MPI_SUM, MPIContext::root, MPI_COMM_WORLD);
    s.avg = sum / static_cast<double>(ctx.size());
    return s;
}

// ── Printing (same widths and style as the bench:: table helpers) ────────────

void printMemoryHeader() {
    std::cout << std::left  << std::setw(26) << "Distribution @ workload"
              << std::right << std::setw(12) << "Frames"
              << std::setw(14) << "samples MIN"
              << std::setw(14) << "samples MAX"
              << std::setw(14) << "samples AVG"
              << std::setw(14) << "MiB AVG" << "\n"
              << std::string(94, '-') << "\n";
}

void printMemoryRow(const std::string& label,
                    std::size_t        frames,
                    const Spread&      samples) {
    constexpr double kSampleMiB = sizeof(double) / (1024.0 * 1024.0);
    std::cout << std::left  << std::setw(26) << label
              << std::right << std::setw(12) << frames
              << std::fixed << std::setprecision(0)
              << std::setw(14) << samples.min
              << std::setw(14) << samples.max
              << std::setw(14) << samples.avg
              << std::setprecision(2)
              << std::setw(14) << (samples.avg * kSampleMiB) << "\n";
}

void printPeakRss(const std::string& label, const Spread& kib, bool measurable) {
    std::cout << std::left << std::setw(26) << label;
    if (!measurable) {
        std::cout << std::right << std::setw(14) << "n/a"
                  << std::setw(14) << "n/a" << std::setw(14) << "n/a" << "\n";
        return;
    }
    std::cout << std::right << std::fixed << std::setprecision(2)
              << std::setw(14) << (kib.min / 1024.0)
              << std::setw(14) << (kib.max / 1024.0)
              << std::setw(14) << (kib.avg / 1024.0) << "\n";
}

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

    // Optional geometry overrides.  The frame size is the knob that matters for
    // parallel efficiency: the per-iteration overhead of the OpenMP loop is
    // fixed, while the work of one iteration grows as O(N log N), so larger
    // frames amortise it.  Pushing it too far leaves too few frames to spread
    // across ranks and threads, so both ends of the range are worth probing.
    std::size_t frame = 1024;
    std::size_t hop   = 0;        // 0 → frame / 2
    if (argc >= 4) frame = static_cast<std::size_t>(std::max(2, std::atoi(argv[3])));
    if (argc >= 5) hop   = static_cast<std::size_t>(std::max(0, std::atoi(argv[4])));
    if (hop == 0) hop = frame / 2;

    // Workload sweep, expressed as seconds of audio.  A single measurement at
    // one problem size says nothing about scaling: what matters is how the
    // hybrid gain evolves as the number of frames grows, the same way the FFT
    // table sweeps transform sizes.  Passing a duration on the command line
    // collapses the sweep to that single point.
    std::vector<double> durations = {1.0, 5.0, 10.0, 30.0};
    if (argc >= 6) durations = { std::max(0.1, std::atof(argv[5])) };

    // Which input distribution this process runs.  One per process, because
    // the measured peak cannot be split between two of them (see file header).
    Distribution dist = Distribution::Scatter;
    if (argc >= 7) {
        const std::string s = argv[6];
        if      (s == "bcast" || s == "broadcast") dist = Distribution::Broadcast;
        else if (s == "scatter")                   dist = Distribution::Scatter;
        else {
            if (ctx.isRoot())
                std::cerr << "Unknown strategy '" << s
                          << "' (expected 'scatter' or 'bcast').\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const std::string distName =
        (dist == Distribution::Broadcast) ? "bcast" : "scatter";

    constexpr std::uint32_t SR = 44100;

    // Checked here rather than left to the analyzer's exception: every rank
    // would throw at once, and an uncaught throw under MPI gives a far less
    // readable failure than an explicit abort.
    if ((frame & (frame - 1)) != 0) {
        if (ctx.isRoot())
            std::cerr << "Frame size must be a power of two (got " << frame << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Frame geometry is fixed across the sweep, so one analyzer serves every
    // workload: only the signal handed to analyze() changes.
    MPI_STFTAnalyzer<IterativeFFT, HannWindow> analyzer(ctx, frame, hop, SR, dist);

    // Captured on every rank before clamping: omp_set_num_threads() writes the
    // same internal variable that omp_get_max_threads() reads back, so after
    // setThreads(1) the real per-rank core count is no longer readable.
    const int nThreads = maxThreads();

    if (ctx.isRoot()) {
        std::cout << "AMSC_STFT Benchmark Suite (MPI / hybrid)\n"
                  << "  ranks        : " << ctx.size() << "\n"
                  << "  threads/rank : up to " << nThreads << "\n"
                  << "  repetitions  : " << reps << "  (warmup " << warmup << ")\n"
                  << "  frame / hop  : " << frame << " / " << hop << " samples\n"
                  << "  distribution : " << distName << "\n"
                  << "  workloads    : ";
        for (std::size_t i = 0; i < durations.size(); ++i)
            std::cout << durations[i]
                      << (i + 1 < durations.size() ? ", " : " s of audio\n");
        bench::printSectionTitle(
            "Distributed STFT (MPI_STFTAnalyzer<IterativeFFT, HannWindow>)");
        bench::printTableHeader();
    }

    for (const double sec : durations) {
        const std::size_t signalLen =
            static_cast<std::size_t>(sec * static_cast<double>(SR));
        const std::size_t frames =
            STFTAnalyzer<IterativeFFT, HannWindow>::numFrames(signalLen, frame, hop);

        // A signal shorter than one frame produces no frames at all: there is
        // nothing to measure, and the row would report a meaningless 0.  Every
        // rank evaluates the same condition, so the collective stays balanced.
        if (frames == 0) continue;

        // Root generates the signal; non-root pass an empty vector (the analyzer
        // broadcasts internally).  Deterministic generator → reproducible.
        std::vector<double> signal;
        if (ctx.isRoot()) signal = bench::makeSignal(signalLen, SR);

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
        setThreads(nThreads);
        const bench::Stats sHybrid = measureMPI(ctx, warmup, reps, runOnce);
        if (ctx.isRoot()) {
            bench::printRow("hybrid (" + std::to_string(nThreads) + " thr/rank)",
                            frames, sHybrid, sMPI.mean);
            std::cout << "\n";
        }
    }

    // ── Memory: analytic ────────────────────────────────────────────────────
    // How many input samples each rank has to hold is exact arithmetic on the
    // frame layout, not a measurement, so both strategies can be reported side
    // by side from a single run without either disturbing the other.
    if (ctx.isRoot()) {
        bench::printSectionTitle("Per-rank input footprint (analytic)");
        printMemoryHeader();
    }

    for (const double sec : durations) {
        const std::size_t signalLen =
            static_cast<std::size_t>(sec * static_cast<double>(SR));
        const std::size_t frames =
            STFTAnalyzer<IterativeFFT, HannWindow>::numFrames(signalLen, frame, hop);

        if (frames == 0) continue;   // skipped above too, keep the tables aligned

        // Broadcast: the whole signal lands on every rank, whatever P is.
        const Spread bcastHeld = reduceSpread(ctx, static_cast<double>(signalLen));

        // Scatter: only the samples this rank's own frames read.  Asked of the
        // analyzer rather than recomputed here, so the table cannot drift away
        // from the layout the scatter actually uses (the answer depends on the
        // frame geometry, not on which strategy the analyzer was built with).
        const Spread scatterHeld = reduceSpread(ctx,
            static_cast<double>(analyzer.localSampleCount(signalLen, ctx.rank())));

        if (ctx.isRoot()) {
            std::ostringstream w;
            w << std::fixed << std::setprecision(1) << sec << " s";
            printMemoryRow("bcast   @ " + w.str(), frames, bcastHeld);
            printMemoryRow("scatter @ " + w.str(), frames, scatterHeld);
        }
    }

    // ── Memory: measured ────────────────────────────────────────────────────
    // One process, one strategy, one high-water mark.  Reported once for the
    // whole run: VmHWM cannot be attributed to an individual workload either,
    // since the mark left by the largest one covers all of them.
    const double myPeakKiB = peakRssKiB();
    const Spread peak       = reduceSpread(ctx, myPeakKiB);
    const Spread measurable = reduceSpread(ctx, myPeakKiB > 0.0 ? 1.0 : 0.0);

    if (ctx.isRoot()) {
        bench::printSectionTitle("Peak resident memory per rank (VmHWM, whole process)");
        std::cout << std::left  << std::setw(26) << "Metric"
                  << std::right << std::setw(14) << "MIN(MiB)"
                  << std::setw(14) << "MAX(MiB)"
                  << std::setw(14) << "AVG(MiB)" << "\n"
                  << std::string(68, '-') << "\n";
        printPeakRss("peak RSS (" + distName + ")", peak, measurable.min > 0.0);
    }

    if (ctx.isRoot()) {
        std::cout << "\nNotes:\n"
                  << "  - time per rep = slowest rank (MPI_MAX).\n"
                  << "  - speedup is hybrid vs MPI-pure at the SAME workload.\n"
                  << "  - each block is one workload; the frame count grows down\n"
                  << "    the table, so the hybrid gain can be read against it.\n"
                  << "  - vary -np across runs to read strong scaling;\n"
                  << "    -np 1 is the serial/OpenMP-equivalent baseline.\n"
                  << "  - args: [reps] [warmup] [frame] [hop] [seconds] [strategy];\n"
                  << "    giving [seconds] replaces the sweep with one workload.\n"
                  << "  - the analytic table counts what the distribution forces a\n"
                  << "    rank to hold.  Root also owns the source signal it read,\n"
                  << "    under either strategy: the scatter changes what the other\n"
                  << "    P-1 ranks must hold, not what the reader holds.  That is\n"
                  << "    why the measured MAX barely moves and MIN/AVG do.\n"
                  << "  - peak RSS covers the whole process, so it belongs to the\n"
                  << "    one strategy this run used.  For before/after, run twice:\n"
                  << "      mpirun -np <P> ./benchmark_MPI_Main <reps> <warmup> "
                  << frame << " " << hop << " 30 bcast\n"
                  << "      mpirun -np <P> ./benchmark_MPI_Main <reps> <warmup> "
                  << frame << " " << hop << " 30 scatter\n"
                  << "  - VmHWM is Linux-only; elsewhere the row reads n/a and only\n"
                  << "    the analytic table is available.\n";
    }

    return 0;
}
