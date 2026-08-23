/**
 * @file benchmark_MPI_Main.cpp
 * @brief Distributed STFT benchmark entry point (MPI and hybrid MPI+OpenMP).
 *
 * @details
 * Benchmarks MPI_STFTAnalyzer<IterativeFFT, HannWindow> on a long signal,
 * comparing three configurations measured in the SAME run (section "mpi",
 * kind = "timing" — see the schema documented in benchmark_Suite.hpp):
 *
 *   - serial     : root rank only, 1 thread, no distribution — the baseline
 *   - MPI pure   : 1 OpenMP thread per rank
 *   - hybrid     : all available OpenMP threads per rank
 *
 * Both parallel rows report their speedup over the serial row of the same
 * workload, so `speedup` is a parallel speedup in the usual sense and
 * dividing it by the number of cores in use (ranks x threads) gives the
 * efficiency directly. The gain of the threads over the processes is the
 * ratio of the two speedups.
 *
 * Wall-clock time per repetition is the time of the SLOWEST rank
 * (MPI_Reduce with MPI_MAX), since that is what bounds the collective.
 * Only the root rank writes rows.
 *
 * A fourth row pair (section "mpi_granularity") answers, for the distributed
 * case, the question benchmark_Main asks for the shared-memory one: given the
 * SAME ranks and the SAME threads per rank, is a rank better off spending its
 * threads across its own frames or inside each transform?  MPI splits the
 * frames either way; only the intra-rank level changes, so the two rows differ
 * in one variable and share the serial baseline of the rows above.
 *
 * Two more row kinds report the per-rank memory footprint (see Distribution
 * in MPI_STFTAnalyzer.hpp), which is what the broadcast-vs-scatter change is
 * actually about:
 *
 *   - memory_analytic : how many input samples one rank holds under EACH
 *                        strategy, at every workload — exact arithmetic, so
 *                        both strategies come out of a single run.
 *   - memory_rss       : measured peak resident memory (VmHWM), one row for
 *                        the whole run, belonging to the single strategy
 *                        this process was launched with (see below for why).
 *
 * Usage:
 *   mpirun -np <P> ./benchmark_MPI_Main [reps] [warmup] [frame] [hop] [seconds] [strategy]
 *
 *   mpirun -np 2 ./benchmark_MPI_Main
 *   OMP_NUM_THREADS=4 mpirun -np 2 ./benchmark_MPI_Main            # hybrid
 *   mpirun -np 2 ./benchmark_MPI_Main 7 2 8192 4096 20             # coarser frames
 *   mpirun -np 2 ./benchmark_MPI_Main 7 2 1024 512 30 bcast        # broadcast strategy
 *
 * `hop` may be given as 0, meaning frame/2.  `strategy` is `scatter` (default)
 * or `bcast`.  The rank count cannot be varied from inside the program — it is
 * fixed by mpirun at launch — so the strong scaling sweep lives in the
 * companion script scripts/run_suite.sh, which launches this binary once per
 * rank count.
 *
 * Output is CSV on stdout, nothing else — see benchmark_Suite.hpp.
 *
 * @par Why one strategy per process
 * VmHWM is a high-water mark over the whole process lifetime: it never
 * falls, so a process that ran the broadcast first would carry that peak
 * into the scatter's measurement and report the two as equal. The measured
 * peak is therefore attributed to the single strategy the process was
 * launched with; comparing the two takes two runs:
 *
 *   mpirun -np 4 ./benchmark_MPI_Main 7 2 1024 512 30 bcast
 *   mpirun -np 4 ./benchmark_MPI_Main 7 2 1024 512 30 scatter
 *
 * The analytic footprint has no such problem — it is arithmetic, not a
 * measurement — so that row kind reports both strategies at every workload
 * in a single run.
 *
 * Only std::chrono + STL + project code (+ MPI) are used.
 */

#include "benchmark_Suite.hpp"

#include "mpi/MPIContext.hpp"
#include "stft/MPI_STFTAnalyzer.hpp"
#include "stft/STFTAnalyzer.hpp"
#include "fft/IterativeFFT.hpp"
#include "fft/ParallelFFT.hpp"
#include "window/HannWindow.hpp"

#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace stft;

namespace {

// ─── Memory instrumentation ───────────────────────────────────────────────────

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
 * Elsewhere this returns 0 and the memory_rss row is written with its MiB
 * fields blank rather than a wrong number.
 */
double peakRssKiB() {
#if defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string   line;
    while (std::getline(status, line)) {
        if (line.starts_with("VmHWM:")) {             // "VmHWM:\t  123456 kB"
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
    double min = 0.0;  ///< smallest per-rank value
    double max = 0.0;  ///< largest per-rank value
    double avg = 0.0;  ///< mean per-rank value
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

/// @brief Writes one `kind = "memory_analytic"` row (see the schema in
///        benchmark_Suite.hpp).
void writeAnalyticRow(const std::string& strategy, int ranks,
                      std::size_t frames, double workloadS,
                      const Spread& samplesHeld) {
    constexpr double kSampleMiB = sizeof(double) / (1024.0 * 1024.0);
    bench::Row r;
    r.kind = "memory_analytic";
    r.strategy = strategy;
    r.ranks = ranks;
    r.frames = static_cast<long long>(frames);
    r.workloadS = workloadS;
    r.minMib = samplesHeld.min * kSampleMiB;
    r.maxMib = samplesHeld.max * kSampleMiB;
    r.avgMib = samplesHeld.avg * kSampleMiB;
    bench::writeRow(std::cout, r);
}

/// @brief Writes one `kind = "memory_rss"` row (see the schema in
///        benchmark_Suite.hpp).
void writeRssRow(const std::string& strategy, int ranks,
                 const Spread& kib, bool measurable) {
    bench::Row r;
    r.kind = "memory_rss";
    r.strategy = strategy;
    r.ranks = ranks;
    if (measurable) {
        r.minMib = kib.min / 1024.0;
        r.maxMib = kib.max / 1024.0;
        r.avgMib = kib.avg / 1024.0;
    }
    bench::writeRow(std::cout, r);
}

/// @brief OpenMP's current max thread count, or 1 if built without OpenMP.
int maxThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/// @brief Sets OpenMP's thread count; a no-op if built without OpenMP.
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
    // section sweeps transform sizes.  Passing a duration on the command line
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

    // Same engine the analyzer composes, driven directly on the root rank: the
    // serial baseline of the rows below. Constructed once, like the analyzer,
    // since only the signal handed to it changes across the sweep.
    STFTAnalyzer<IterativeFFT, HannWindow> serialEngine(frame, hop, SR);

    // Captured on every rank before clamping: omp_set_num_threads() writes the
    // same internal variable that omp_get_max_threads() reads back, so after
    // setThreads(1) the real per-rank core count is no longer readable.
    const int nThreads = maxThreads();

    // Same ranks, same distribution, same threads per rank — spent inside each
    // transform instead of across the rank's own frames. Declared here rather
    // than beside `analyzer` because the factory has to capture nThreads by
    // value: ParallelFFT would otherwise read the ambient count when built,
    // and the setThreads(1) below would collapse it to serial along with the
    // frame loop instead of leaving it parallel independently of it.
    // The same two ceilings benchmark_Main applies to its own granularity pair,
    // and for the same reason: ParallelFFT splits ONE 1024-point transform over
    // its threads, and past a handful of them the synchronisation costs more
    // than the transform — measured at 17.8x slower than IterativeFFT on 16
    // threads. Under a rank/thread sweep that reaches 24 or 48 threads per rank
    // the arm stops being a measurement and becomes the longest thing in the
    // job, which is how a rehearsal on MOX ran for four hours.
    //
    // Until now these variables were read only by benchmark_Main, so a caller
    // that set AMSC_SKIP_GRANULARITY was silently obeyed in one binary and
    // ignored in the other.
    const bool withGranularity = std::getenv("AMSC_SKIP_GRANULARITY") == nullptr;
    const int granThreads = [nThreads] {
        const char* env = std::getenv("AMSC_GRANULARITY_THREADS");
        const int wanted = env ? std::atoi(env) : 8;
        return std::max(1, std::min(nThreads, wanted));
    }();

    MPI_STFTAnalyzer<ParallelFFT, HannWindow> transformAnalyzer(
        ctx, frame, hop, SR,
        [n = static_cast<std::size_t>(granThreads)] { return ParallelFFT(n); },
        Parallelism::Transform, dist);

    if (ctx.isRoot()) bench::writeHeader(std::cout);

    // ─── Distributed STFT: serial / MPI pure / hybrid ──────────────────────
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

        // Serial reference: the root rank alone, one thread, no distribution at
        // all. The other ranks fall through the lambda and wait in the barrier
        // that measureMPI puts before each timed region, so the MPI_MAX over
        // the per-rank times is root's serial time. Measured here, in the same
        // binary and run as the rows compared against it, so `speedup` is an
        // honest ratio rather than a comparison against a separate program.
        auto runSerial = [&]{
            if (!ctx.isRoot()) return;
            const SpectrogramData out = serialEngine.analyze(signal);
            bench::doNotOptimize(out);
        };

        // 1 thread per rank for both of the next two rows.
        setThreads(1);

        const bench::Stats sSerial = measureMPI(ctx, warmup, reps, runSerial);
        if (ctx.isRoot())
            bench::writeTiming(std::cout, "mpi", "serial (1 rank, 1 thr)", 1, 1,
                               nThreads, std::nullopt,
                               static_cast<long long>(frames), sec, sSerial);

        // MPI pure: 1 thread per rank
        const bench::Stats sMPI = measureMPI(ctx, warmup, reps, runOnce);
        if (ctx.isRoot())
            bench::writeTiming(std::cout, "mpi", "MPI pure (1 thr/rank)", ctx.size(),
                               1, nThreads, std::nullopt,
                               static_cast<long long>(frames), sec, sMPI, sSerial.mean);

        // Hybrid: all threads per rank
        setThreads(nThreads);
        const bench::Stats sHybrid = measureMPI(ctx, warmup, reps, runOnce);
        if (ctx.isRoot())
            bench::writeTiming(std::cout, "mpi",
                               "hybrid (" + std::to_string(nThreads) + " thr/rank)",
                               ctx.size(), nThreads, nThreads, std::nullopt,
                               static_cast<long long>(frames), sec, sHybrid, sSerial.mean);

        // ── Granularity: where a rank spends its own threads ───────────────
        // sHybrid above already is the "across frames" arm — the analyzer's
        // default — so only the "inside the transform" arm is measured here,
        // and both are written as a pair against the same serial baseline.
        // setThreads() does not reach this one: under Parallelism::Transform
        // the frame loop has no OpenMP region and the engine's thread count
        // was fixed when the factory captured it.
        if (withGranularity) {
        const bench::Stats sTransform =
            measureMPI(ctx, warmup, reps, [&] {
                const SpectrogramData out = transformAnalyzer.analyze(signal);
                bench::doNotOptimize(out);
            });
        if (ctx.isRoot()) {
            bench::writeTiming(std::cout, "mpi_granularity",
                               "OpenMP frames + IterativeFFT (" +
                                   std::to_string(nThreads) + " thr/rank)",
                               ctx.size(), nThreads, nThreads, std::nullopt,
                               static_cast<long long>(frames), sec, sHybrid,
                               sSerial.mean);
            // granThreads, not nThreads: when the ceiling bites, this arm ran
            // with fewer threads than the one above it and the label has to say
            // so — otherwise the pair reads as a like-for-like comparison that
            // it no longer is. Set AMSC_GRANULARITY_THREADS at or above the
            // sweep's thread count to keep the two arms comparable.
            bench::writeTiming(std::cout, "mpi_granularity",
                               "sequential frames + ParallelFFT (" +
                                   std::to_string(granThreads) + " thr/rank)",
                               ctx.size(), granThreads, granThreads, std::nullopt,
                               static_cast<long long>(frames), sec, sTransform,
                               sSerial.mean);
        }
        }
    }

    // ─── Phase split: where the wall-clock time actually goes ──────────────
    // The scaling curves show speedup peaking around 24 ranks and falling
    // beyond it, which a fitted model attributes to a term proportional to the
    // rank count. A fit cannot say WHICH phase produces that term, and the two
    // candidates behave differently: the gather moves a volume fixed by the
    // problem size, while every root-centric collective also pays a per-message
    // cost that grows with the number of ranks. Measuring the phases separates
    // them — at fixed problem size, a cost that grows with P is latency, not
    // volume.
    //
    // Reported as MPI_MAX over the ranks, matching measureMPI's convention:
    // the collective is bounded by its slowest participant.
    //
    // Measured once per thread count, not once overall: the pure-MPI and
    // hybrid rows above answer a different question each, and only writing
    // both phase splits explicitly — each carrying its own thread count —
    // lets a row be matched to the scaling row it explains.
    std::vector<int> phaseThreadCounts = {1};
    if (nThreads > 1) phaseThreadCounts.push_back(nThreads);

    for (const int phaseThreads : phaseThreadCounts) {
    setThreads(phaseThreads);
    for (const double sec : durations) {
        const std::size_t signalLen =
            static_cast<std::size_t>(sec * static_cast<double>(SR));
        const std::size_t frames =
            STFTAnalyzer<IterativeFFT, HannWindow>::numFrames(signalLen, frame, hop);
        if (frames == 0) continue;

        std::vector<double> signal;
        if (ctx.isRoot()) signal = bench::makeSignal(signalLen, SR);

        // One warm-up call so first-touch page faults and the connection setup
        // of the collectives are not charged to the phase they happen to land
        // in; then the phases of a single settled call are read back.
        ctx.barrier();
        { const SpectrogramData warm = analyzer.analyze(signal);
          bench::doNotOptimize(warm); }

        ctx.barrier();
        { const SpectrogramData out = analyzer.analyze(signal);
          bench::doNotOptimize(out); }

        const auto& ph = analyzer.phaseTimes();
        const Spread dist_    = reduceSpread(ctx, ph.distributeMs);
        const Spread comp_    = reduceSpread(ctx, ph.computeMs);
        const Spread gath_    = reduceSpread(ctx, ph.gatherMs);

        if (ctx.isRoot()) {
            const auto row = [&](const std::string& name, const Spread& sp) {
                bench::Row r;
                r.kind      = "timing";
                r.section   = "mpi_phases";
                r.variant   = name;
                r.ranks     = ctx.size();
                r.threads   = phaseThreads;
                r.frames    = static_cast<long long>(frames);
                r.workloadS = sec;
                // min_ms carries the SLOWEST rank, matching measureMPI's
                // convention elsewhere in this file: a collective is bounded by
                // its slowest participant, so that is the critical path and the
                // number comparable with the scaling rows. mean_ms next to it
                // exposes imbalance — the wider the gap, the more of the phase
                // is ranks waiting rather than working.
                r.minMs     = sp.max;
                r.meanMs    = sp.avg;
                bench::writeRow(std::cout, r);
            };
            const std::string suffix =
                " (" + std::to_string(phaseThreads) + " thr/rank)";
            row("distribute (scatter/bcast of input)" + suffix, dist_);
            row("compute (local STFT)"                + suffix, comp_);
            row("gather (spectrogram to root)"        + suffix, gath_);
        }
    }
    }
    setThreads(nThreads);

    // ── Memory: analytic ────────────────────────────────────────────────────
    // How many input samples each rank has to hold is exact arithmetic on the
    // frame layout, not a measurement, so both strategies can be reported side
    // by side from a single run without either disturbing the other.
    for (const double sec : durations) {
        const std::size_t signalLen =
            static_cast<std::size_t>(sec * static_cast<double>(SR));
        const std::size_t frames =
            STFTAnalyzer<IterativeFFT, HannWindow>::numFrames(signalLen, frame, hop);

        if (frames == 0) continue;   // skipped above too, for the same reason

        // Broadcast: the whole signal lands on every rank, whatever P is.
        const Spread bcastHeld = reduceSpread(ctx, static_cast<double>(signalLen));

        // Scatter: only the samples this rank's own frames read.  Asked of the
        // analyzer rather than recomputed here, so the row cannot drift away
        // from the layout the scatter actually uses (the answer depends on the
        // frame geometry, not on which strategy the analyzer was built with).
        const Spread scatterHeld = reduceSpread(ctx,
            static_cast<double>(analyzer.localSampleCount(signalLen, ctx.rank())));

        if (ctx.isRoot()) {
            writeAnalyticRow("bcast",   ctx.size(), frames, sec, bcastHeld);
            writeAnalyticRow("scatter", ctx.size(), frames, sec, scatterHeld);
        }
    }

    // ── Memory: measured ────────────────────────────────────────────────────
    // One process, one strategy, one high-water mark. Reported once for the
    // whole run: VmHWM cannot be attributed to an individual workload either,
    // since the mark left by the largest one covers all of them.
    const double myPeakKiB = peakRssKiB();
    const Spread peak       = reduceSpread(ctx, myPeakKiB);
    const Spread measurable = reduceSpread(ctx, myPeakKiB > 0.0 ? 1.0 : 0.0);

    if (ctx.isRoot())
        writeRssRow(distName, ctx.size(), peak, measurable.min > 0.0);

    return 0;
}
