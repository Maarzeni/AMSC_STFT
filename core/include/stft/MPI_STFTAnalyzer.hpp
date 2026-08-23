/**
 * @file MPI_STFTAnalyzer.hpp
 * @brief Distributed STFT using MPI process-level parallelism.
 *
 * @details
 * MPI_STFTAnalyzer<FFT, Window> computes the STFT of an audio signal by
 * distributing time frames across MPI ranks and composing STFTAnalyzer (which
 * parallelises within a rank using OpenMP) for a hybrid MPI+OpenMP strategy.
 *
 * ─── Distribution algorithm ──────────────────────────────────────────────────
 *
 * 1. Root rank holds the signal.  It broadcasts the signal length and sample
 *    rate so every rank can compute the frame layout without communication.
 * 2. Frames are block-distributed: rank r processes frames [start_r, start_r +
 *    count_r).  The first `rem` ranks get one extra frame when totalFrames is
 *    not divisible by P.
 * 3. The sample data is moved to the ranks, in one of two ways (see
 *    Distribution below): either the whole signal is broadcast, or each rank
 *    receives only the samples its own frames read.
 * 4. Each rank calls STFTAnalyzer::analyzeRange() on its block.  The per-rank
 *    magnitudes are a contiguous slice of the final spectrogram buffer.
 * 5. MPI_Gatherv assembles the full magnitude matrix on the root rank.
 *    Non-root ranks receive an empty SpectrogramData.  The magnitudes travel
 *    as float (stft::Magnitude), half the bytes the double matrix would cost.
 *
 * ─── Memory scalability ─────────────────────────────────────────────────────
 * Step 3 is what decides whether the algorithm weak-scales.  Broadcasting the
 * whole signal leaves every rank holding O(N) samples no matter how many ranks
 * are added: more nodes buy time, not capacity.  The scatter distributes the
 * input as well as the work, so the per-rank input footprint falls as O(N/P).
 * Distribution::Scatter is the default; Broadcast is kept for comparison.
 *
 * The gather side is deliberately unchanged in SHAPE: root still assembles the
 * complete totalFrames x numBins matrix, which at typical geometries is as
 * large as the input signal itself.  The input bottleneck is removed, the
 * output one is not — storing the matrix as float halves what it costs, but a
 * volume that does not fall with P still bounds the speedup, and measurement
 * shows it is what makes the strong-scaling curve turn over.  Removing it means
 * not centralising the output at all (parallel write), which is a change to
 * what the algorithm produces, not to how it produces it.
 *
 * ─── Hybrid MPI + OpenMP ────────────────────────────────────────────────────
 * By default STFTAnalyzer uses `#pragma omp parallel for` over its frame block,
 * so each MPI rank automatically uses all available cores.  Recommended usage:
 *   mpirun -np `<nodes>` --map-by node --bind-to none ./mpi_main
 *   OMP_NUM_THREADS=`<cores_per_node>` mpirun ...
 *
 * Passing Parallelism::Transform (plus a parallel engine, see the FFT factory
 * constructors below) moves the intra-rank threads inside each transform
 * instead: MPI still distributes the frames, but a rank's cores cooperate on
 * one frame at a time rather than on several frames at once.
 *
 * @tparam FFT     Any type satisfying IsFFT<FFT>           (e.g. IterativeFFT)
 * @tparam Window  Any type satisfying WindowFunction<Window> (e.g. HannWindow)
 */

#pragma once

#include "mpi/MPIContext.hpp"
#include "stft/STFTAnalyzer.hpp"
#include "stft/SpectrogramData.hpp"
#include "fft/BaseFFT.hpp"
#include "window/BaseWindow.hpp"

#include <mpi.h>
#include <vector>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace stft {

/**
 * @brief The MPI datatype of one spectrogram magnitude.
 *
 * Kept next to the only collective that transfers magnitudes, and asserted
 * against the element type it must match: a datatype that disagrees with the
 * buffer it describes is not a compile error, it is silent corruption of half
 * the spectrogram.  Change stft::Magnitude and this fails to compile.
 */
inline const MPI_Datatype kMagnitudeType = MPI_FLOAT;
static_assert(std::is_same_v<Magnitude, float>,
              "kMagnitudeType must be updated to match stft::Magnitude.");

/**
 * @brief How the input samples are moved from root to the other ranks.
 *
 * Broadcast: every rank receives the entire signal (O(N) samples per rank).
 *            Simple, one collective, but the per-rank footprint is independent
 *            of the rank count, so the largest signal that fits is set by the
 *            memory of a single node.
 * Scatter:   every rank receives only the samples covered by its own frames,
 *            roughly N/P plus a halo of (frameSize - hopSize) shared with the
 *            next block (O(N/P) samples per rank).
 *
 * Declared at namespace scope so call sites can name it without repeating the
 * analyzer's template arguments.
 */
enum class Distribution {
    Broadcast,  ///< MPI_Bcast the whole signal to every rank
    Scatter     ///< MPI_Scatterv only the samples each rank's frames read
};

/// Distributed STFT using MPI process-level parallelism.
template<typename FFT, typename Window>
    requires IsFFT<FFT> && WindowFunction<Window>
class MPI_STFTAnalyzer {
public:
    /// Same factory type the local engine uses; see STFTAnalyzer::FFTFactory.
    using FFTFactory = typename STFTAnalyzer<FFT, Window>::FFTFactory;

    // ── Construction ──────────────────────────────────────────────────────────

    /**
     * @param ctx        Live MPIContext (must remain alive for the duration).
     * @param frameSize  STFT frame size (power of two, >= 2).
     * @param hopSize    Hop between successive frames (>= 1).
     * @param sampleRate Audio sample rate in Hz (informational).
     * @param dist       Input distribution strategy (see Distribution).
     * @param mode       Intra-rank parallelism (see Parallelism).  Trails dist
     *                   so that existing positional calls keep their meaning.
     */
    MPI_STFTAnalyzer(const MPIContext& ctx,
                     std::size_t frameSize,
                     std::size_t hopSize,
                     std::uint32_t sampleRate = 0,
                     Distribution dist = Distribution::Scatter,
                     Parallelism mode = Parallelism::Frames)
        : ctx_(ctx),
          engine_(frameSize, hopSize, sampleRate, mode),
          dist_(dist)
    {}

    /**
     * @brief As above, with the local engines built by a caller-supplied factory.
     *
     * Distribution and Parallelism are independent: the first says how the
     * samples reach the ranks, the second how each rank spends its cores once
     * they arrive.  Parallelism::Transform gives a third hybrid shape — MPI
     * across the frames, OpenMP inside each transform — next to the usual
     * Parallelism::Frames, where both levels work across frames.
     */
    MPI_STFTAnalyzer(const MPIContext& ctx,
                     std::size_t frameSize,
                     std::size_t hopSize,
                     std::uint32_t sampleRate,
                     FFTFactory factory,
                     Parallelism mode = Parallelism::Frames,
                     Distribution dist = Distribution::Scatter)
        : ctx_(ctx),
          engine_(frameSize, hopSize, sampleRate, std::move(factory), mode),
          dist_(dist)
    {}

    /// As above, from an already-constructed engine used as a prototype.
    MPI_STFTAnalyzer(const MPIContext& ctx,
                     std::size_t frameSize,
                     std::size_t hopSize,
                     std::uint32_t sampleRate,
                     FFT prototype,
                     Parallelism mode = Parallelism::Transform,
                     Distribution dist = Distribution::Scatter)
        requires std::copy_constructible<FFT>
        : ctx_(ctx),
          engine_(frameSize, hopSize, sampleRate, std::move(prototype), mode),
          dist_(dist)
    {}

    /// @return How the input samples are moved to the ranks (see Distribution).
    [[nodiscard]] Distribution distribution() const noexcept { return dist_; }

    /// @return Where the local engine spends its thread budget (see Parallelism).
    [[nodiscard]] Parallelism  parallelism()  const noexcept {
        return engine_.parallelism();
    }

    /**
     * @brief Size of rank `r`'s sample block, i.e. what Scatter sends it.
     *
     * Exposed so that benchmarks can report the per-rank input footprint from
     * the same formula the scatter itself uses, instead of duplicating it.
     * Under Broadcast the answer is always `signalLen`, on every rank.  Root is
     * a special case under Scatter too: it reads its block in place inside the
     * signal it already owns, so this is the size of the block it computes on,
     * not the amount of memory it holds.
     *
     * @return (count_r - 1) * hopSize + frameSize, clamped to the signal end,
     *         or 0 for a rank that owns no frames.
     */
    [[nodiscard]] std::size_t
    localSampleCount(std::size_t signalLen, int r) const noexcept
    {
        const Layout lay = layout(signalLen);
        return lay.samples(r).count;
    }

    // ── Analysis ──────────────────────────────────────────────────────────────

    /**
     * @brief Wall-clock split of the last analyze() call, in milliseconds.
     *
     * Measured per rank, without extra barriers: a barrier would move waiting
     * time out of the phase that causes it and into an artificial one, which
     * is precisely the attribution this is meant to establish. Time a rank
     * spends blocked inside a collective therefore counts against that
     * collective, which is what "where does the time go" should mean.
     *
     * Only meaningful after analyze() has returned. The benchmark reduces
     * these with MPI_MAX to get the critical path.
     */
    struct PhaseTimes {
        double distributeMs = 0.0;   ///< Step 3: broadcast or scatter of the input
        double computeMs    = 0.0;   ///< Step 4: local STFT over this rank's frames
        double gatherMs     = 0.0;   ///< Step 5: gather of the magnitudes on root
    };

    /// @brief Phase split of the most recent analyze() call on this rank.
    [[nodiscard]] const PhaseTimes& phaseTimes() const noexcept { return phases_; }

    /**
     * @brief Distributed STFT.  All ranks must call this collectively.
     *
     * @param signalOnRoot  On rank 0: the full audio signal, read in place and
     *                      never modified.
     *                      On other ranks: unused — pass an empty vector.  The
     *                      incoming samples land in a buffer this call owns, so
     *                      whatever a non-root rank passes is neither read nor
     *                      written.
     *
     * Taken by const reference, which is the difference between distributing a
     * signal and copying it first.  A by-value parameter would force root to
     * copy the whole signal on every call before distributing it — at ten
     * minutes of 44.1 kHz audio, 106 MiB allocated and memcpy'd before a single
     * byte moves between ranks — and since only root would pay that cost, every
     * other rank would sit idle waiting for it inside the distribute phase.
     * Both distribution paths instead write into a buffer owned here (see
     * below), so root can read the caller's signal exactly where it already is.
     * @return On rank 0: a fully populated SpectrogramData.
     *         On other ranks: an empty SpectrogramData (numFrames == 0).
     */
    [[nodiscard]] SpectrogramData
    analyze(const std::vector<double>& signalOnRoot) const
    {
        const std::size_t frameSize = engine_.frameSize();
        const std::size_t numBins   = frameSize / 2 + 1;
        const int P                 = ctx_.size();
        const int me                = ctx_.rank();

        // ── Step 1: broadcast signal length + sample rate ──────────────────
        // Two scalars, negligible next to the sample data, and they let every
        // rank derive the frame layout locally — which both strategies need.
        unsigned long signalLen  = (me == MPIContext::root)
                                   ? static_cast<unsigned long>(signalOnRoot.size())
                                   : 0UL;
        unsigned int  sampleRate = (me == MPIContext::root)
                                   ? static_cast<unsigned int>(engine_.sampleRate())
                                   : 0U;

        MPI_Bcast(&signalLen,  1, MPI_UNSIGNED_LONG, MPIContext::root, MPI_COMM_WORLD);
        MPI_Bcast(&sampleRate, 1, MPI_UNSIGNED,      MPIContext::root, MPI_COMM_WORLD);

        // ── Step 2: compute frame distribution ────────────────────────────
        const Layout      lay         = layout(static_cast<std::size_t>(signalLen));
        const std::size_t totalFrames = lay.totalFrames;
        const std::size_t myCount     = lay.frames(me);
        const std::size_t myStart     = lay.start(me);

        // ── Step 3: move the sample data to the ranks ─────────────────────
        // Both branches produce (localSignal, localStart) such that local frame
        // localStart + fi is global frame myStart + fi.
        //
        // Root never allocates here under either strategy: it already holds the
        // signal and reads its own block in place.  Only the other ranks need
        // somewhere to receive into, and this is it — the whole signal under
        // Broadcast, this rank's slice under Scatter.
        std::vector<double> incoming;        // non-root ranks only
        std::size_t         localStart = 0;

        const double tDistribute0 = MPI_Wtime();

        if (dist_ == Distribution::Broadcast) {
            // MPI_Bcast's buffer is a single INOUT parameter, so the C binding
            // cannot express "read-only on root" and there is no const overload
            // to reach for.  On root the call only ever reads it — root is the
            // source — so casting away const here is sound.  Non-root ranks
            // broadcast into a buffer of their own, never the caller's.
            double* buf;
            if (me == MPIContext::root) {
                buf = const_cast<double*>(signalOnRoot.data());
            } else {
                incoming.resize(signalLen, 0.0);
                buf = incoming.data();
            }

            MPI_Bcast(buf,
                      static_cast<int>(signalLen),
                      MPI_DOUBLE,
                      MPIContext::root,
                      MPI_COMM_WORLD);

            localStart = myStart;         // indices stay global on every rank
        } else {
            // sendcounts / displs are in samples and are only read on root.
            // Successive blocks overlap by (frameSize - hopSize) samples: the
            // halo the next block needs to complete its first frame.  Scatterv
            // only reads the send buffer, so overlapping displacements are
            // fine here; the same overlap in Gatherv would be a write race.
            //
            // Both counts and displacements are `int`: above 2^31 elements
            // (~13.5 h of 44.1 kHz audio, on the Scatterv input as well as on
            // the Gatherv output below) they overflow.  Documented, not fixed —
            // lifting it means MPI_Type_create_struct or the MPI-4 _c variants.
            if (me == MPIContext::root) {
                std::vector<int> sendcounts(P, 0);
                std::vector<int> displs(P, 0);

                for (int r = 0; r < P; ++r) {
                    const auto s  = lay.samples(r);
                    sendcounts[r] = static_cast<int>(s.count);
                    displs[r]     = static_cast<int>(s.offset);
                }

                // MPI_IN_PLACE: root's own block stays where it already is, in
                // the signal it holds anyway.  Receiving it into a separate
                // slice would add N/P to the peak of the rank that is already
                // the largest — measurably so, ~50 MiB at P=4 on a 10-minute
                // signal — and buy nothing.  Root therefore keeps addressing
                // its frames globally, exactly as in the broadcast path.
                MPI_Scatterv(signalOnRoot.data(),
                             sendcounts.data(),
                             displs.data(),
                             MPI_DOUBLE,
                             MPI_IN_PLACE,
                             0,
                             MPI_DOUBLE,
                             MPIContext::root,
                             MPI_COMM_WORLD);

                localStart = myStart;
            } else {
                // The O(N/P) footprint this strategy exists for: a non-root rank
                // allocates its own slice and nothing else — there is no copy of
                // the caller's signal to hold onto in the meantime.
                const std::size_t myLen = lay.samples(me).count;
                incoming.resize(myLen, 0.0);

                MPI_Scatterv(nullptr,
                             nullptr,
                             nullptr,
                             MPI_DOUBLE,
                             incoming.data(),
                             static_cast<int>(myLen),
                             MPI_DOUBLE,
                             MPIContext::root,
                             MPI_COMM_WORLD);

                // Slice-relative indexing: computeFrame's offset (frameIdx *
                // hop) is already measured from the start of the slice, so
                // frame 0 of the slice is exactly global frame myStart.  A
                // rank with no frames holds an empty slice, and analyzeRange
                // returns immediately without reading it.
                localStart = 0;
            }
        }

        phases_.distributeMs = (MPI_Wtime() - tDistribute0) * 1000.0;

        // Root always works on the full signal it owns, addressed globally; the
        // other ranks work on whatever they received.  Under Broadcast that is
        // also the full signal, so localStart is global there too; under Scatter
        // it is the rank's own slice and localStart is 0.
        const std::vector<double>& localSignal =
            (me == MPIContext::root) ? signalOnRoot : incoming;

        // ── Step 4: local STFT (OpenMP-parallel within each rank) ─────────
        const double tCompute0 = MPI_Wtime();
        SpectrogramData local = engine_.analyzeRange(localSignal, localStart, myCount);
        phases_.computeMs = (MPI_Wtime() - tCompute0) * 1000.0;

        // ── Step 5: gather on root via MPI_Gatherv ────────────────────────
        // recvcounts and displs are in units of magnitudes (numBins per frame),
        // which since SpectrogramData went to float are half the bytes they
        // were — this is the transfer that bounds the whole algorithm's
        // scaling, so halving it is the cheapest win available on this path.
        const double tGather0 = MPI_Wtime();

        std::vector<int> recvcounts(P, 0);
        std::vector<int> displs(P, 0);

        if (me == MPIContext::root) {
            for (int r = 0; r < P; ++r) {
                recvcounts[r] = static_cast<int>(lay.frames(r) * numBins);
                displs[r]     = static_cast<int>(lay.start(r)  * numBins);
            }
        }

        SpectrogramData result;
        Magnitude* recvbuf = nullptr;

        if (me == MPIContext::root) {
            result.numFrames  = totalFrames;
            result.numBins    = numBins;
            result.frameSize  = frameSize;
            result.hopSize    = engine_.hopSize();
            result.sampleRate = static_cast<std::uint32_t>(sampleRate);
            // Not zero-filled: the MPI_Gatherv below covers this buffer
            // exactly — the per-rank counts sum to totalFrames and their
            // displacements are contiguous — so a fill here would write
            // 53 MiB of zeros for the collective to overwrite byte for byte.
            // At this geometry that measured 11 ms against 5 ms for the
            // transfer itself.  See MagnitudeBuffer in SpectrogramData.hpp.
            result.magnitudes.resize(totalFrames * numBins);
            recvbuf = result.magnitudes.data();
        }

        MPI_Gatherv(local.magnitudes.data(),
                    static_cast<int>(myCount * numBins),
                    kMagnitudeType,
                    recvbuf,
                    recvcounts.data(),
                    displs.data(),
                    kMagnitudeType,
                    MPIContext::root,
                    MPI_COMM_WORLD);

        phases_.gatherMs = (MPI_Wtime() - tGather0) * 1000.0;

        return result;  // empty on non-root ranks
    }

private:
    /**
     * @brief Block distribution of the frames over the ranks, and of the
     *        samples those frames read.
     *
     * Kept as one small value type because analyze() and localSampleCount()
     * must agree exactly: the scatter would silently corrupt the spectrogram
     * if the two ever disagreed on a block boundary.
     */
    struct Layout {
        /// One rank's contiguous block of a signal or a frame range.
        struct Slice {
            std::size_t offset = 0;   ///< first sample of the block
            std::size_t count  = 0;   ///< samples in the block (0 if idle)
        };

        std::size_t signalLen   = 0;
        std::size_t totalFrames = 0;
        std::size_t frameSize   = 0;
        std::size_t hopSize     = 0;
        std::size_t base        = 0;  ///< frames per rank, rounded down
        std::size_t rem         = 0;  ///< first `rem` ranks take one extra

        [[nodiscard]] std::size_t frames(int r) const noexcept {
            return base + (static_cast<std::size_t>(r) < rem ? 1 : 0);
        }

        [[nodiscard]] std::size_t start(int r) const noexcept {
            const std::size_t rr = static_cast<std::size_t>(r);
            return rr * base + std::min(rr, rem);
        }

        /**
         * @brief The samples read by rank r's frames:
         *        offset = start(r) * hop, count = (frames(r) - 1) * hop + frameSize.
         *
         * An idle rank gets {0, 0} rather than {start(r) * hop, 0}: when base
         * is 0 an idle rank's start is totalFrames, whose sample offset can
         * land past the end of the signal, and a displacement outside the send
         * buffer is not worth relying on even with a count of 0.
         *
         * The clamp to the signal end is defensive.  numFrames() counts only
         * frames that fit whole — (totalFrames - 1) * hop + frameSize <= N — so
         * at the current frame count it never binds; if it ever did,
         * computeFrame()'s bounds check zero-pads the tail as usual.
         */
        [[nodiscard]] Slice samples(int r) const noexcept {
            const std::size_t n = frames(r);
            if (n == 0) return {};
            const std::size_t offset = start(r) * hopSize;
            const std::size_t avail  = signalLen - std::min(offset, signalLen);
            return { offset, std::min((n - 1) * hopSize + frameSize, avail) };
        }
    };

    [[nodiscard]] Layout layout(std::size_t signalLen) const noexcept {
        Layout lay;
        lay.signalLen   = signalLen;
        lay.frameSize   = engine_.frameSize();
        lay.hopSize     = engine_.hopSize();
        lay.totalFrames = STFTAnalyzer<FFT,Window>::numFrames(
                              signalLen, lay.frameSize, lay.hopSize);

        const std::size_t P = static_cast<std::size_t>(ctx_.size());
        lay.base = lay.totalFrames / P;
        lay.rem  = lay.totalFrames % P;
        return lay;
    }

    /// Written by analyze(), which is const: the split is observed state about
    /// the last call, not part of the analyzer's logical value.
    mutable PhaseTimes         phases_{};

    const MPIContext&          ctx_;
    STFTAnalyzer<FFT, Window>  engine_;
    Distribution               dist_;
};

} // namespace stft
