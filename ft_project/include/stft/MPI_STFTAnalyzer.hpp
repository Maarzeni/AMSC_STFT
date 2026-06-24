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
 * 2. MPI_Bcast sends the sample data from root to all non-root ranks.
 * 3. Frames are block-distributed: rank r processes frames [start_r, start_r +
 *    count_r).  The first `rem` ranks get one extra frame when totalFrames is
 *    not divisible by P.
 * 4. Each rank calls STFTAnalyzer::analyzeRange() on its block.  The per-rank
 *    magnitudes are a contiguous slice of the final spectrogram buffer.
 * 5. MPI_Gatherv assembles the full magnitude matrix on the root rank.
 *    Non-root ranks receive an empty SpectrogramData.
 *
 * ─── Hybrid MPI + OpenMP ────────────────────────────────────────────────────
 * STFTAnalyzer uses `#pragma omp parallel for` over its frame block, so each
 * MPI rank automatically uses all available cores.  Recommended usage:
 *   mpirun -np <nodes> --map-by node --bind-to none ./mpi_main
 *   OMP_NUM_THREADS=<cores_per_node> mpirun ...
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
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace stft {

template<typename FFT, typename Window>
    requires IsFFT<FFT> && WindowFunction<Window>
class MPI_STFTAnalyzer {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    /**
     * @param ctx        Live MPIContext (must remain alive for the duration).
     * @param frameSize  STFT frame size (power of two, >= 2).
     * @param hopSize    Hop between successive frames (>= 1).
     * @param sampleRate Audio sample rate in Hz (informational).
     */
    MPI_STFTAnalyzer(const MPIContext& ctx,
                     std::size_t frameSize,
                     std::size_t hopSize,
                     std::uint32_t sampleRate = 0)
        : ctx_(ctx),
          engine_(frameSize, hopSize, sampleRate)
    {}

    // ── Analysis ──────────────────────────────────────────────────────────────

    /**
     * @brief Distributed STFT.  All ranks must call this collectively.
     *
     * @param signalOnRoot  On rank 0: the full audio signal.
     *                      On other ranks: any value (contents are ignored and
     *                      overwritten by the broadcast).
     * @return On rank 0: a fully populated SpectrogramData.
     *         On other ranks: an empty SpectrogramData (numFrames == 0).
     */
    [[nodiscard]] SpectrogramData
    analyze(std::vector<double> signalOnRoot) const
    {
        const std::size_t frameSize = engine_.frameSize();
        const std::size_t hopSize   = engine_.hopSize();
        const std::size_t numBins   = frameSize / 2 + 1;
        const int P                 = ctx_.size();
        const int me                = ctx_.rank();

        // ── Step 1: broadcast signal length + sample rate ──────────────────
        unsigned long signalLen  = (me == MPIContext::root)
                                   ? static_cast<unsigned long>(signalOnRoot.size())
                                   : 0UL;
        unsigned int  sampleRate = (me == MPIContext::root)
                                   ? static_cast<unsigned int>(engine_.sampleRate())
                                   : 0U;

        MPI_Bcast(&signalLen,  1, MPI_UNSIGNED_LONG, MPIContext::root, MPI_COMM_WORLD);
        MPI_Bcast(&sampleRate, 1, MPI_UNSIGNED,      MPIContext::root, MPI_COMM_WORLD);

        // ── Step 2: broadcast signal data ─────────────────────────────────
        if (me != MPIContext::root)
            signalOnRoot.resize(signalLen, 0.0);

        MPI_Bcast(signalOnRoot.data(),
                  static_cast<int>(signalLen),
                  MPI_DOUBLE,
                  MPIContext::root,
                  MPI_COMM_WORLD);

        // ── Step 3: compute frame distribution ────────────────────────────
        const std::size_t totalFrames =
            STFTAnalyzer<FFT,Window>::numFrames(signalLen, frameSize, hopSize);

        const std::size_t base = totalFrames / static_cast<std::size_t>(P);
        const std::size_t rem  = totalFrames % static_cast<std::size_t>(P);

        auto framesForRank = [&](int r) -> std::size_t {
            return base + (static_cast<std::size_t>(r) < rem ? 1 : 0);
        };
        auto startForRank = [&](int r) -> std::size_t {
            const std::size_t rr = static_cast<std::size_t>(r);
            return rr * base + std::min(rr, rem);
        };

        const std::size_t myCount = framesForRank(me);
        const std::size_t myStart = startForRank(me);

        // ── Step 4: local STFT (OpenMP-parallel within each rank) ─────────
        SpectrogramData local = engine_.analyzeRange(signalOnRoot, myStart, myCount);

        // ── Step 5: gather on root via MPI_Gatherv ────────────────────────
        // recvcounts and displs are in units of doubles (numBins per frame).
        std::vector<int> recvcounts(P, 0);
        std::vector<int> displs(P, 0);

        if (me == MPIContext::root) {
            for (int r = 0; r < P; ++r) {
                recvcounts[r] = static_cast<int>(framesForRank(r) * numBins);
                displs[r]     = static_cast<int>(startForRank(r)  * numBins);
            }
        }

        SpectrogramData result;
        double* recvbuf = nullptr;

        if (me == MPIContext::root) {
            result.numFrames  = totalFrames;
            result.numBins    = numBins;
            result.frameSize  = frameSize;
            result.hopSize    = hopSize;
            result.sampleRate = static_cast<std::uint32_t>(sampleRate);
            result.magnitudes.resize(totalFrames * numBins, 0.0);
            recvbuf = result.magnitudes.data();
        }

        MPI_Gatherv(local.magnitudes.data(),
                    static_cast<int>(myCount * numBins),
                    MPI_DOUBLE,
                    recvbuf,
                    recvcounts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    MPIContext::root,
                    MPI_COMM_WORLD);

        return result;  // empty on non-root ranks
    }

private:
    const MPIContext&          ctx_;
    STFTAnalyzer<FFT, Window>  engine_;
};

} // namespace stft
