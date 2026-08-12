/**
 * @file test_MPI_STFTAnalyzer.cpp
 * @brief Consistency test: MPI distributed result == serial STFTAnalyzer result.
 *
 * Run with at least 2 ranks:
 *   mpirun -np 2 ./test_MPI_STFTAnalyzer
 *   mpirun -np 4 ./test_MPI_STFTAnalyzer
 *
 * Each test verifies that MPI_STFTAnalyzer::analyze produces elementwise
 * identical magnitudes to STFTAnalyzer::analyze, regardless of the number
 * of ranks.
 *
 * Every case is run under both distribution strategies (Broadcast and
 * Scatter): they differ only in how the samples reach the ranks, so they are
 * held to the same reference and the same tolerance.  CMake registers the
 * suite at 1, 2, 3 and 4 ranks — 3 is the interesting one, since it leaves an
 * uneven remainder for every signal length used here.
 */

#include <gtest/gtest.h>
#include "mpi/MPIContext.hpp"
#include "stft/MPI_STFTAnalyzer.hpp"
#include "stft/STFTAnalyzer.hpp"
#include "stft/SpectrogramData.hpp"
#include "fft/IterativeFFT.hpp"
#include "window/HannWindow.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace stft;

// Global MPIContext — initialised once before any test runs.
static MPIContext* g_ctx = nullptr;

// Custom main to initialise MPI before GTest.
int main(int argc, char** argv) {
    MPIContext ctx(argc, argv);
    g_ctx = &ctx;
    ::testing::InitGoogleTest(&argc, argv);
    // Only root prints test output; all ranks run the tests.
    if (!ctx.isRoot())
        ::testing::UnitTest::GetInstance()->listeners().Release(
            ::testing::UnitTest::GetInstance()->listeners().default_result_printer());
    int result = RUN_ALL_TESTS();
    return result;
}

// ══════════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════════

namespace {

std::vector<double> sineSignal(std::size_t N, double freq, double sampleRate) {
    std::vector<double> sig(N);
    for (std::size_t n = 0; n < N; ++n)
        sig[n] = std::sin(2.0 * std::numbers::pi * freq * n / sampleRate);
    return sig;
}

/// The two strategies every case is run under, and a label for the messages.
constexpr Distribution kStrategies[] = { Distribution::Broadcast,
                                         Distribution::Scatter };

const char* label(Distribution d) {
    return d == Distribution::Broadcast ? "Broadcast" : "Scatter";
}

/**
 * @brief Analyse `signal` serially and distributedly, then compare.
 *
 * All ranks must reach this: analyze() is collective.  Root checks the
 * magnitudes against the serial reference; the other ranks check the other
 * half of the contract, namely that they come back empty.
 */
void expectMatchesSerial(const std::vector<double>& signal,
                         std::size_t   frame,
                         std::size_t   hop,
                         std::uint32_t sr,
                         Distribution  dist)
{
    STFTAnalyzer<IterativeFFT, HannWindow> serial(frame, hop, sr);
    const SpectrogramData serialSpec = serial.analyze(signal);

    MPI_STFTAnalyzer<IterativeFFT, HannWindow> mpiAnalyzer(*g_ctx, frame, hop,
                                                           sr, dist);
    const SpectrogramData mpiSpec = mpiAnalyzer.analyze(signal);

    if (!g_ctx->isRoot()) {
        EXPECT_EQ(mpiSpec.numFrames, std::size_t{0}) << label(dist);
        EXPECT_TRUE(mpiSpec.empty())                 << label(dist);
        return;
    }

    ASSERT_EQ(mpiSpec.numFrames, serialSpec.numFrames) << label(dist);
    ASSERT_EQ(mpiSpec.numBins,   serialSpec.numBins)   << label(dist);
    EXPECT_EQ(mpiSpec.frameSize,  frame)               << label(dist);
    EXPECT_EQ(mpiSpec.hopSize,    hop)                 << label(dist);
    EXPECT_EQ(mpiSpec.sampleRate, sr)                  << label(dist);

    for (std::size_t f = 0; f < serialSpec.numFrames; ++f)
        for (std::size_t b = 0; b < serialSpec.numBins; ++b)
            EXPECT_NEAR(mpiSpec.at(f, b), serialSpec.at(f, b), 1e-12)
                << "mismatch at frame " << f << " bin " << b
                << " (" << label(dist) << ", " << g_ctx->size() << " ranks)";
}

} // namespace

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

TEST(MPISTFTConsistency, DCSignalMatchesSerial) {
    constexpr std::size_t   FRAME = 512;
    constexpr std::size_t   HOP   = 256;
    constexpr std::uint32_t SR    = 44100;

    // 4 seconds of DC
    const std::size_t N = 4 * SR;
    const std::vector<double> signal(N, 1.0);

    for (const Distribution dist : kStrategies)
        expectMatchesSerial(signal, FRAME, HOP, SR, dist);
}

TEST(MPISTFTConsistency, SinusoidMatchesSerial) {
    constexpr std::size_t   FRAME = 1024;
    constexpr std::size_t   HOP   = 512;
    constexpr std::uint32_t SR    = 44100;

    // 440 Hz sine, 1 second
    const std::vector<double> signal = sineSignal(SR, 440.0, SR);

    for (const Distribution dist : kStrategies)
        expectMatchesSerial(signal, FRAME, HOP, SR, dist);
}

TEST(MPISTFTConsistency, MetadataMatchesSerial) {
    constexpr std::size_t   FRAME = 512;
    constexpr std::size_t   HOP   = 128;
    constexpr std::uint32_t SR    = 22050;

    // 1 second, constant 0.5.  The small hop makes the blocks overlap heavily,
    // so the scattered slices carry a large halo relative to their length.
    const std::vector<double> signal(SR, 0.5);

    for (const Distribution dist : kStrategies)
        expectMatchesSerial(signal, FRAME, HOP, SR, dist);
}

// ── Degenerate frame counts ──────────────────────────────────────────────────
// The interesting part of the scatter is what it does when there is little or
// nothing to scatter: the collective still has to complete on every rank.

TEST(MPISTFTConsistency, SignalShorterThanOneFrame) {
    constexpr std::size_t   FRAME = 1024;
    constexpr std::size_t   HOP   = 512;
    constexpr std::uint32_t SR    = 44100;

    // Half a frame: numFrames() is 0, so every rank is idle and every send
    // count is 0.  The result must still be a well-formed empty spectrogram.
    const std::vector<double> signal(FRAME / 2, 0.25);

    for (const Distribution dist : kStrategies) {
        expectMatchesSerial(signal, FRAME, HOP, SR, dist);

        MPI_STFTAnalyzer<IterativeFFT, HannWindow> mpiAnalyzer(*g_ctx, FRAME, HOP,
                                                               SR, dist);
        const SpectrogramData spec = mpiAnalyzer.analyze(signal);

        if (g_ctx->isRoot()) {
            EXPECT_EQ(spec.numFrames, std::size_t{0})         << label(dist);
            EXPECT_TRUE(spec.magnitudes.empty())              << label(dist);
            EXPECT_EQ(spec.numBins, FRAME / 2 + 1)            << label(dist);
        }
    }
}

TEST(MPISTFTConsistency, FewerFramesThanRanks) {
    constexpr std::size_t   FRAME = 256;
    constexpr std::size_t   HOP   = 128;
    constexpr std::uint32_t SR    = 44100;

    // A signal of FRAME + (K-1)*HOP samples yields exactly K frames.  With
    // K = 1 and K = 3 every rank beyond the first (or third) is idle at the
    // rank counts the suite is registered for.
    for (const std::size_t frames : {std::size_t{1}, std::size_t{3}}) {
        const std::vector<double> signal =
            sineSignal(FRAME + (frames - 1) * HOP, 1000.0, SR);

        for (const Distribution dist : kStrategies)
            expectMatchesSerial(signal, FRAME, HOP, SR, dist);
    }
}
