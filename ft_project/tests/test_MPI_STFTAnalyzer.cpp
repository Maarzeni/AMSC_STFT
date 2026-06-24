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
    std::vector<double> signal(N, 1.0);

    // Serial result (computed on all ranks identically — no MPI needed)
    STFTAnalyzer<IterativeFFT, HannWindow> serial(FRAME, HOP, SR);
    const SpectrogramData serialSpec = serial.analyze(signal);

    // Distributed result
    MPI_STFTAnalyzer<IterativeFFT, HannWindow> mpiAnalyzer(*g_ctx, FRAME, HOP, SR);
    const SpectrogramData mpiSpec = mpiAnalyzer.analyze(signal);

    if (g_ctx->isRoot()) {
        ASSERT_EQ(mpiSpec.numFrames, serialSpec.numFrames);
        ASSERT_EQ(mpiSpec.numBins,   serialSpec.numBins);

        for (std::size_t f = 0; f < serialSpec.numFrames; ++f)
            for (std::size_t b = 0; b < serialSpec.numBins; ++b)
                EXPECT_NEAR(mpiSpec.at(f, b), serialSpec.at(f, b), 1e-12)
                    << "mismatch at frame " << f << " bin " << b;
    }
}

TEST(MPISTFTConsistency, SinusoidMatchesSerial) {
    constexpr std::size_t   FRAME = 1024;
    constexpr std::size_t   HOP   = 512;
    constexpr std::uint32_t SR    = 44100;

    // 440 Hz sine, 1 second
    std::vector<double> signal = sineSignal(SR, 440.0, SR);

    STFTAnalyzer<IterativeFFT, HannWindow> serial(FRAME, HOP, SR);
    const SpectrogramData serialSpec = serial.analyze(signal);

    MPI_STFTAnalyzer<IterativeFFT, HannWindow> mpiAnalyzer(*g_ctx, FRAME, HOP, SR);
    const SpectrogramData mpiSpec = mpiAnalyzer.analyze(signal);

    if (g_ctx->isRoot()) {
        ASSERT_EQ(mpiSpec.numFrames, serialSpec.numFrames);
        ASSERT_EQ(mpiSpec.numBins,   serialSpec.numBins);

        for (std::size_t f = 0; f < serialSpec.numFrames; ++f)
            for (std::size_t b = 0; b < serialSpec.numBins; ++b)
                EXPECT_NEAR(mpiSpec.at(f, b), serialSpec.at(f, b), 1e-12)
                    << "mismatch at frame " << f << " bin " << b;
    }
}

TEST(MPISTFTConsistency, MetadataMatchesSerial) {
    constexpr std::size_t   FRAME = 512;
    constexpr std::size_t   HOP   = 128;
    constexpr std::uint32_t SR    = 22050;

    std::vector<double> signal(SR, 0.5);  // 1 second, constant 0.5

    STFTAnalyzer<IterativeFFT, HannWindow> serial(FRAME, HOP, SR);
    const SpectrogramData serialSpec = serial.analyze(signal);

    MPI_STFTAnalyzer<IterativeFFT, HannWindow> mpiAnalyzer(*g_ctx, FRAME, HOP, SR);
    const SpectrogramData mpiSpec = mpiAnalyzer.analyze(signal);

    if (g_ctx->isRoot()) {
        EXPECT_EQ(mpiSpec.numFrames,  serialSpec.numFrames);
        EXPECT_EQ(mpiSpec.numBins,    serialSpec.numBins);
        EXPECT_EQ(mpiSpec.frameSize,  FRAME);
        EXPECT_EQ(mpiSpec.hopSize,    HOP);
        EXPECT_EQ(mpiSpec.sampleRate, SR);
    }
}
