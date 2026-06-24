/**
 * @file test_STFTAnalyzer.cpp
 * @brief Unit tests for STFTAnalyzer (serial, no MPI).
 *
 * Tests cover:
 *  - numFrames() frame-count arithmetic
 *  - DC signal energy concentrated at bin 0
 *  - Sinusoid at a bin-aligned frequency peaks at the expected bin
 *  - Out-of-range parameters (bad frameSize, bad hopSize) throw
 *  - Empty and short signals
 *  - SpectrogramData accessors and metadata
 */

#include <gtest/gtest.h>
#include "stft/STFTAnalyzer.hpp"
#include "stft/SpectrogramData.hpp"
#include "fft/IterativeFFT.hpp"
#include "window/HannWindow.hpp"
#include "window/HammingWindow.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace stft;
using SerialSTFT = STFTAnalyzer<IterativeFFT, HannWindow>;

// ══════════════════════════════════════════════════════════════════════════════
// TEST FIXTURE
// ══════════════════════════════════════════════════════════════════════════════

class STFTAnalyzerTest : public ::testing::Test {
protected:
    static constexpr std::size_t   FRAME = 1024;
    static constexpr std::size_t   HOP   = 512;
    static constexpr std::uint32_t SR    = 44100;

    SerialSTFT analyzer{FRAME, HOP, SR};
};

// ══════════════════════════════════════════════════════════════════════════════
// 1. Frame-count arithmetic
// ══════════════════════════════════════════════════════════════════════════════

TEST(NumFramesTest, ShortSignalGivesZero) {
    EXPECT_EQ(SerialSTFT::numFrames(0,    1024, 512), 0u);
    EXPECT_EQ(SerialSTFT::numFrames(512,  1024, 512), 0u);
    EXPECT_EQ(SerialSTFT::numFrames(1023, 1024, 512), 0u);
}

TEST(NumFramesTest, ExactFrameGivesOne) {
    EXPECT_EQ(SerialSTFT::numFrames(1024, 1024, 512), 1u);
}

TEST(NumFramesTest, StandardCases) {
    // signal = 1024 + k*512 gives 1 + k frames
    EXPECT_EQ(SerialSTFT::numFrames(1024 + 1 * 512, 1024, 512), 2u);
    EXPECT_EQ(SerialSTFT::numFrames(1024 + 2 * 512, 1024, 512), 3u);
    EXPECT_EQ(SerialSTFT::numFrames(1024 + 9 * 512, 1024, 512), 10u);
}

TEST(NumFramesTest, HopEqualsFrame) {
    EXPECT_EQ(SerialSTFT::numFrames(4096, 1024, 1024), 4u);
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. Construction validation
// ══════════════════════════════════════════════════════════════════════════════

TEST(STFTAnalyzerConstructionTest, ThrowsOnNonPowerOfTwoFrame) {
    EXPECT_THROW(SerialSTFT(1000, 512, 44100), std::invalid_argument);
    EXPECT_THROW(SerialSTFT(3,    1,   44100), std::invalid_argument);
}

TEST(STFTAnalyzerConstructionTest, ThrowsOnFrameSizeLessThanTwo) {
    EXPECT_THROW(SerialSTFT(0, 1, 44100), std::invalid_argument);
    EXPECT_THROW(SerialSTFT(1, 1, 44100), std::invalid_argument);
}

TEST(STFTAnalyzerConstructionTest, ThrowsOnZeroHop) {
    EXPECT_THROW(SerialSTFT(1024, 0, 44100), std::invalid_argument);
}

TEST(STFTAnalyzerConstructionTest, ValidConstruction) {
    EXPECT_NO_THROW(SerialSTFT(512,  256, 44100));
    EXPECT_NO_THROW(SerialSTFT(2048, 1,   48000));
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. DC signal → energy at bin 0
// ══════════════════════════════════════════════════════════════════════════════

TEST_F(STFTAnalyzerTest, DCSignalPeaksAtBinZero) {
    // Constant signal: all samples = 1.0
    std::vector<double> signal(FRAME * 4, 1.0);
    const auto spec = analyzer.analyze(signal);

    ASSERT_FALSE(spec.empty());
    ASSERT_GE(spec.numFrames, 1u);

    // For any frame: bin 0 should hold the largest magnitude.
    for (std::size_t f = 0; f < spec.numFrames; ++f) {
        double bin0 = spec.at(f, 0);
        for (std::size_t b = 1; b < spec.numBins; ++b)
            EXPECT_GE(bin0, spec.at(f, b))
                << "bin 0 should dominate for DC signal (frame " << f << ")";
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. Sinusoid at a bin-aligned frequency peaks at the expected bin
// ══════════════════════════════════════════════════════════════════════════════

TEST_F(STFTAnalyzerTest, BinAlignedSinusoidPeaksAtCorrectBin) {
    // k = 5 → frequency = k * SR / frameSize = 5 * 44100 / 1024 ≈ 215 Hz
    constexpr std::size_t k = 5;

    // 6 complete frames of the sinusoid
    const std::size_t signalLen = FRAME + 5 * HOP;
    std::vector<double> signal(signalLen);
    for (std::size_t n = 0; n < signalLen; ++n)
        signal[n] = std::sin(2.0 * std::numbers::pi * k * n / static_cast<double>(FRAME));

    const auto spec = analyzer.analyze(signal);
    ASSERT_FALSE(spec.empty());

    // Find the peak bin in the middle frame.
    const std::size_t mid = spec.numFrames / 2;
    std::size_t peakBin = 0;
    double      peakMag = 0.0;
    for (std::size_t b = 0; b < spec.numBins; ++b) {
        if (spec.at(mid, b) > peakMag) {
            peakMag = spec.at(mid, b);
            peakBin = b;
        }
    }

    // Allow ±1 bin tolerance for leakage.
    EXPECT_NEAR(static_cast<int>(peakBin), static_cast<int>(k), 1)
        << "Sinusoid at bin " << k << " should peak near that bin";
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. Empty and short signals
// ══════════════════════════════════════════════════════════════════════════════

TEST_F(STFTAnalyzerTest, EmptySignalGivesEmptySpectrogram) {
    std::vector<double> empty;
    const auto spec = analyzer.analyze(empty);
    EXPECT_TRUE(spec.empty());
    EXPECT_EQ(spec.numFrames, 0u);
}

TEST_F(STFTAnalyzerTest, SignalShorterThanFrameGivesEmptySpectrogram) {
    std::vector<double> tooShort(FRAME - 1, 0.5);
    const auto spec = analyzer.analyze(tooShort);
    EXPECT_TRUE(spec.empty());
    EXPECT_EQ(spec.numFrames, 0u);
}

// ══════════════════════════════════════════════════════════════════════════════
// 6. Spectrogram metadata
// ══════════════════════════════════════════════════════════════════════════════

TEST_F(STFTAnalyzerTest, SpectrogramMetadataIsCorrect) {
    std::vector<double> signal(FRAME + 2 * HOP, 0.0);
    const auto spec = analyzer.analyze(signal);

    EXPECT_EQ(spec.frameSize,  FRAME);
    EXPECT_EQ(spec.hopSize,    HOP);
    EXPECT_EQ(spec.sampleRate, SR);
    EXPECT_EQ(spec.numBins,    FRAME / 2 + 1);
    EXPECT_EQ(spec.numFrames,  3u);            // 1 + (FRAME+2*HOP - FRAME)/HOP = 3
    EXPECT_EQ(spec.magnitudes.size(), 3 * (FRAME / 2 + 1));
}

// ══════════════════════════════════════════════════════════════════════════════
// 7. SpectrogramData helpers
// ══════════════════════════════════════════════════════════════════════════════

TEST(SpectrogramDataTest, AtThrowsOutOfRange) {
    SpectrogramData s;
    s.numFrames = 2; s.numBins = 4;
    s.magnitudes.resize(8, 1.0);
    EXPECT_THROW(s.at(2, 0), std::out_of_range);
    EXPECT_THROW(s.at(0, 4), std::out_of_range);
}

TEST(SpectrogramDataTest, BinFrequencyAndFrameTime) {
    SpectrogramData s;
    s.frameSize = 1024; s.sampleRate = 44100; s.hopSize = 512;
    // bin 1 frequency = 1 * 44100 / 1024 ≈ 43.07 Hz
    EXPECT_NEAR(s.binFrequency(1), 43.07, 0.1);
    // frame 2 time = 2 * 512 / 44100 ≈ 0.02322 s
    EXPECT_NEAR(s.frameTime(2), 2.0 * 512.0 / 44100.0, 1e-6);
}

// ══════════════════════════════════════════════════════════════════════════════
// 8. Different window types (template parameter)
// ══════════════════════════════════════════════════════════════════════════════

TEST(STFTAnalyzerTemplateTest, WorksWithHammingWindow) {
    STFTAnalyzer<IterativeFFT, HammingWindow> analyzer(512, 256, 22050);
    std::vector<double> signal(512 + 256, 0.5);
    const auto spec = analyzer.analyze(signal);
    EXPECT_EQ(spec.numFrames, 2u);
    EXPECT_EQ(spec.numBins,   257u);
}
