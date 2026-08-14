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
// 4b. Multi-tone signal with known frequencies
//
// A sum of three bin-aligned sinusoids with distinct amplitudes. Because each
// component sits exactly on a bin, the STFT must recover all three peaks at the
// expected bins, with the correct relative amplitudes, in every frame.
//
// Normalization note: computeFrame divides by (frameSize * coherentGain), i.e.
// by sum(w). For a real sinusoid of amplitude A the one-sided magnitude at the
// peak bin is therefore A/2, and the two adjacent bins hold ~A/4 (Hann main
// lobe). Amplitudes are chosen so that the side lobe of one component stays
// below the peak of the next (0.25 < 0.30), keeping the peaks unambiguous.
// ══════════════════════════════════════════════════════════════════════════════

namespace {

// Bin index and amplitude of each synthetic component.
constexpr std::size_t kToneBins[3] = {8, 32, 96};
constexpr double      kToneAmps[3] = {1.0, 0.6, 0.3};

// Tolerances calibrated on the measured output (peak error < 1e-5,
// noise floor far from the components < 4e-5).
constexpr double kPeakTol  = 1e-3;
constexpr double kFloorTol = 1e-3;

// x[n] = sum_i A_i * sin(2*pi*k_i*n / frameSize)
std::vector<double> makeMultiTone(std::size_t length, std::size_t frameSize) {
    std::vector<double> signal(length, 0.0);
    for (std::size_t n = 0; n < length; ++n)
        for (int c = 0; c < 3; ++c)
            signal[n] += kToneAmps[c] *
                std::sin(2.0 * std::numbers::pi * kToneBins[c] * n /
                         static_cast<double>(frameSize));
    return signal;
}

} // namespace

TEST_F(STFTAnalyzerTest, MultiToneSignalPeaksAtAllExpectedBins) {
    const auto signal = makeMultiTone(FRAME + 5 * HOP, FRAME);
    const auto spec   = analyzer.analyze(signal);

    ASSERT_FALSE(spec.empty());
    const std::size_t mid = spec.numFrames / 2;

    for (int c = 0; c < 3; ++c) {
        const std::size_t k = kToneBins[c];

        // The expected bin must be a strict local maximum.
        EXPECT_GT(spec.at(mid, k), spec.at(mid, k - 1))
            << "bin " << k << " should exceed its lower neighbour";
        EXPECT_GT(spec.at(mid, k), spec.at(mid, k + 1))
            << "bin " << k << " should exceed its upper neighbour";

        // And carry the expected amplitude A/2.
        EXPECT_NEAR(spec.at(mid, k), kToneAmps[c] / 2.0, kPeakTol)
            << "amplitude mismatch at bin " << k;
    }

    // Bins far from every component must be essentially silent.
    for (std::size_t b = 0; b < spec.numBins; ++b) {
        bool nearTone = false;
        for (int c = 0; c < 3; ++c)
            if (b + 3 >= kToneBins[c] && b <= kToneBins[c] + 3) nearTone = true;

        if (!nearTone)
            EXPECT_LT(spec.at(mid, b), kFloorTol)
                << "unexpected energy at bin " << b;
    }
}

TEST_F(STFTAnalyzerTest, MultiTonePreservesAmplitudeOrdering) {
    const auto signal = makeMultiTone(FRAME + 5 * HOP, FRAME);
    const auto spec   = analyzer.analyze(signal);

    ASSERT_FALSE(spec.empty());
    const std::size_t mid = spec.numFrames / 2;

    // Amplitudes are 1.0 > 0.6 > 0.3, so the peaks must follow the same order:
    // windowing and normalization must not distort relative magnitudes.
    EXPECT_GT(spec.at(mid, kToneBins[0]), spec.at(mid, kToneBins[1]));
    EXPECT_GT(spec.at(mid, kToneBins[1]), spec.at(mid, kToneBins[2]));

    // Ratios between components are preserved as well.
    EXPECT_NEAR(spec.at(mid, kToneBins[1]) / spec.at(mid, kToneBins[0]),
                kToneAmps[1] / kToneAmps[0], 1e-3);
    EXPECT_NEAR(spec.at(mid, kToneBins[2]) / spec.at(mid, kToneBins[0]),
                kToneAmps[2] / kToneAmps[0], 1e-3);
}

TEST_F(STFTAnalyzerTest, MultiTonePeakFrequenciesMatchKnownValues) {
    const auto signal = makeMultiTone(FRAME + 5 * HOP, FRAME);
    const auto spec   = analyzer.analyze(signal);

    ASSERT_FALSE(spec.empty());

    // Tie the peak bins back to physical frequencies: k * SR / FRAME.
    // ≈ 344.5 Hz, 1378.1 Hz, 4134.4 Hz for k = 8, 32, 96 at 44.1 kHz.
    for (int c = 0; c < 3; ++c) {
        const double expectedHz =
            static_cast<double>(kToneBins[c]) * SR / static_cast<double>(FRAME);

        EXPECT_NEAR(spec.binFrequency(kToneBins[c]), expectedHz, 1e-6)
            << "frequency mismatch for component " << c;
    }
}

TEST_F(STFTAnalyzerTest, MultiToneIsStableAcrossFrames) {
    const auto signal = makeMultiTone(FRAME + 5 * HOP, FRAME);
    const auto spec   = analyzer.analyze(signal);

    ASSERT_FALSE(spec.empty());
    ASSERT_GT(spec.numFrames, 1u);

    // The signal is stationary and every frame starts at a multiple of HOP,
    // so each component is bin-aligned in every frame: the peaks must be
    // identical throughout, regardless of the per-frame phase offset.
    for (std::size_t f = 0; f < spec.numFrames; ++f)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(spec.at(f, kToneBins[c]), kToneAmps[c] / 2.0, kPeakTol)
                << "frame " << f << ", bin " << kToneBins[c];
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
