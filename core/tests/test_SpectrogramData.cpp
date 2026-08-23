/**
 * @file test_SpectrogramData.cpp
 * @brief Unit tests for SpectrogramData: the flat, row-major buffer that
 *        both STFTAnalyzer and MPI_STFTAnalyzer return.
 *
 * Tests cover:
 *  - at() bounds checking
 *  - The row-major (frame, then bin) layout the file's own docs promise
 *    MPI_Gatherv can rely on
 *  - binFrequency() / frameTime() conversion formulas
 *  - binFrequency() / frameTime() before the geometry fields are set
 */

#include <gtest/gtest.h>
#include "stft/SpectrogramData.hpp"

#include <cstddef>
#include <stdexcept>

using namespace stft;

TEST(SpectrogramDataTest, AtThrowsOutOfRange) {
    SpectrogramData s;
    s.numFrames = 2; s.numBins = 4;
    s.magnitudes.resize(8, Magnitude{1});
    // at() is [[nodiscard]]; EXPECT_THROW only cares whether the call throws
    // and intentionally discards the (never-reached) return value.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
    EXPECT_THROW(s.at(2, 0), std::out_of_range);
    EXPECT_THROW(s.at(0, 4), std::out_of_range);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

// Writes through the documented layout formula directly, then reads back
// through at(): if at() ever used a different formula (e.g. a transposed
// bin * numFrames + frame), this would catch it. MPI_STFTAnalyzer's
// MPI_Gatherv relies on the documented layout holding exactly.
TEST(SpectrogramDataTest, AtIndexesRowMajorByFrameThenBin) {
    SpectrogramData s;
    s.numFrames = 3;
    s.numBins   = 4;
    s.magnitudes.resize(s.numFrames * s.numBins);

    for (std::size_t f = 0; f < s.numFrames; ++f)
        for (std::size_t b = 0; b < s.numBins; ++b)
            s.magnitudes[f * s.numBins + b] = static_cast<Magnitude>(f * 10 + b);

    for (std::size_t f = 0; f < s.numFrames; ++f)
        for (std::size_t b = 0; b < s.numBins; ++b)
            EXPECT_FLOAT_EQ(s.at(f, b), static_cast<Magnitude>(f * 10 + b))
                << "frame " << f << ", bin " << b;
}

TEST(SpectrogramDataTest, BinFrequencyAndFrameTime) {
    SpectrogramData s;
    s.frameSize = 1024; s.sampleRate = 44100; s.hopSize = 512;
    // bin 1 frequency = 1 * 44100 / 1024 ≈ 43.07 Hz
    EXPECT_NEAR(s.binFrequency(1), 43.07, 0.1);
    // frame 2 time = 2 * 512 / 44100 ≈ 0.02322 s
    EXPECT_NEAR(s.frameTime(2), 2.0 * 512.0 / 44100.0, 1e-6);
}

// A default-constructed SpectrogramData has frameSize = hopSize = sampleRate
// = 0; binFrequency()/frameTime() must return 0.0 rather than divide by zero.
TEST(SpectrogramDataTest, FrequencyAndTimeAreZeroBeforeGeometryIsSet) {
    const SpectrogramData s;
    EXPECT_EQ(s.binFrequency(5), 0.0);
    EXPECT_EQ(s.frameTime(3), 0.0);
}
