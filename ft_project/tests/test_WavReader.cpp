#include <gtest/gtest.h>
#include "audio/WavReader.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace stft;

// Helper to check if a file exists
bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

TEST(WavReaderTest, LoadValidFile) {

    std::string testPath = "../tests/data/test_audio.wav";

    if (!fileExists(testPath)) {
        GTEST_SKIP() << "test_audio.wav not found in tests/data/";
    }

    uint32_t sampleRate = 0;
    auto samples = WavReader::load(testPath, sampleRate);

    // Basic checks
    EXPECT_GT(sampleRate, 0);
    EXPECT_GT(samples.size(), 0);

    // DEBUG OUTPUT (useful for now)
    std::cout << "\n=== WAV DEBUG INFO ===" << std::endl;
    std::cout << "Sample rate: " << sampleRate << std::endl;
    std::cout << "Number of samples: " << samples.size() << std::endl;

    std::cout << "First 20 samples:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(20, samples.size()); i++) {
        std::cout << samples[i] << std::endl;
    }
    std::cout << "======================\n" << std::endl;
}

TEST(WavReaderTest, ThrowsOnMissingFile) {
    uint32_t sampleRate = 0;

    EXPECT_THROW(
        WavReader::load("non_existent.wav", sampleRate),
        std::runtime_error
    );
}