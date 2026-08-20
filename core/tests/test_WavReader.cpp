#include <gtest/gtest.h>
#include "audio/WavReader.hpp"
#include <filesystem>
#include <string>

using namespace stft;

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must be supplied by CMake (see tests/CMakeLists.txt)"
#endif

TEST(WavReaderTest, LoadValidFile) {
    const std::string testPath = std::string(TEST_DATA_DIR) + "/test_audio.wav";

    if (!std::filesystem::exists(testPath)) {
        GTEST_SKIP() << "test_audio.wav not found in " << TEST_DATA_DIR;
    }

    auto [samples, sampleRate] = WavReader::load(testPath);

    EXPECT_GT(sampleRate, 0u);
    EXPECT_GT(samples.size(), 0u);
}

TEST(WavReaderTest, ThrowsOnMissingFile) {
    EXPECT_THROW(WavReader::load("non_existent.wav"), std::runtime_error);
}
