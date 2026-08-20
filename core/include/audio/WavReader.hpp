#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace stft {

/**
 * @brief Result of loading an audio file: its first channel and sample rate.
 */
struct WavData {
    std::vector<double> samples;    ///< Channel-0 samples, normalised to [-1.0, 1.0].
    std::uint32_t sampleRate = 0;   ///< Sample rate in Hz.
};

/**
 * @brief Loads mono audio samples from a WAV file on disk.
 *
 * A thin wrapper around the vendored AudioFile library: it keeps that
 * dependency out of the rest of the project (only WavReader.cpp includes
 * AudioFile.h) and narrows its general-purpose, multi-channel interface down
 * to the single signal STFTAnalyzer needs.
 */
class WavReader {
public:
    WavReader() = default;

    /**
     * @brief Loads an audio file and extracts its first channel.
     *
     * Multi-channel files are not downmixed: only channel 0 is returned,
     * every other channel is discarded.
     *
     * @param filename  Path to the audio file.
     * @return The channel-0 samples and the file's sample rate. Destructure
     *         with a structured binding: `auto [samples, sampleRate] = ...`.
     *
     * @throws std::runtime_error if the file cannot be loaded, or if it has
     *         no audio channels at all.
     */
    static WavData load(const std::string& filename);
};

} // namespace stft
