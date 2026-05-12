#ifndef WAV_READER_HPP
#define WAV_READER_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace stft {

/**
 * @class WavReader
 * @brief Utility class for loading WAV audio files.
 *
 * Provides a minimal interface for extracting mono audio data
 * and sample rate information from a WAV file.
 */
class WavReader {
public:
    WavReader() = default;

    /**
     * @brief Loads a WAV file from disk.
     * @param filename Path to the WAV file.
     * @param sampleRate Output sample rate of the audio file.
     * @return Vector containing mono audio samples.
     */
    static std::vector<double> load(const std::string& filename,
                                    uint32_t& sampleRate);
};

} // namespace stft

#endif