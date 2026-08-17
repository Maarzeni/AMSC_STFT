#include "audio/WavReader.hpp"

// AudioFile.h is a vendored third-party header (not our code to edit); its
// AIFF decoder reinterpret_casts an int32_t as a float, which trips
// -Wstrict-aliasing under -O3. Silenced locally, around this one include site,
// rather than for the whole project.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
#include "audio/AudioFile.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <stdexcept>

namespace stft {

/**
 * @brief Loads a WAV file and extracts mono audio data.
 * @param filename Path to input WAV file.
 * @param sampleRate Output sample rate.
 * @return Mono audio signal (first channel).
 */
std::vector<double> WavReader::load(const std::string& filename,
                                    uint32_t& sampleRate) {

    AudioFile<double> audioFile;

    if (!audioFile.load(filename)) {
        throw std::runtime_error("Failed to load WAV file: " + filename);
    }

    sampleRate = audioFile.getSampleRate();

    if (audioFile.getNumChannels() == 0) {
        throw std::runtime_error("Invalid audio file: no channels found.");
    }

    // Mono signal extraction (first channel)
    return audioFile.samples[0];
}

} // namespace stft