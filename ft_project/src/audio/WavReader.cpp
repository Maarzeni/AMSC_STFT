#include "audio/WavReader.hpp"
#include "audio/AudioFile.h"
#include <stdexcept>

namespace amsc_stft {

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

} // namespace amsc_stft