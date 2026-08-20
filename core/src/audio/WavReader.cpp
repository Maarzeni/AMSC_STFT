#include "audio/WavReader.hpp"

// AudioFile.h is a vendored third-party header (not our code to edit); its
// AIFF decoder reinterpret_casts an int32_t as a float, which trips
// -Wstrict-aliasing under -O3. Silenced locally, around this one include
// site, rather than for the whole project.
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

WavData WavReader::load(const std::string& filename) {
    AudioFile<double> audioFile;

    // AudioFile prints its own message to stdout on failure, which would
    // appear before — and alongside — the exception below. One diagnostic
    // per failure: ours, which names the file and travels with the error.
    audioFile.shouldLogErrorsToConsole(false);

    if (!audioFile.load(filename)) {
        throw std::runtime_error("Failed to load WAV file: " + filename);
    }

    if (audioFile.getNumChannels() == 0) {
        throw std::runtime_error("Invalid audio file: no channels found.");
    }

    return WavData{std::move(audioFile.samples.front()), audioFile.getSampleRate()};
}

} // namespace stft
