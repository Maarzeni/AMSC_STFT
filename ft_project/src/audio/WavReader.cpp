#include "audio/WavReader.hpp"
#include "audio/AudioFile.h" // The external dependency
#include <iostream>
#include <stdexcept>

namespace amsc_stft {

std::vector<double> WavReader::load(const std::string& filename, uint32_t& sampleRate) {
    AudioFile<double> audioFile; // class from the AudioFile.h library!

    if (!audioFile.load(filename)) {
        throw std::runtime_error("Failed to load WAV file: " + filename);
    }

    sampleRate = audioFile.getSampleRate();
    
    // For spectral analysis, we usually work with mono signals.
    // If the file is stereo, we'll just take the first channel (index 0).
    if (audioFile.getNumChannels() == 0) {
        throw std::runtime_error("Audio file has no channels.");
    }

    return audioFile.samples[0]; 
}

} // namespace amsc_stft