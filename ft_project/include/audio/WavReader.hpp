#ifndef WAV_READER_HPP
#define WAV_READER_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace amsc_stft {

class WavReader {
public:
    WavReader() = default;

    static std::vector<double> load(const std::string& filename,
                                    uint32_t& sampleRate);
};

} // namespace amsc_stft

#endif