/**
 * @file main.cpp
 * @brief Serial STFT demonstration — WAV in, PNG spectrogram out.
 *
 * @details
 * Usage:
 *   ./main <audio.wav> [frameSize=1024] [hopSize=512] [window=hann|hamming|blackman]
 *
 * The WAV file must already exist.  The program exits with a clear error
 * message (non-zero exit code) if the file is missing or unreadable.
 *
 * The output PNG is written next to the input file:
 *   /path/to/audio.wav  →  /path/to/audio_spectrogram.png
 *
 * Window choices:
 *   hann      (default) — good all-round choice
 *   hamming             — sharper main lobe, non-zero endpoints
 *   blackman            — best sidelobe rejection, widest main lobe
 *
 * Examples:
 *   ./main recording.wav
 *   ./main recording.wav 2048 1024 blackman
 *   OMP_NUM_THREADS=8 ./main recording.wav 1024 256 hamming
 */

#include "stft/STFTAnalyzer.hpp"
#include "stft/SpectrogramData.hpp"
#include "output/ImageExporter.hpp"
#include "audio/WavReader.hpp"
#include "fft/IterativeFFT.hpp"
#include "window/HannWindow.hpp"
#include "window/HammingWindow.hpp"
#include "window/BlackmanWindow.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace stft;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void printUsage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " <audio.wav> [frameSize=1024] [hopSize=512]"
           " [window=hann|hamming|blackman]\n"
        << "\n"
        << "  audio.wav   existing WAV file (mono; first channel used if stereo)\n"
        << "  frameSize   STFT frame size, must be a power of two (default 1024)\n"
        << "  hopSize     hop between frames in samples (default 512)\n"
        << "  window      windowing function: hann (default), hamming, blackman\n"
        << "\n"
        << "Output PNG is written next to the input file:\n"
        << "  <stem>_spectrogram.png\n";
}

/// Derive the output PNG path from the input WAV path.
/// e.g. /path/to/audio.wav → /path/to/audio_spectrogram.png
static std::string outputPath(const std::string& wavPath) {
    const std::size_t dot = wavPath.rfind('.');
    const std::string stem =
        (dot == std::string::npos) ? wavPath : wavPath.substr(0, dot);
    return stem + "_spectrogram.png";
}

static bool isPowerOfTwo(std::size_t n) {
    return n >= 2 && (n & (n - 1)) == 0;
}

// ── Per-window run() instantiates the correct template parameter ──────────────

template<typename Window>
int run(const std::vector<double>& signal,
        std::uint32_t  sampleRate,
        std::size_t    frameSize,
        std::size_t    hopSize,
        const std::string& wavPath)
{
    const std::string outPng = outputPath(wavPath);

    std::cout << "  Frame size   : " << frameSize  << " samples\n"
              << "  Hop size     : " << hopSize    << " samples\n"
              << "  Sample rate  : " << sampleRate << " Hz\n"
              << "  Signal len   : " << signal.size() << " samples  ("
              << static_cast<double>(signal.size()) / sampleRate << " s)\n";

    STFTAnalyzer<IterativeFFT, Window> analyzer(frameSize, hopSize, sampleRate);

    const std::size_t totalFrames =
        STFTAnalyzer<IterativeFFT, Window>::numFrames(
            signal.size(), frameSize, hopSize);

    if (totalFrames == 0) {
        std::cerr << "Error: signal is shorter than one frame ("
                  << frameSize << " samples).\n";
        return 1;
    }

    std::cout << "  Frames       : " << totalFrames << "\n"
              << "  Freq bins    : " << (frameSize / 2 + 1) << "\n"
              << "\nComputing STFT...\n";

    const SpectrogramData spec = analyzer.analyze(signal);

    // Print dominant frequency for the middle frame.
    const std::size_t mid = spec.numFrames / 2;
    std::size_t peakBin = 0;
    double      peakMag = 0.0;
    for (std::size_t b = 0; b < spec.numBins; ++b) {
        if (spec.at(mid, b) > peakMag) {
            peakMag = spec.at(mid, b);
            peakBin = b;
        }
    }
    std::cout << "  Dominant freq (middle frame): "
              << spec.binFrequency(peakBin) << " Hz  (bin " << peakBin << ")\n"
              << "\nExporting spectrogram to: " << outPng << " ...\n";

    ImageExporter::exportPNG(spec, outPng);

    std::cout << "Done. Spectrogram saved to: " << outPng << "\n";
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string wavPath = argv[1];

    // Parse optional arguments.
    std::size_t   frameSize = 1024;
    std::size_t   hopSize   = 512;
    std::string   window    = "hann";

    if (argc >= 3) {
        try { frameSize = static_cast<std::size_t>(std::stoul(argv[2])); }
        catch (...) {
            std::cerr << "Error: invalid frameSize '" << argv[2] << "'.\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    if (argc >= 4) {
        try { hopSize = static_cast<std::size_t>(std::stoul(argv[3])); }
        catch (...) {
            std::cerr << "Error: invalid hopSize '" << argv[3] << "'.\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    if (argc >= 5) {
        window = argv[4];
        std::transform(window.begin(), window.end(), window.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    }

    // Validate.
    if (!isPowerOfTwo(frameSize)) {
        std::cerr << "Error: frameSize must be a power of two >= 2 (got "
                  << frameSize << ").\n";
        return 1;
    }
    if (hopSize < 1) {
        std::cerr << "Error: hopSize must be >= 1.\n";
        return 1;
    }
    if (window != "hann" && window != "hamming" && window != "blackman") {
        std::cerr << "Error: unknown window '" << window
                  << "'. Choose hann, hamming, or blackman.\n";
        return 1;
    }

    // Load WAV.
    std::cout << "Loading: " << wavPath << "\n";
    std::uint32_t sampleRate = 0;
    std::vector<double> signal;
    try {
        signal = WavReader::load(wavPath, sampleRate);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    if (signal.empty()) {
        std::cerr << "Error: WAV file contains no samples.\n";
        return 1;
    }

    std::cout << "Window       : " << window << "\n";

    // Dispatch to the templated run<Window>().
    if      (window == "hann")     return run<HannWindow>    (signal, sampleRate, frameSize, hopSize, wavPath);
    else if (window == "hamming")  return run<HammingWindow> (signal, sampleRate, frameSize, hopSize, wavPath);
    else                           return run<BlackmanWindow>(signal, sampleRate, frameSize, hopSize, wavPath);
}
