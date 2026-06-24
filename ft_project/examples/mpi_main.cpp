/**
 * @file mpi_main.cpp
 * @brief End-to-end MPI distributed STFT demonstration.
 *
 * Runs the distributed STFT on a synthetic composite sine signal (440 + 880 +
 * 1760 Hz) or on a WAV file if a path is given as argv[1].
 *
 * Usage:
 *   mpirun -np 4 ./mpi_main [path/to/audio.wav]
 *   OMP_NUM_THREADS=4 mpirun -np 2 ./mpi_main
 */

#include "mpi/MPIContext.hpp"
#include "stft/MPI_STFTAnalyzer.hpp"
#include "stft/SpectrogramData.hpp"
#include "fft/IterativeFFT.hpp"
#include "window/HannWindow.hpp"
#include "audio/WavReader.hpp"

#include <cmath>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

namespace {

std::vector<double> syntheticSignal(std::size_t numSamples, std::uint32_t sampleRate) {
    std::vector<double> sig(numSamples);
    const double fs = static_cast<double>(sampleRate);
    for (std::size_t n = 0; n < numSamples; ++n) {
        const double t = static_cast<double>(n) / fs;
        sig[n] = 0.5 * std::sin(2.0 * std::numbers::pi * 440.0  * t)
               + 0.3 * std::sin(2.0 * std::numbers::pi * 880.0  * t)
               + 0.2 * std::sin(2.0 * std::numbers::pi * 1760.0 * t);
    }
    return sig;
}

} // namespace

int main(int argc, char** argv) {
    stft::MPIContext ctx(argc, argv);

    constexpr std::size_t   FRAME_SIZE  = 1024;
    constexpr std::size_t   HOP_SIZE    = 512;
    constexpr std::uint32_t SAMPLE_RATE = 44100;

    std::vector<double> signal;
    std::uint32_t sampleRate = SAMPLE_RATE;

    if (ctx.isRoot()) {
        if (argc > 1) {
            try {
                signal = stft::WavReader::load(argv[1], sampleRate);
                std::cout << "[root] Loaded WAV: " << argv[1]
                          << "  samples=" << signal.size()
                          << "  sampleRate=" << sampleRate << " Hz\n";
            } catch (const std::exception& e) {
                std::cerr << "[root] Failed to load WAV (" << e.what()
                          << "), using synthetic signal.\n";
            }
        }
        if (signal.empty()) {
            signal = syntheticSignal(2 * SAMPLE_RATE, SAMPLE_RATE);
            std::cout << "[root] Using synthetic signal: "
                      << signal.size() << " samples at " << SAMPLE_RATE << " Hz\n";
        }
    }

    stft::MPI_STFTAnalyzer<stft::IterativeFFT, stft::HannWindow> analyzer(
        ctx, FRAME_SIZE, HOP_SIZE, sampleRate);

    ctx.barrier();
    const stft::SpectrogramData spec = analyzer.analyze(signal);

    if (ctx.isRoot()) {
        std::cout << "\n=== Spectrogram ===\n"
                  << "  Frames:      " << spec.numFrames  << "\n"
                  << "  Freq bins:   " << spec.numBins    << "\n"
                  << "  Frame size:  " << spec.frameSize  << " samples\n"
                  << "  Hop size:    " << spec.hopSize    << " samples\n"
                  << "  Sample rate: " << spec.sampleRate << " Hz\n"
                  << "  Duration:    " << spec.frameTime(spec.numFrames) << " s\n";

        if (!spec.empty()) {
            std::cout << "\nFirst frame, first 5 bins:\n";
            const std::size_t nShow = std::min<std::size_t>(5, spec.numBins);
            for (std::size_t b = 0; b < nShow; ++b)
                std::cout << "  bin " << b
                          << "  (" << spec.binFrequency(b) << " Hz)"
                          << "  mag=" << spec.at(0, b) << "\n";
        }
        std::cout << "\nRanks used: " << ctx.size() << "\n";
    }

    return 0;
}
