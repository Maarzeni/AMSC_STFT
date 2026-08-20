/**
 * @file mpi_main.cpp
 * @brief Distributed STFT demonstration — WAV in, PNG spectrogram out.
 *
 * @details
 * The frames are block-distributed over the MPI ranks, each rank transforms
 * its block with OpenMP inside, and the root rank gathers the full spectrogram
 * and writes it out.  This is `main.cpp` with MPI_STFTAnalyzer in place of
 * STFTAnalyzer, and is kept deliberately parallel to it, line by line.
 *
 * Usage:
 *   mpirun -np <P> ./mpi_main [audio.wav] [frameSize=1024] [hopSize=512] [window]
 *
 * Every argument is optional.  A bare file name is looked up in
 * core/examples/data/, and with no argument at all the bundled test_audio.wav
 * is analysed; if the file cannot be read, a synthetic signal is used instead.
 *
 * The spectrogram is written by the root rank to results/results_examples/ as
 *   <stem>_<window>_f<frameSize>_h<hopSize>_mpi<P>.png
 *
 * Examples:
 *   mpirun -np 4 ./mpi_main
 *   mpirun -np 4 ./mpi_main scale.wav 2048 1024 blackman
 *   OMP_NUM_THREADS=4 mpirun -np 2 ./mpi_main test_audio.wav
 */

#include "mpi/MPIContext.hpp"
#include "stft/MPI_STFTAnalyzer.hpp"
#include "stft/SpectrogramData.hpp"
#include "output/ImageExporter.hpp"
#include "audio/WavReader.hpp"
#include "fft/IterativeFFT.hpp"
#include "window/HannWindow.hpp"
#include "window/HammingWindow.hpp"
#include "window/BlackmanWindow.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

// Absolute paths baked in by CMake (see examples/CMakeLists.txt), so the
// program finds its audio and writes its output whatever directory it is
// launched from.  The fallbacks only matter for a hand-rolled build.
#ifndef EXAMPLES_DATA_DIR
#define EXAMPLES_DATA_DIR "../examples/data"
#endif
#ifndef EXAMPLES_OUTPUT_DIR
#define EXAMPLES_OUTPUT_DIR "results/results_examples"
#endif

using namespace stft;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void printUsage(const char* prog) {
    std::cerr
        << "Usage: mpirun -np <P> " << prog
        << " [audio.wav] [frameSize=1024] [hopSize=512]"
           " [window=hann|hamming|blackman]\n"
        << "\n"
        << "  audio.wav   mono WAV file; a bare name is looked up in\n"
        << "              " << EXAMPLES_DATA_DIR << " (default: test_audio.wav)\n"
        << "  frameSize   STFT frame size, must be a power of two (default 1024)\n"
        << "  hopSize     hop between frames in samples (default 512)\n"
        << "  window      windowing function: hann (default), hamming, blackman\n"
        << "\n"
        << "The PNG spectrogram is written to " << EXAMPLES_OUTPUT_DIR << "\n"
        << "(override with $STFT_EXAMPLES_DIR).\n";
}

static bool isPowerOfTwo(std::size_t n) {
    return n >= 2 && (n & (n - 1)) == 0;
}

/// A bare file name resolves against the bundled audio directory.
static std::string resolveInput(const std::string& arg) {
    if (arg.find('/') != std::string::npos) return arg;
    return std::string(EXAMPLES_DATA_DIR) + "/" + arg;
}

/// $STFT_EXAMPLES_DIR, else the path baked in by CMake, else ./results/
/// results_examples — the last one for a read-only source tree (container).
static std::filesystem::path outputDir() {
    namespace fs = std::filesystem;
    const char* env = std::getenv("STFT_EXAMPLES_DIR");
    fs::path dir = (env && *env) ? fs::path(env) : fs::path(EXAMPLES_OUTPUT_DIR);

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (!fs::is_directory(dir, ec)) {
        dir = fs::current_path() / "results" / "results_examples";
        fs::create_directories(dir, ec);
    }
    return dir;
}

/// Fallback signal when no WAV can be read: three partials an octave apart.
static std::vector<double> syntheticSignal(std::size_t numSamples, std::uint32_t sampleRate) {
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

// ── Per-window run() instantiates the correct template parameter ──────────────

template<typename Window>
int run(const MPIContext&  ctx,
        const std::vector<double>& signal,
        std::uint32_t      sampleRate,
        std::size_t        frameSize,
        std::size_t        hopSize,
        const std::string& window,
        const std::string& stem)
{
    const double fs = static_cast<double>(sampleRate);

    const MPI_STFTAnalyzer<IterativeFFT, Window> analyzer(
        ctx, frameSize, hopSize, sampleRate);

    if (ctx.isRoot())
        std::cout << "Computing STFT: "
                  << STFTAnalyzer<IterativeFFT, Window>::numFrames(
                         signal.size(), frameSize, hopSize)
                  << " frames over " << ctx.size() << " MPI ranks...\n";

    // analyze() is collective, so every rank reaches it and a signal too short
    // for one frame is diagnosed afterwards, on the empty result, rather than
    // by an early return that would leave the other ranks blocked.
    ctx.barrier();
    const auto t0 = std::chrono::steady_clock::now();
    const SpectrogramData spec = analyzer.analyze(signal);
    const auto t1 = std::chrono::steady_clock::now();

    if (!ctx.isRoot()) return 0;     // only root holds the gathered spectrogram

    if (spec.empty()) {
        std::cerr << "Error: signal is shorter than one frame ("
                  << frameSize << " samples).\n";
        return 1;
    }

    // Dominant frequency of the middle frame: the one number that shows the
    // transform actually looked at the signal.
    const std::size_t mid = spec.numFrames / 2;
    std::size_t peakBin = 0;
    double      peakMag = 0.0;
    for (std::size_t b = 0; b < spec.numBins; ++b) {
        if (spec.at(mid, b) > peakMag) {
            peakMag = spec.at(mid, b);
            peakBin = b;
        }
    }

    const std::filesystem::path png =
        outputDir() / (stem + "_mpi" + std::to_string(ctx.size()) + ".png");
    ImageExporter::exportPNG(spec, png.string());

    std::cout << "\n=== STFT analysis (MPI + OpenMP) ========================\n"
              << "  Sample rate    : " << sampleRate << " Hz\n"
              << "  Signal length  : " << signal.size() << " samples ("
                                       << signal.size() / fs << " s)\n"
              << "  Frame / hop    : " << frameSize << " / " << hopSize << " samples\n"
              << "  Window         : " << window << "\n"
              << "  Frames x bins  : " << spec.numFrames << " x " << spec.numBins << "\n"
              << "  Time res.      : " << 1000.0 * frameSize / fs << " ms per frame\n"
              << "  Freq. res.     : " << fs / frameSize << " Hz per bin\n"
              << "  Dominant freq. : " << spec.binFrequency(peakBin)
                                       << " Hz (middle frame)\n"
              << "  MPI ranks      : " << ctx.size() << "\n"
              << "  Analysis time  : "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n"
              << "  Spectrogram    : " << png.string() << "\n"
              << "=========================================================\n";
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    MPIContext ctx(argc, argv);

    // Every rank parses the same argv, so the frame layout needs no broadcast.
    std::string wavPath   = resolveInput(argc >= 2 ? argv[1] : "test_audio.wav");
    std::size_t frameSize = 1024;
    std::size_t hopSize   = 512;
    std::string window    = "hann";

    if (argc >= 3) {
        try { frameSize = static_cast<std::size_t>(std::stoul(argv[2])); }
        catch (...) {
            if (ctx.isRoot()) {
                std::cerr << "Error: invalid frameSize '" << argv[2] << "'.\n";
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (argc >= 4) {
        try { hopSize = static_cast<std::size_t>(std::stoul(argv[3])); }
        catch (...) {
            if (ctx.isRoot()) {
                std::cerr << "Error: invalid hopSize '" << argv[3] << "'.\n";
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (argc >= 5) {
        window = argv[4];
        std::transform(window.begin(), window.end(), window.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    }

    // Validate.  Every rank checks the same arguments and stops together.
    if (!isPowerOfTwo(frameSize)) {
        if (ctx.isRoot())
            std::cerr << "Error: frameSize must be a power of two >= 2 (got "
                      << frameSize << ").\n";
        return 1;
    }
    if (hopSize < 1) {
        if (ctx.isRoot()) std::cerr << "Error: hopSize must be >= 1.\n";
        return 1;
    }
    if (window != "hann" && window != "hamming" && window != "blackman") {
        if (ctx.isRoot())
            std::cerr << "Error: unknown window '" << window
                      << "'. Choose hann, hamming, or blackman.\n";
        return 1;
    }

    // Only the root reads the WAV; MPI_STFTAnalyzer distributes it from there.
    // The other ranks pass an empty signal and receive their slice inside
    // analyze().
    std::uint32_t sampleRate = 44100;
    std::vector<double> signal;
    std::string stem = std::filesystem::path(wavPath).stem().string();

    if (ctx.isRoot()) {
        try {
            auto wav   = WavReader::load(wavPath);
            signal     = std::move(wav.samples);
            sampleRate = wav.sampleRate;
            std::cout << "Input: " << wavPath << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Note: cannot use '" << wavPath << "' (" << e.what()
                      << ") — analysing a synthetic signal instead.\n";
            sampleRate = 44100;
            signal     = syntheticSignal(2 * sampleRate, sampleRate);
            stem       = "synthetic";
        }
    }

    stem += "_" + window + "_f" + std::to_string(frameSize)
                 + "_h" + std::to_string(hopSize);

    try {
        if      (window == "hann")    return run<HannWindow>    (ctx, signal, sampleRate, frameSize, hopSize, window, stem);
        else if (window == "hamming") return run<HammingWindow> (ctx, signal, sampleRate, frameSize, hopSize, window, stem);
        else                          return run<BlackmanWindow>(ctx, signal, sampleRate, frameSize, hopSize, window, stem);
    } catch (const std::exception& e) {
        std::cerr << "[rank " << ctx.rank() << "] Error: " << e.what() << "\n";
        return 1;
    }
}
