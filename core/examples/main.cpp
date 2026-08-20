/**
 * @file main.cpp
 * @brief Shared-memory STFT demonstration: WAV in, PNG spectrogram out.
 *
 * @details
 * The frame loop is parallelised with OpenMP inside STFTAnalyzer, so one run
 * uses every core of a single machine.  `mpi_main.cpp` is the same program with
 * MPI_STFTAnalyzer in place of STFTAnalyzer, and is kept parallel to this one
 * line by line so that the two can be read side by side.
 *
 * @par Usage
 * @code
 * ./main [audio.wav] [frameSize=1024] [hopSize=512] [window=hann|hamming|blackman]
 * @endcode
 * Every argument is optional.  A bare file name is looked up in
 * `core/examples/data/`, and with no argument at all the bundled
 * `test_audio.wav` is analysed; if the file cannot be read, a synthetic signal
 * is analysed instead.
 *
 * The spectrogram is written to `results/results_examples/` as
 * `<stem>_<window>_f<frameSize>_h<hopSize>_serial.png`.
 *
 * @par Examples
 * @code
 * ./main
 * ./main scale.wav 2048 1024 blackman
 * OMP_NUM_THREADS=8 ./main test_audio.wav 1024 256 hamming
 * @endcode
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
#include <bit>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

/**
 * @def EXAMPLES_DATA_DIR
 * @brief Directory a bare input file name is resolved against.
 *
 * Injected as an absolute path by CMake (see examples/CMakeLists.txt) so the
 * binary finds its input from any working directory.  The fallback below only
 * applies to a hand-rolled build.
 */
#ifndef EXAMPLES_DATA_DIR
#define EXAMPLES_DATA_DIR "../examples/data"
#endif

/**
 * @def EXAMPLES_OUTPUT_DIR
 * @brief Default directory for the PNG spectrogram, injected by CMake.
 */
#ifndef EXAMPLES_OUTPUT_DIR
#define EXAMPLES_OUTPUT_DIR "results/results_examples"
#endif

using namespace stft;

namespace {

constexpr std::string_view kDataDir   = EXAMPLES_DATA_DIR;
constexpr std::string_view kOutputDir = EXAMPLES_OUTPUT_DIR;

/// Environment variable that overrides ::kOutputDir at run time.
constexpr const char* kOutputDirEnv = "STFT_EXAMPLES_DIR";

constexpr std::string_view kDefaultInput      = "test_audio.wav";
constexpr std::string_view kDefaultWindow     = "hann";
constexpr std::size_t      kDefaultFrameSize  = 1024;
constexpr std::size_t      kDefaultHopSize    = 512;
constexpr std::uint32_t    kFallbackSampleRate = 44100;

/// Validated command-line settings for one analysis.
struct Config {
    std::string input;      ///< Path to the WAV file to analyse.
    std::size_t frameSize;  ///< STFT frame size in samples; a power of two.
    std::size_t hopSize;    ///< Hop between successive frames, in samples.
    std::string window;     ///< Window name: "hann", "hamming" or "blackman".
};

/// Prints the command-line synopsis on std::cerr.
void printUsage(std::string_view prog) {
    std::cerr
        << "Usage: " << prog
        << " [audio.wav] [frameSize=1024] [hopSize=512]"
           " [window=hann|hamming|blackman]\n"
        << "\n"
        << "  audio.wav   mono WAV file; a bare name is looked up in\n"
        << "              " << kDataDir << " (default: " << kDefaultInput << ")\n"
        << "  frameSize   STFT frame size, must be a power of two (default 1024)\n"
        << "  hopSize     hop between frames in samples (default 512)\n"
        << "  window      windowing function: hann (default), hamming, blackman\n"
        << "\n"
        << "The PNG spectrogram is written to " << kOutputDir << "\n"
        << "(override with $" << kOutputDirEnv << ").\n";
}

/**
 * @brief Parses a whole non-negative decimal integer.
 * @param arg Text to parse; must be digits only, with no sign and no suffix.
 * @return The parsed value, or std::nullopt if @p arg is not such an integer.
 */
[[nodiscard]] std::optional<std::size_t> parseSize(std::string_view arg) {
    std::size_t value = 0;
    const auto* const last = arg.data() + arg.size();
    const auto [end, ec] = std::from_chars(arg.data(), last, value);
    if (ec != std::errc{} || end != last) return std::nullopt;
    return value;
}

/// True if @p n is a power of two of at least 2, as the FFT requires.
[[nodiscard]] bool isPowerOfTwo(std::size_t n) noexcept {
    return n >= 2 && std::has_single_bit(n);
}

/// True if @p name is one of the window functions this example can instantiate.
[[nodiscard]] bool isKnownWindow(std::string_view name) noexcept {
    return name == "hann" || name == "hamming" || name == "blackman";
}

/// Resolves a bare file name against ::kDataDir; leaves any path unchanged.
[[nodiscard]] std::string resolveInput(std::string_view arg) {
    const std::filesystem::path given(arg);
    if (given.has_parent_path()) return given.string();
    return (std::filesystem::path(kDataDir) / given).string();
}

/**
 * @brief Locates the output directory and creates it if needed.
 * @return $STFT_EXAMPLES_DIR, else ::kOutputDir, else `./results/results_examples`
 *         — the last one covers a read-only source tree, as in a container.
 */
[[nodiscard]] std::filesystem::path outputDir() {
    namespace fs = std::filesystem;

    fs::path dir{kOutputDir};
    if (const char* env = std::getenv(kOutputDirEnv); env != nullptr && *env != '\0')
        dir = fs::path(env);

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (!fs::is_directory(dir, ec)) {
        dir = fs::current_path() / "results" / "results_examples";
        fs::create_directories(dir, ec);
    }
    return dir;
}

/**
 * @brief Fallback signal used when no WAV can be read.
 * @param numSamples Length of the generated signal, in samples.
 * @param sampleRate Sample rate the partials are generated at, in Hz.
 * @return Three partials an octave apart (440, 880 and 1760 Hz).
 */
[[nodiscard]] std::vector<double> syntheticSignal(std::size_t numSamples,
                                                  std::uint32_t sampleRate) {
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

/**
 * @brief Analyses @p signal, writes the PNG spectrogram and prints a report.
 *
 * @tparam Window Window function to instantiate STFTAnalyzer with.
 * @param signal     Samples to analyse.
 * @param sampleRate Sample rate of @p signal, in Hz.
 * @param cfg        Validated command-line settings.
 * @param stem       Output file name without directory or `.png` extension.
 * @return 0 on success, 1 if @p signal is shorter than a single frame.
 */
template<typename Window>
[[nodiscard]] int run(const std::vector<double>& signal,
                      std::uint32_t      sampleRate,
                      const Config&      cfg,
                      const std::string& stem)
{
    const double fs = static_cast<double>(sampleRate);

    const std::size_t totalFrames =
        STFTAnalyzer<IterativeFFT, Window>::numFrames(signal.size(),
                                                      cfg.frameSize, cfg.hopSize);
    if (totalFrames == 0) {
        std::cerr << "Error: signal is shorter than one frame ("
                  << cfg.frameSize << " samples).\n";
        return 1;
    }

    const STFTAnalyzer<IterativeFFT, Window> analyzer(cfg.frameSize, cfg.hopSize,
                                                      sampleRate);

    std::cout << "Computing STFT: " << totalFrames << " frames...\n";
    const auto t0 = std::chrono::steady_clock::now();
    const SpectrogramData spec = analyzer.analyze(signal);
    const auto t1 = std::chrono::steady_clock::now();

    // Dominant bin of the middle frame: the one number in the report that shows
    // the transform followed the signal.  Rows are contiguous in the flat
    // magnitude buffer, so the frame can be viewed as a span without copying.
    const std::size_t mid = spec.numFrames / 2;
    const std::span<const double> midFrame(
        spec.magnitudes.data() + mid * spec.numBins, spec.numBins);
    const std::size_t peakBin = static_cast<std::size_t>(
        std::ranges::max_element(midFrame) - midFrame.begin());

    const std::filesystem::path png = outputDir() / (stem + "_serial.png");
    ImageExporter::exportPNG(spec, png.string());

    std::cout << "\n=== STFT analysis (serial + OpenMP) =====================\n"
              << "  Sample rate    : " << sampleRate << " Hz\n"
              << "  Signal length  : " << signal.size() << " samples ("
                                       << signal.size() / fs << " s)\n"
              << "  Frame / hop    : " << cfg.frameSize << " / " << cfg.hopSize
                                       << " samples\n"
              << "  Window         : " << cfg.window << "\n"
              << "  Frames x bins  : " << spec.numFrames << " x " << spec.numBins << "\n"
              << "  Time res.      : " << 1000.0 * cfg.frameSize / fs << " ms per frame\n"
              << "  Freq. res.     : " << fs / cfg.frameSize << " Hz per bin\n"
              << "  Dominant freq. : " << spec.binFrequency(peakBin)
                                       << " Hz (middle frame)\n"
              << "  Analysis time  : "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n"
              << "  Spectrogram    : " << png.string() << "\n"
              << "=========================================================\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::span args(argv, static_cast<std::size_t>(argc));

    std::string input     {kDefaultInput};
    std::string window    {kDefaultWindow};
    std::size_t frameSize = kDefaultFrameSize;
    std::size_t hopSize   = kDefaultHopSize;

    if (args.size() >= 2) input = args[1];
    if (args.size() >= 3) {
        const std::optional<std::size_t> parsed = parseSize(args[2]);
        if (!parsed) {
            std::cerr << "Error: invalid frameSize '" << args[2] << "'.\n";
            printUsage(args[0]);
            return 1;
        }
        frameSize = *parsed;
    }
    if (args.size() >= 4) {
        const std::optional<std::size_t> parsed = parseSize(args[3]);
        if (!parsed) {
            std::cerr << "Error: invalid hopSize '" << args[3] << "'.\n";
            printUsage(args[0]);
            return 1;
        }
        hopSize = *parsed;
    }
    if (args.size() >= 5) {
        window = args[4];
        std::ranges::transform(window, window.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    }

    if (!isPowerOfTwo(frameSize)) {
        std::cerr << "Error: frameSize must be a power of two >= 2 (got "
                  << frameSize << ").\n";
        return 1;
    }
    if (hopSize == 0) {
        std::cerr << "Error: hopSize must be >= 1.\n";
        return 1;
    }
    if (!isKnownWindow(window)) {
        std::cerr << "Error: unknown window '" << window
                  << "'. Choose hann, hamming, or blackman.\n";
        return 1;
    }

    const Config cfg{ .input     = resolveInput(input),
                      .frameSize = frameSize,
                      .hopSize   = hopSize,
                      .window    = std::move(window) };

    std::uint32_t sampleRate = kFallbackSampleRate;
    std::vector<double> signal;
    std::string stem;
    try {
        WavData wav = WavReader::load(cfg.input);
        signal     = std::move(wav.samples);
        sampleRate = wav.sampleRate;
        stem       = std::filesystem::path(cfg.input).stem().string();
        std::cout << "Input: " << cfg.input << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Note: cannot use '" << cfg.input << "' (" << e.what()
                  << ") — analysing a synthetic signal instead.\n";
        signal = syntheticSignal(2 * std::size_t{kFallbackSampleRate},
                                 kFallbackSampleRate);
        stem   = "synthetic";
    }

    stem += "_" + cfg.window + "_f" + std::to_string(cfg.frameSize)
                + "_h" + std::to_string(cfg.hopSize);

    try {
        if (cfg.window == "hann")    return run<HannWindow>    (signal, sampleRate, cfg, stem);
        if (cfg.window == "hamming") return run<HammingWindow> (signal, sampleRate, cfg, stem);
        return                              run<BlackmanWindow>(signal, sampleRate, cfg, stem);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
