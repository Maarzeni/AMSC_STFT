/**
 * @file test_ImageExporter.cpp
 * @brief Validates ImageExporter::exportPNG() against the PNG file it writes.
 *
 * @details
 * ImageExporter has a single public entry point that writes a file to disk,
 * so these tests are black-box: they call exportPNG(), then read the bytes
 * back and check them against the PNG/zlib/DEFLATE specifications directly,
 * without depending on any of ImageExporter's internal (anonymous-namespace)
 * helpers.
 *
 * The zlib stream ImageExporter writes always uses "stored" (uncompressed)
 * DEFLATE blocks, which are trivial to reverse (length-prefixed raw byte
 * copies, no Huffman coding). That makes it possible to reconstruct the
 * exact scanline bytes and check actual pixel colours below, not just the
 * file's outer framing.
 */

#include <gtest/gtest.h>
#include "output/ImageExporter.hpp"
#include "stft/SpectrogramData.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace stft;

namespace {

/// Reads a whole binary file into memory.
std::vector<std::uint8_t> readFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}

std::uint32_t readBE32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8)  |
            static_cast<std::uint32_t>(p[3]);
}

struct PngChunk {
    std::string type;
    std::vector<std::uint8_t> data;
};

/// Walks the chunk sequence that follows the 8-byte PNG signature.
std::vector<PngChunk> parseChunks(const std::vector<std::uint8_t>& png) {
    std::vector<PngChunk> chunks;
    std::size_t pos = 8;  // skip signature
    while (pos + 8 <= png.size()) {
        const std::uint32_t len = readBE32(&png[pos]);
        std::string type(png.begin() + static_cast<long>(pos) + 4,
                         png.begin() + static_cast<long>(pos) + 8);
        const std::size_t dataStart = pos + 8;
        chunks.push_back({std::move(type),
                          std::vector<std::uint8_t>(
                              png.begin() + static_cast<long>(dataStart),
                              png.begin() + static_cast<long>(dataStart + len))});
        pos = dataStart + len + 4;  // data + trailing CRC32
    }
    return chunks;
}

const PngChunk* findChunk(const std::vector<PngChunk>& chunks, const std::string& type) {
    const auto it = std::find_if(chunks.begin(), chunks.end(),
        [&](const PngChunk& c) { return c.type == type; });
    return (it != chunks.end()) ? &*it : nullptr;
}

/**
 * @brief Reverses the "stored" (uncompressed) DEFLATE framing inside a zlib
 *        stream: a 2-byte zlib header, then one or more blocks of
 *        [1 flag byte][2-byte LE length][2-byte LE ~length][raw bytes].
 *        Only valid because ImageExporter never actually compresses; a real
 *        DEFLATE stream would need a real decompressor.
 */
std::vector<std::uint8_t> decodeStoredZlib(const std::vector<std::uint8_t>& z) {
    std::vector<std::uint8_t> out;
    std::size_t pos = 2;  // skip zlib header (CMF, FLG)
    bool isFinal = false;
    while (!isFinal && pos < z.size()) {
        isFinal = (z[pos] & 0x01) != 0;
        ++pos;
        const auto len = static_cast<std::uint16_t>(
            static_cast<unsigned>(z[pos]) | (static_cast<unsigned>(z[pos + 1]) << 8));
        pos += 4;  // len (2 bytes) + one's-complement ~len (2 bytes)
        out.insert(out.end(), z.begin() + static_cast<long>(pos),
                             z.begin() + static_cast<long>(pos + len));
        pos += len;
    }
    return out;
}

constexpr std::array<std::uint8_t, 8> kPngSignature =
    {137, 80, 78, 71, 13, 10, 26, 10};

/// Unique-per-test temp file path, cleaned up by the RAII wrapper below.
class TempPngFile {
public:
    explicit TempPngFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {}
    ~TempPngFile() { std::filesystem::remove(path_); }
    const std::filesystem::path& path() const { return path_; }
private:
    std::filesystem::path path_;
};

} // namespace

// ─── Error handling ─────────────────────────────────────────────────────────

TEST(ImageExporterTest, ThrowsOnEmptySpectrogram) {
    const SpectrogramData empty;  // default-constructed: no frames, no bins
    TempPngFile file("amsc_stft_test_empty.png");

    EXPECT_THROW(ImageExporter::exportPNG(empty, file.path().string()), std::runtime_error);
}

TEST(ImageExporterTest, ThrowsWhenFileCannotBeOpened) {
    SpectrogramData spec;
    spec.numFrames = 1;
    spec.numBins   = 1;
    spec.magnitudes = {1.0};

    const auto badPath = std::filesystem::temp_directory_path() /
                          "amsc_stft_missing_dir_xyz" / "out.png";

    EXPECT_THROW(ImageExporter::exportPNG(spec, badPath.string()), std::runtime_error);
}

// ─── PNG framing ────────────────────────────────────────────────────────────

TEST(ImageExporterTest, WritesValidSignatureAndIHDRDimensions) {
    constexpr std::size_t numFrames = 3;
    constexpr std::size_t numBins   = 5;

    SpectrogramData spec;
    spec.numFrames = numFrames;
    spec.numBins   = numBins;
    spec.magnitudes.resize(numFrames * numBins);
    for (std::size_t i = 0; i < spec.magnitudes.size(); ++i)
        spec.magnitudes[i] = static_cast<double>(i) + 0.5;

    TempPngFile file("amsc_stft_test_dims.png");
    ImageExporter::exportPNG(spec, file.path().string());

    const auto bytes = readFile(file.path());
    ASSERT_GE(bytes.size(), kPngSignature.size());
    EXPECT_TRUE(std::equal(kPngSignature.begin(), kPngSignature.end(), bytes.begin()));

    const auto chunks = parseChunks(bytes);
    const PngChunk* ihdr = findChunk(chunks, "IHDR");
    ASSERT_NE(ihdr, nullptr);
    ASSERT_EQ(ihdr->data.size(), 13u);

    EXPECT_EQ(readBE32(&ihdr->data[0]), numFrames);  // width  = time axis
    EXPECT_EQ(readBE32(&ihdr->data[4]), numBins);    // height = frequency axis
    EXPECT_EQ(ihdr->data[8], 8);   // bit depth
    EXPECT_EQ(ihdr->data[9], 2);   // colour type: RGB

    ASSERT_NE(findChunk(chunks, "IDAT"), nullptr);
    ASSERT_NE(findChunk(chunks, "IEND"), nullptr);
}

// ─── Pixel content ──────────────────────────────────────────────────────────

// Verifies the two trickiest pieces of exportPNG() at once: the dB/heatmap
// colour mapping (loud -> bright, quiet -> dark) and the vertical flip that
// puts the highest frequency bin in row 0. A 1-frame, 2-bin spectrogram with
// one loud and one silent bin makes both checkable with exact expected
// colours instead of a plausibility range.
TEST(ImageExporterTest, PixelColorsMatchHeatmapAndFrequencyAxisIsFlipped) {
    SpectrogramData spec;
    spec.numFrames = 1;
    spec.numBins   = 2;
    spec.magnitudes = {1.0, 0.0};  // bin 0 (DC) loud, bin 1 (Nyquist) silent

    TempPngFile file("amsc_stft_test_pixels.png");
    ImageExporter::exportPNG(spec, file.path().string());

    const auto bytes  = readFile(file.path());
    const auto chunks = parseChunks(bytes);
    const PngChunk* idat = findChunk(chunks, "IDAT");
    ASSERT_NE(idat, nullptr);

    const auto raw = decodeStoredZlib(idat->data);

    // Each scanline: 1 filter byte + 1 pixel * 3 RGB bytes.
    ASSERT_EQ(raw.size(), 2 * (1 + 3));

    // Row 0 (top) = bin 1 = Nyquist = silent -> black.
    EXPECT_EQ(raw[0], 0);  // filter type: None
    EXPECT_NEAR(raw[1], 0, 1);
    EXPECT_NEAR(raw[2], 0, 1);
    EXPECT_NEAR(raw[3], 0, 1);

    // Row 1 (bottom) = bin 0 = DC = loudest bin -> white.
    EXPECT_EQ(raw[4], 0);  // filter type: None
    EXPECT_NEAR(raw[5], 255, 1);
    EXPECT_NEAR(raw[6], 255, 1);
    EXPECT_NEAR(raw[7], 255, 1);
}

TEST(ImageExporterTest, AllZeroSpectrogramRendersFullyBlack) {
    SpectrogramData spec;
    spec.numFrames = 1;
    spec.numBins   = 1;
    spec.magnitudes = {0.0};  // no signal at all -> peak is degenerate

    TempPngFile file("amsc_stft_test_silence.png");
    ImageExporter::exportPNG(spec, file.path().string());

    const auto bytes  = readFile(file.path());
    const auto chunks = parseChunks(bytes);
    const auto raw    = decodeStoredZlib(findChunk(chunks, "IDAT")->data);

    ASSERT_EQ(raw.size(), 1 + 3);
    EXPECT_NEAR(raw[1], 0, 1);
    EXPECT_NEAR(raw[2], 0, 1);
    EXPECT_NEAR(raw[3], 0, 1);
}
