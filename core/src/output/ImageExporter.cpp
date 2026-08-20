/**
 * @file ImageExporter.cpp
 * @brief Self-contained PNG spectrogram exporter.
 *
 * Implements a minimal PNG encoder that needs no external libraries.
 *
 * PNG structure written:
 *   PNG signature (8 bytes)
 *   IHDR chunk   (width, height, 8-bit RGB)
 *   IDAT chunk   (zlib stream of scanlines, each prefixed with filter byte 0x00)
 *   IEND chunk
 *
 * The zlib stream uses stored (uncompressed) DEFLATE blocks, which are valid
 * per RFC 1951 and RFC 1950.  CRC32 (PNG) and Adler32 (zlib) are computed
 * from first principles.
 */

#include "output/ImageExporter.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace stft {
namespace {

// ─── CRC32 (PNG chunk checksums) ───────────────────────────────────────────────

constexpr std::array<std::uint32_t, 256> makeCrc32Table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t n = 0; n < 256; ++n) {
        std::uint32_t c = n;
        for (int k = 0; k < 8; ++k)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        table[n] = c;
    }
    return table;
}

/// Computed once, at compile time: no runtime initialisation, no lazy-init
/// state to race on if this were ever called from more than one thread.
constexpr std::array<std::uint32_t, 256> kCrc32Table = makeCrc32Table();

std::uint32_t crc32(const std::uint8_t* data, std::size_t len,
                    std::uint32_t crc = 0xFFFFFFFFu) {
    for (std::size_t i = 0; i < len; ++i)
        crc = kCrc32Table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// ─── Adler32 (zlib checksum) ───────────────────────────────────────────────────

std::uint32_t adler32(const std::uint8_t* data, std::size_t len) {
    std::uint32_t s1 = 1, s2 = 0;
    for (std::size_t i = 0; i < len; ++i) {
        s1 = (s1 + data[i]) % 65521u;
        s2 = (s2 + s1)      % 65521u;
    }
    return (s2 << 16) | s1;
}

// ─── Big-endian / little-endian helpers ────────────────────────────────────────

void writeBE32(std::vector<std::uint8_t>& buf, std::uint32_t v) {
    buf.push_back(static_cast<std::uint8_t>(v >> 24));
    buf.push_back(static_cast<std::uint8_t>(v >> 16));
    buf.push_back(static_cast<std::uint8_t>(v >>  8));
    buf.push_back(static_cast<std::uint8_t>(v));
}

void writeBE16(std::vector<std::uint8_t>& buf, std::uint16_t v) {
    buf.push_back(static_cast<std::uint8_t>(v >> 8));
    buf.push_back(static_cast<std::uint8_t>(v));
}

/// DEFLATE's own LEN/NLEN fields are little-endian, unlike every other
/// length field in the PNG/zlib formats this file writes.
void writeLE16(std::vector<std::uint8_t>& buf, std::uint16_t v) {
    buf.push_back(static_cast<std::uint8_t>(v));
    buf.push_back(static_cast<std::uint8_t>(v >> 8));
}

// ─── PNG chunk builder ─────────────────────────────────────────────────────────

/// @brief Appends one PNG chunk (length + type + data + CRC32) to `out`.
void writeChunk(std::vector<std::uint8_t>& out,
                std::string_view type,
                const std::vector<std::uint8_t>& data)
{
    assert(type.size() == 4 && "PNG chunk types are exactly 4 bytes");
    writeBE32(out, static_cast<std::uint32_t>(data.size()));
    const std::size_t typeStart = out.size();
    out.insert(out.end(), type.begin(), type.end());
    out.insert(out.end(), data.begin(), data.end());
    const std::uint32_t checksum =
        crc32(out.data() + typeStart, type.size() + data.size());
    writeBE32(out, checksum);
}

// ─── Zlib stored-block encoder ─────────────────────────────────────────────────

/**
 * @brief Wraps `raw` in a valid zlib stream (CM=8, CINFO for window size,
 *        FCHECK, then one or more stored DEFLATE blocks, then Adler32) —
 *        without compressing it.
 */
std::vector<std::uint8_t> zlibStored(const std::uint8_t* raw, std::size_t rawLen) {
    std::vector<std::uint8_t> out;

    // zlib header: CMF=0x78 (deflate, window=32768), FLG such that CMF*256+FLG
    // is divisible by 31.  0x78*256 = 30720; 30720 % 31 = 0, so FLG=0x01.
    out.push_back(0x78);
    out.push_back(0x01);

    // Stored DEFLATE blocks: each block holds up to 65535 bytes.
    constexpr std::size_t BLOCK = 65535;
    std::size_t remaining = rawLen;
    const std::uint8_t* ptr = raw;

    while (remaining > 0) {
        const std::size_t blockLen = std::min(remaining, BLOCK);
        const bool isFinal = (blockLen == remaining);
        const auto len  = static_cast<std::uint16_t>(blockLen);
        const auto nlen = static_cast<std::uint16_t>(~len);

        out.push_back(isFinal ? 0x01u : 0x00u);  // BFINAL | BTYPE=00 (stored)
        writeLE16(out, len);
        writeLE16(out, nlen);

        out.insert(out.end(), ptr, ptr + blockLen);
        ptr       += blockLen;
        remaining -= blockLen;
    }

    // zlib Adler32 trailer (big-endian).
    writeBE32(out, adler32(raw, rawLen));

    return out;
}

// ─── Heat colormap ─────────────────────────────────────────────────────────────

/// black → purple → red → orange → yellow → white.
std::array<std::uint8_t, 3> heatmap(double t) {
    t = std::clamp(t, 0.0, 1.0);

    struct Stop { double pos; std::uint8_t r, g, b; };
    static constexpr std::array<Stop, 5> stops = {{
        {0.00,   0,   0,   0},   // black
        {0.25,  80,   0, 120},   // purple
        {0.50, 220,  30,   0},   // red
        {0.75, 255, 160,   0},   // orange/yellow
        {1.00, 255, 255, 255},   // white
    }};

    std::size_t i = 0;
    while (i < stops.size() - 2 && t > stops[i + 1].pos) ++i;
    const double lo = stops[i].pos, hi = stops[i + 1].pos;
    const double f  = (hi > lo) ? (t - lo) / (hi - lo) : 0.0;

    auto lerp = [&](std::uint8_t a, std::uint8_t b) -> std::uint8_t {
        return static_cast<std::uint8_t>(a + f * (static_cast<double>(b) - a));
    };
    return {lerp(stops[i].r, stops[i+1].r),
            lerp(stops[i].g, stops[i+1].g),
            lerp(stops[i].b, stops[i+1].b)};
}

} // anonymous namespace

// ─── Public API ────────────────────────────────────────────────────────────────

void ImageExporter::exportPNG(const SpectrogramData& spec,
                              const std::string& filename,
                              double dynamicRangeDb)
{
    if (spec.empty() || spec.numFrames == 0 || spec.numBins == 0)
        throw std::runtime_error("ImageExporter::exportPNG: spectrogram is empty.");

    const std::size_t W = spec.numFrames;  // image width  (time axis)
    const std::size_t H = spec.numBins;    // image height (frequency axis)

    // ── Convert magnitudes to normalised dB ───────────────────────────────────
    double peak = 0.0;
    for (double v : spec.magnitudes)
        peak = std::max(peak, v);
    if (peak == 0.0) peak = 1.0;  // all-zero spectrogram → render as black

    // Build a [H x W] matrix of normalised values in [0,1].
    // Row 0 in the image = highest frequency bin (bin H-1), so we flip vertically.
    std::vector<double> normalised(H * W);
    for (std::size_t f = 0; f < W; ++f) {
        for (std::size_t b = 0; b < H; ++b) {
            const double mag = spec.magnitudes[f * H + b];
            const double db  = 20.0 * std::log10(mag / peak + 1e-12);
            const double t   = 1.0 - std::clamp(-db / dynamicRangeDb, 0.0, 1.0);
            // Flip: row 0 = top = Nyquist = bin H-1
            normalised[(H - 1 - b) * W + f] = t;
        }
    }

    // ── Build raw scanline bytes (filter byte 0x00 + RGB pixels per row) ──────
    // Each scanline: 1 filter byte + W*3 pixel bytes
    const std::size_t scanlineBytes = 1 + W * 3;
    std::vector<std::uint8_t> rawData(H * scanlineBytes);

    for (std::size_t row = 0; row < H; ++row) {
        std::uint8_t* scanline = rawData.data() + row * scanlineBytes;
        scanline[0] = 0x00;  // per-scanline filter TYPE: None (no predictor)
        for (std::size_t col = 0; col < W; ++col) {
            const auto [r, g, b] = heatmap(normalised[row * W + col]);
            scanline[1 + col * 3 + 0] = r;
            scanline[1 + col * 3 + 1] = g;
            scanline[1 + col * 3 + 2] = b;
        }
    }

    // ── Assemble PNG bytes ────────────────────────────────────────────────────
    std::vector<std::uint8_t> png;
    png.reserve(rawData.size() + 512);

    // PNG signature
    constexpr std::array<std::uint8_t, 8> sig = {137, 80, 78, 71, 13, 10, 26, 10};
    png.insert(png.end(), sig.begin(), sig.end());

    // IHDR
    {
        std::vector<std::uint8_t> ihdr;
        writeBE32(ihdr, static_cast<std::uint32_t>(W));
        writeBE32(ihdr, static_cast<std::uint32_t>(H));
        ihdr.push_back(8);    // bit depth
        ihdr.push_back(2);    // colour type: RGB
        ihdr.push_back(0);    // compression method: deflate (the only value PNG allows)
        ihdr.push_back(0);    // filter method: adaptive (the only value PNG allows)
        ihdr.push_back(0);    // interlace method: none
        writeChunk(png, "IHDR", ihdr);
    }

    // IDAT
    {
        const std::vector<std::uint8_t> compressed =
            zlibStored(rawData.data(), rawData.size());
        writeChunk(png, "IDAT", compressed);
    }

    // IEND
    writeChunk(png, "IEND", {});

    // ── Write to disk ─────────────────────────────────────────────────────────
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error(
            "ImageExporter::exportPNG: cannot open '" + filename + "' for writing.");

    out.write(reinterpret_cast<const char*>(png.data()),
              static_cast<std::streamsize>(png.size()));
    if (!out)
        throw std::runtime_error(
            "ImageExporter::exportPNG: I/O error writing '" + filename + "'.");
}

} // namespace stft
