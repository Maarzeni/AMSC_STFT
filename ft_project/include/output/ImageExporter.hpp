/**
 * @file ImageExporter.hpp
 * @brief Exports a SpectrogramData matrix as a PNG spectrogram image.
 *
 * @details
 * The PNG is produced by a self-contained encoder (no libpng, no stb) that
 * writes uncompressed DEFLATE blocks wrapped in a standards-correct zlib stream.
 * All checksums (CRC32 for PNG chunks, Adler32 for zlib) are computed in code.
 *
 * Image layout:
 *  - Width  = number of time frames
 *  - Height = number of frequency bins
 *  - Row 0 (top)    = highest frequency bin (Nyquist)
 *  - Row height-1   = DC bin (0 Hz)
 *  - Each column corresponds to one STFT frame (left = early, right = late)
 *
 * Magnitude-to-colour mapping:
 *  - Magnitudes are converted to dB (20*log10(mag/peak + ε))
 *  - Clamped to [-dynamicRangeDb, 0]
 *  - Normalised to [0,1] and mapped through a heat colormap:
 *      black → purple → red → orange → yellow → white
 */

#pragma once

#include "stft/SpectrogramData.hpp"
#include <string>

namespace stft {

class ImageExporter {
public:
    /**
     * @brief Render a spectrogram to a PNG file.
     *
     * @param spec             The spectrogram to export.
     * @param filename         Output PNG file path.
     * @param dynamicRangeDb   Colour dynamic range in dB (default 80 dB).
     *                         Bins more than this many dB below the peak are
     *                         rendered as black.
     * @throws std::runtime_error if spec is empty, the file cannot be opened,
     *         or an I/O error occurs during writing.
     */
    static void exportPNG(const SpectrogramData& spec,
                          const std::string& filename,
                          double dynamicRangeDb = 80.0);
};

} // namespace stft
