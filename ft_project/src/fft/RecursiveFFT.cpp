#include "fft/RecursiveFFT.hpp"
#include <cmath>
#include <numbers> // Richiede C++20 (std::numbers::pi)

namespace amsc_stft {

void RecursiveFFT::forward_impl(std::vector<std::complex<double>>& data) {
    if (data.empty()) return;
    
    // La validazione della potenza di 2 è già stata fatta da BaseFFT::forward()
    fft_recursive(data, /* inverse = */ false);
}

void RecursiveFFT::inverse_impl(std::vector<std::complex<double>>& data) {
    if (data.empty()) return;

    fft_recursive(data, /* inverse = */ true);

    // Normalizzazione
    const size_t n = data.size();
    const double norm = 1.0 / static_cast<double>(n);
    for (auto& sample : data) {
        sample *= norm;
    }
}

void RecursiveFFT::fft_recursive(std::vector<std::complex<double>>& data, bool inverse) {
    const size_t n = data.size();
    if (n <= 1) return;

    // Divide: separa elementi di indice pari e dispari
    std::vector<std::complex<double>> even(n / 2);
    std::vector<std::complex<double>> odd(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = data[2 * i];
        odd[i] = data[2 * i + 1];
    }

    // Impera: chiamate ricorsive
    fft_recursive(even, inverse);
    fft_recursive(odd, inverse);

    // Combina
    const double pi = std::numbers::pi;
    const double angle_sign = inverse ? 1.0 : -1.0;
    const double angle = angle_sign * 2.0 * pi / static_cast<double>(n);

    std::complex<double> w(1.0, 0.0);
    const std::complex<double> wn(std::cos(angle), std::sin(angle));

    for (size_t k = 0; k < n / 2; ++k) {
        std::complex<double> t = w * odd[k];
        data[k] = even[k] + t;
        data[k + n / 2] = even[k] - t;
        w *= wn;
    }
}

} // namespace amsc_stft
