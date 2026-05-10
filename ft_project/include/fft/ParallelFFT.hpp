/**
 * @file ParallelFFT.hpp
 * @brief Multi-threaded FFT implementation using CRTP.
 */

#ifndef PARALLEL_FFT_HPP
#define PARALLEL_FFT_HPP

#include "fft/BaseFFT.hpp"
#include <vector>
#include <complex>
#include <thread>
#include <cstddef>

namespace amsc_stft {

class ParallelFFT : public BaseFFT<ParallelFFT> {
    friend class BaseFFT<ParallelFFT>;

public:
    /**
     * @brief Construct with specified thread count.
     * @param num_threads Number of threads (0 = use hardware concurrency).
     */
    explicit ParallelFFT(size_t num_threads = 0);

protected:
    // Core implementations called by BaseFFT
    void forward_impl(std::vector<std::complex<double>>& data);
    void inverse_impl(std::vector<std::complex<double>>& data);

private:
    size_t num_threads_;

    // Helper methods
    static size_t bitReverse(size_t i, size_t log_n);
    static size_t log2int(size_t n);
    
    // The parallel butterfly engine
    void butterflyPassParallel(std::vector<std::complex<double>>& data, size_t n, bool inverse) const;
};

} // namespace amsc_stft

#endif // PARALLEL_FFT_HPP