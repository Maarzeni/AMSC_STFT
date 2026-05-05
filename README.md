# AMSC_STFT

This work focuses on implementing one of the most commonly used time-frequency decomposition tools: the Short-Time Fourier Transform (STFT).

### Project Structure

```text
ft_project/
│
├── CMakeLists.txt
│
├── include/
│   ├── audio/
│   │   ├── AudioFile.h            ← External dependency (do not modify)
│   │   ├── AudioBuffer.hpp        ← Custom wrapper for AudioFile
│   │   └── WavReader.hpp          ← Header-only minimal WAV reader
│   │
│   ├── fft/
│   │   ├── BaseFFT.hpp            ← CRTP base class for FFT implementations
│   │   ├── IterativeFFT.hpp
│   │   ├── RecursiveFFT.hpp
│   │   └── ParallelFFT.hpp
│   │
│   ├── window/
│   │   ├── BaseWindow.hpp
│   │   ├── HannWindow.hpp
│   │   ├── HammingWindow.hpp
│   │   └── BlackmanWindow.hpp
│   │
│   ├── stft/
│   │   ├── STFTAnalyzer.hpp       ← Parallelized template implementation
│   │   └── SpectrogramData.hpp
│   │
│   ├── output/
│   │   └── ImageExporter.hpp
│   │
│   └── benchmark/
│       └── BenchmarkSuite.hpp     ← HPC cluster-ready benchmark definitions
│
├── src/
│   ├── audio/
│   │   └── WavReader.cpp
│   │
│   ├── fft/
│   │   ├── IterativeFFT.cpp
│   │   ├── RecursiveFFT.cpp
│   │   └── ParallelFFT.cpp
│   │
│   ├── output/
│   │   └── ImageExporter.cpp
│   │
│   └── benchmark/
│       └── BenchmarkSuite.cpp     ← Execution logic for performance tests
│
├── tests/
│   ├── CMakeLists.txt
│   ├── test_AudioBuffer.cpp
│   ├── test_WavReader.cpp
│   ├── test_IterativeFFT.cpp
│   ├── test_RecursiveFFT.cpp
│   ├── test_ParallelFFT.cpp
│   ├── test_HannWindow.cpp
│   ├── test_HammingWindow.cpp
│   ├── test_BlackmanWindow.cpp
│   ├── test_STFTAnalyzer.cpp
│   └── test_ImageExporter.cpp
│
├── benchmarks/ 
│   ├── CMakeLists.txt
│   └── bench_main.cpp             ← Main entry point for the benchmark suite
│
└── examples/
    └── main.cpp                   ← End-to-end demonstration of the library
```

### Design & Architecture Notes

*   **Compile-Time Polymorphism (`BaseFFT.hpp`):** 
    Instead of relying on traditional runtime polymorphism (abstract base classes and virtual functions), the core FFT implementations utilize the **Curiously Recurring Template Pattern (CRTP)**. `BaseFFT.hpp` acts as the concept interface. This design choice ensures zero-overhead static dispatch and yields significantly clearer compile-time error messages, greatly improving the developer experience for future users of the library.

*   **Template Design & Parallelization (`STFTAnalyzer.hpp`):** 
    The STFT analysis tool is implemented as a heavily templated class to accommodate various underlying data types. It is explicitly designed to leverage parallel computing, ensuring efficient processing of large audio signals.

*   **Cluster-Ready Benchmarking (`benchmark/` & `bench_main.cpp`):** 
    The benchmark module is dedicated to rigorously evaluating the computational time of various combinations of FFT algorithms and windowing functions. The architecture of `BenchmarkSuite` has been specifically tailored to be deployed and executed on a High-Performance Computing (HPC) cluster. `bench_main.cpp` serves as the centralized orchestrator for triggering these performance tests.

*   **Library Demonstration (`examples/main.cpp`):**
    Provides a comprehensive, end-to-end practical application of the library. It acts as both a testbed and an accessible tutorial for end-users, showcasing the complete pipeline from reading an audio file to generating and exporting the STFT spectrogram.