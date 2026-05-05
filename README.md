# AMSC_STFT

This work focuses on implementing one of the most commonly used time-frequency decomposition tools: the Short-Time Fourier Transform (STFT).

```text
ft_project/
│
├── CMakeLists.txt
│
├── include/
│   └── audio/
│       ├── AudioFile.h            ← copied from lib, do not modify
│       ├── AudioBuffer.hpp        ← your wrapper around AudioFile
│       └── WavReader.hpp          ← header-only, minimal code
│
│   ├── fft/
│   │   ├── FFTConcept.hpp
│   │   ├── IterativeFFT.hpp
│   │   ├── RecursiveFFT.hpp
│   │   └── ParallelFFT.hpp
│   │
│   ├── window/
│   │   ├── WindowConcept.hpp
│   │   ├── HannWindow.hpp
│   │   ├── HammingWindow.hpp
│   │   └── BlackmanWindow.hpp
│   │
│   ├── stft/
│   │   ├── STFTAnalyzer.hpp
│   │   └── SpectrogramData.hpp
│   │
│   ├── output/
│   │   └── ImageExporter.hpp
│   │
│   └── benchmark/
│       ├── BenchmarkSuite.hpp
│       └── BenchmarkResult.hpp
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
│       └── BenchmarkSuite.cpp
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
│   └── bench_main.cpp
│
└── examples/
    └── main.cpp
