# AMSC_STFT

This work focuses on developing a C++ framework for spectral analysis of audio signals based on Fourier transforms and signal processing techniques.

## Project Description

**Spectral Audio Analysis with FFT and STFT**

The project implements a complete pipeline for audio signal analysis, including:

- Fast Fourier Transform (FFT) in multiple variants: recursive, iterative, and parallel;
- Short-Time Fourier Transform (STFT), built on top of the FFT with segmentation of the audio signal into overlapping time frames;
- Windowing techniques to study spectral resolution effects, including:
  - Hann window
  - Hamming window
  - Blackman window
- Parallelized STFT computation;
- Spectral analysis of real audio signals;
- Spectrogram generation;
- Performance comparison between different implementations.

The system is able to:

- read audio files (WAV format, using external Python support or a dedicated library);
- process the signal using STFT;
- return:
  - time-frequency matrix;
  - spectral information;
  - final spectrogram as an image.

## Project Structure

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
│   │   ├── STFTAnalyzer.hpp       ← parallelized with OpenMP
│   │   ├── MPI_STFTAnalyzer.hpp   ← Distributed STFT (MPI)
│   │   └── SpectrogramData.hpp
│   │
│   ├── output/
│   │   └── ImageExporter.hpp
│   │
│   └── benchmark/
│       └── BenchmarkSuite.hpp
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
│   ├── mpi/
│   │   └── MPIContext.cpp
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
│   ├── test_ImageExporter.cpp
│   └── test_MPI_STFTAnalyzer.cpp
│
├── benchmarks/ 
│   ├── CMakeLists.txt
│   ├── bench_main.cpp
│   └── mpi_bench_main.cpp         ← Distributed benchmark entry point
│
└── examples/
    ├── main.cpp
    └── mpi_main.cpp               ← End-to-end MPI demonstration
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

*   **Distributed & Hybrid Parallelism (`MPI_STFTAnalyzer.hpp`):**  
    A distributed version of the STFT is implemented using MPI. The input signal is divided into independent frames and distributed across processes. Each process computes its local STFT, and results are gathered to form the final spectrogram.  
    Additionally, OpenMP can be used within each MPI process, enabling a hybrid parallelization strategy:
    - MPI for inter-process distribution (across nodes)
    - OpenMP for intra-process parallelism (within each node)


### Build

```bash
git clone <repository-url>
cd ft_project

mkdir build
cd build

cmake ..
make -j
```

---

### Run Tests

```bash
ctest
```

---

### Run MPI Version

```bash
mpirun -np 4 ./mpi_main
```

Example of hybrid MPI + OpenMP execution:

```bash
OMP_NUM_THREADS=4 mpirun -np 8 ./mpi_main
```

---

### Notes

- `compile_commands.json` is generated automatically for IDE support.
- OpenMP is enabled automatically if available.
- The build produces the `amsc_stft_core` static library.

## Run Tests

### Test 1 — Build System & GoogleTest Integration

To verify that the build system and the **GoogleTest (GTest)** integration are configured correctly, a baseline infrastructure test was implemented.  
This confirms that the environment can successfully:

- configure the project with CMake;
- compile the source files;
- link the `amsc_stft_core` library;
- execute the testing framework correctly.

The initial test performs a simple assertion:
- **Condition:** verifies that `2 + 2 == 4`
- **Purpose:** validates the correctness of the testing infrastructure and library linkage.

Build and run the tests with:

```bash
mkdir -p build
cd build

cmake ..
make -j

ctest --output-on-failure
```

---

### Test 2 — WAV Loading Validation

A second test was introduced to validate the audio input pipeline through the `WavReader` module.

The test uses a small mono WAV file (`test_audio.wav`) containing a 440 Hz sine wave and verifies that:

- the WAV file is loaded correctly;
- the sample rate is extracted successfully;
- audio samples are read into memory;
- invalid file paths generate the expected exception.

This test validates the complete path:

```text
WAV file → WavReader → memory buffer
```

The following checks are performed:

- **LoadValidFile**
  - verifies that the sample rate is greater than zero;
  - verifies that the loaded sample vector is not empty.

- **ThrowsOnMissingFile**
  - verifies that attempting to load a non-existent WAV file throws a `std::runtime_error`.

Run the test suite with:

```bash
cd build

ctest --output-on-failure
```

or execute the test binary directly:

```bash
./tests/test_WavReader
```