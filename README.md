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
│   └── output/
│       └── ImageExporter.hpp
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
│   └── output/
│       └── ImageExporter.cpp
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
|   ├── benchmark_Suite.hpp
│   ├── benchmark_Main.cpp
│   └── benchmark_MPI_Main.cpp         ← Distributed benchmark entry point
│
└── examples/
    ├── main.cpp
    └── mpi_main.cpp               ← End-to-end MPI demonstration
```

### Design & Architecture Notes
* **Compile-Time Polymorphism (`BaseFFT.hpp`):**
  Instead of traditional runtime polymorphism (virtual dispatch, abstract base
  classes), all FFT implementations use the **Curiously Recurring Template
  Pattern (CRTP)**. `BaseFFT.hpp` defines the static interface via CRTP, 
  guaranteeing zero-overhead dispatch and early, descriptive compile-time 
  diagnostics for library users.

  Because concrete types are always fully known at compile time, there is no 
  polymorphic base pointer to manage — and therefore no architectural need for 
  owning pointers to a base type such as:

```cpp
  std::unique_ptr<BaseFFT>
  std::shared_ptr<BaseFFT>
```

  Note that this is a consequence of the static-dispatch design, not a 
  rejection of ownership semantics in general.

*   **Windowing as Callable Objects (`window/`):**
    All window classes (`HannWindow`, `HammingWindow`, `BlackmanWindow`) are **functors** — stateful objects that overload `operator()`. This means a window can be invoked directly as a callable:

    ```cpp
    HannWindow w(1024);
    w(frame);          // functor call syntax  (new)
    w.apply(frame);    // explicit method call  (still works)
    ```

    **Why callable objects?** The windows were already stateful function-like objects (holding precomputed coefficients); making `operator()` explicit is the idiomatic C++ step that lets them participate as first-class callables alongside free functions and lambdas — exactly the interchangeability described in the lecture on callable objects.

    **Advantages over `apply()`-only:**
    - More natural call syntax matching the mathematical notion of *applying* w to a frame.
    - Can be passed to any generic/STL context that accepts a callable (e.g. `std::function`, template parameters, algorithms) without wrappers.
    - Easier extensibility: a future `STFTAnalyzer` can accept `any` callable window — a lambda, a custom struct, or one of the provided classes — through the same template parameter.
    - **Zero overhead**: `operator()` is a one-line inline delegation to `apply()`; no type erasure, no extra allocation, no performance change.

    The `WindowFunction` C++20 concept has been extended to require the callable interface alongside the existing spectral-metadata requirements (`coherentGain()`, `powerBandwidth()`), so that generic STFT code can rely on call syntax while also accessing normalization data:

    ```cpp
    template<stft::WindowFunction W>
    void processFrame(W& window, std::vector<double>& frame) {
        window(frame);                         // apply window
        auto gain = window.coherentGain();     // use metadata for normalization
    }
    ```

    **How it integrates in the STFT pipeline:** each overlapping frame is windowed by invoking `window(frame)` immediately before the FFT. The same window object provides `coherentGain()` and `powerBandwidth()` for spectral normalization of the output.

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

*   **Decoupling Benchmarks:** 
    To adhere to the Single Responsibility Principle, the benchmarking suite has been decoupled from the core library and moved to a standalone benchmarks/ directory. End-users download amsc_stft_core for high-performance signal processing, not to run developer performance tests. Embedding benchmark code inside the core library needlessly inflates compilation times and binary size.


### CI/CD Pipeline

A GitHub Actions pipeline defined in [.github/workflows/main.yaml](.github/workflows/main.yaml) runs automatically on every **push** and **pull request**.

It consists of two sequential jobs:

#### 1. `ci` — Continuous Integration (GitHub-hosted runner, `ubuntu-22.04`)

| Step | Action |
|------|--------|
| Install dependencies | Reads `requirements.txt` and installs CMake, OpenMP, OpenMPI via `apt-get` |
| Build | Configures and compiles the project with `cmake` + `make -j` |
| Test | Executes the full test suite via `ctest --output-on-failure` |
| Build container | Installs Apptainer and builds the Singularity image `amsc_stft.sif` from `Singularity.def` |
| Upload artifact | Saves `amsc_stft.sif` as a workflow artifact (retained 7 days) |

#### 2. `cd` — Continuous Deployment (runs only if `ci` passes)

| Step | Action |
|------|--------|
| Download artifact | Retrieves `amsc_stft.sif` built in the previous job |
| Configure SSH | Sets up key-based access to the Galileo100 cluster using repository secrets (`HPC_SSH_PRIVATE_KEY`, `HPC_CERT`) |
| Deploy | Copies `amsc_stft.sif` and `job.sh` to the HPC scratch directory via `scp` |
| Submit job | Connects via SSH to `login.g100.cineca.it` and submits `job.sh` with `sbatch` |

The deployment requires three repository secrets to be configured: `HPC_USERNAME`, `HPC_SSH_PRIVATE_KEY`, `HPC_CERT`, and `HPC_SCRATCH_PATH`.

---

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

---

### Test 3 — C++20 Concepts Validation

A third test was introduced to validate the **C++20 Concepts** mechanism used for compile-time polymorphism in the FFT implementations.

The test verifies that the `IsFFT` concept correctly enforces the interface contract for FFT classes. The concept requires that any valid FFT implementation must provide:

- `forward(std::vector<std::complex<double>>&)` method
- `inverse(std::vector<std::complex<double>>&)` method

The test includes two mock classes:

- **ValidMockFFT**
  - satisfies all requirements of the `IsFFT` concept;
  - verifies that the concept correctly accepts a valid implementation.

- **InvalidMockFFT**
  - intentionally lacks the `inverse()` method;
  - verifies that the concept correctly rejects an invalid implementation.

Both checks are performed using `static_assert` at compile-time:

- `static_assert(IsFFT<ValidMockFFT>, ...)` — confirms that valid FFTs pass the concept check.
- `static_assert(!IsFFT<InvalidMockFFT>, ...)` — confirms that incomplete implementations are rejected.

If the program compiles successfully, all compile-time checks have passed. A runtime test also verifies the integrity of the testing infrastructure.

Run the test with:

```bash
cd build

ctest --output-on-failure -R BaseFFTTest
```

or execute the test binary directly:

```bash
./tests/test_BaseFFT
```

---

### Test 4 — Recursive FFT Validation

A fourth test suite was introduced to validate the correctness and robustness of the `RecursiveFFT` implementation.

The test verifies both the mathematical properties of the FFT algorithm and the correctness of error handling.

The following scenarios are covered:

- **HandlesEmptyInput**
  - verifies that an empty input vector throws a `std::invalid_argument` exception;
  - ensures that the input vector remains unchanged.

- **ThrowsOnNonPowerOfTwo**
  - verifies that FFT execution fails when the input size is not a power of two;
  - checks both `forward()` and `inverse()` methods.

- **ImpulseResponse**
  - validates the Fourier transform of a Dirac impulse;
  - verifies that `[1, 0, 0, 0]` transforms into `[1, 1, 1, 1]`.

- **ForwardInverseIdentity**
  - verifies the reversibility property of the FFT:

```text
IFFT(FFT(x)) == x
```

  - generates a mixed sinusoidal signal composed of multiple frequencies;
  - applies FFT followed by inverse FFT;
  - verifies that the reconstructed signal matches the original signal within floating-point tolerance.

- **ConstantSignal**
  - validates the DC component behavior;
  - verifies that a constant signal `[2, 2, 2, 2]` produces:

```text
[8, 0, 0, 0]
```

  - confirms that the energy is concentrated at frequency bin 0.

The test suite also includes a helper utility for comparing complex vectors using floating-point tolerance checks.

Run the Recursive FFT test with:

```bash
cd build

ctest --output-on-failure -R RecursiveFFTTest
```

or execute the test binary directly:

```bash
./tests/test_RecursiveFFT
```

---

### Test 5 — Iterative FFT Validation

A fifth test suite was introduced to validate the `IterativeFFT` implementation.

This test ensures that the iterative FFT algorithm produces correct results and behaves consistently with the expected mathematical properties of the Fast Fourier Transform.

The validation process includes:

- input validation;
- power-of-two size verification;
- impulse response correctness;
- forward/inverse transform consistency;
- constant signal analysis;
- floating-point precision verification.

The following scenarios are tested:

- **HandlesEmptyInput**
  - verifies that empty input vectors generate a `std::invalid_argument` exception.

- **ThrowsOnNonPowerOfTwo**
  - verifies that FFT execution is rejected when the input size is not a power of two.

- **ImpulseResponse**
  - validates the transform of a discrete impulse signal.

- **ForwardInverseIdentity**
  - verifies that applying FFT followed by inverse FFT reconstructs the original signal correctly.

- **ConstantSignal**
  - validates the expected DC-frequency behavior for constant input signals.

The iterative implementation is particularly important because it avoids recursive function calls and is generally more cache-efficient for large datasets.

Run the Iterative FFT test with:

```bash
cd build

ctest --output-on-failure -R IterativeFFTTest
```

or execute the test binary directly:

```bash
./tests/test_IterativeFFT
```

---

### Test 6 — Parallel FFT & OpenMP Validation

A sixth test suite was introduced to validate the **multi-threaded implementation** of the FFT, particularly focusing on correct OpenMP parallelization.

This test is divided into functional correctness checks and parallel safety validation. The suite tests different thread-count configurations to ensure consistency across variable core allocations.

The following scenarios are covered:

- **CorrectnessDifferentThreadCounts**
  - validates mathematically correct results utilizing `1, 2, 4, 8` threads;
  - ensures different system configurations do not modify mathematical outcomes.

- **LargeDataParallelProcessing**
  - executes transformation on a very large array (65K elements);
  - stresses the multi-threading loop overhead and evaluates data chunking mechanisms.

- **RoundTripMultipleThreads**
  - confirms reversible data conversion (Forward ↔ Inverse) across all active thread scales.

- **ThreadSafety**
  - triggers heavy concurrency testing simulating 8 isolated concurrent computations;
  - ensures data races and false-sharing do not impact the transformation result.

- **InvalidSizeCheckParallel**
  - verifies that runtime safety boundaries still correctly trigger exceptions even inside threaded scenarios.

- **HybridParallelismSimulation**
  - splits transformations similarly to node-based communication (MPI style);
  - computes sub-blocks with dynamic OpenMP scheduling.

Run the Parallel FFT OpenMP test with:

```bash
cd build

# To test with a specific number of threads (e.g. 4)
OMP_NUM_THREADS=4 ctest --output-on-failure -R ParallelFFTOpenMPTest
```

or execute the test binary directly to observe the custom output and benchmark:

```bash
# Set 8 threads dynamically
OMP_NUM_THREADS=8 ./tests/test_ParallelFFT_OpenMP

# Run the serial baseline
OMP_NUM_THREADS=1 ./tests/test_ParallelFFT_OpenMP
```