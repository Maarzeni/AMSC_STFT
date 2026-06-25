# AMSC_STFT

## Project Overview

AMSC_STFT is a C++ library for spectral analysis of audio signals, built around the Short-Time Fourier Transform (STFT). The project provides a complete pipeline that goes from reading a WAV audio file to generating a time-frequency spectrogram, with multiple FFT implementations and parallelization strategies designed for both shared-memory and distributed-memory architectures.

---

## Mathematical Background

### The Fourier Transform and Its Limitation

The Discrete Fourier Transform (DFT) decomposes a finite signal into its constituent frequencies. Given a signal $x[n]$ of length $N$, the DFT is defined as:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i 2\pi k n / N}, \quad k = 0, 1, \ldots, N-1$$

While the DFT reveals which frequencies are present in a signal, it provides no information about *when* those frequencies occur. For stationary signals this is acceptable, but for audio — where frequency content changes over time — a purely spectral representation is insufficient.

### The Short-Time Fourier Transform

The Short-Time Fourier Transform addresses this limitation by analyzing the signal locally in time. The signal is divided into short, overlapping segments (frames), a window function is applied to each frame to reduce spectral leakage, and a DFT is computed on each windowed segment:

$$X[m, k] = \sum_{n=0}^{L-1} x[n + mH] \cdot w[n] \cdot e^{-i 2\pi k n / L}$$

where $L$ is the frame length, $H$ is the hop size (controlling the overlap between consecutive frames), $w[n]$ is the window function, and $m$ is the frame index.

The result is a two-dimensional time-frequency representation: each column corresponds to a point in time, and each row to a frequency bin.

First of all the audio signal is split into overlapping frames of fixed length. The overlap between consecutive frames is controlled by the hop size $H$, typically set to $L/2$ or $L/4$. Then each frame is multiplied element-wise by a window function before applying the FFT. This reduces spectral leakage caused by the implicit discontinuities at frame boundaries. 

The project supports three window types:
   - **Hann** — good general-purpose choice; effectively eliminates edge discontinuities.
   - **Hamming** — similar to Hann but with a slightly higher sidelobe floor.
   - **Blackman** — wider main lobe but very low sidelobes; useful when high frequency selectivity is needed.

The windowed frame is transformed into the frequency domain using one of the available FFT implementations. Finally the magnitude (or log-magnitude) of each frame's FFT output is assembled into a 2D matrix, which can be exported as an image.

---

## Project Features

- **Multiple FFT implementations**: recursive (Cooley-Tukey), iterative (in-place bit-reversal), and parallel (OpenMP-accelerated).
- **Windowing functions**: Hann, Hamming, and Blackman, each following a common interface.
- **STFT analysis**: parallelized with OpenMP for shared-memory systems.
- **Distributed STFT**: MPI-based implementation that distributes frames across processes, with optional hybrid OpenMP parallelism within each process. (??AAAAAA???)
- **WAV file I/O**: loading and decoding of mono WAV audio files.
- **Spectrogram export**: generation and export of the time-frequency matrix as an image.
- **Benchmark suite**: systematic performance comparison across FFT variants, window functions, and thread/process counts.
- **Automated testing**: comprehensive test suite via GoogleTest, integrated with a GitHub Actions CI pipeline.

---

## Project Structure

```
.github/
└── workflows/
    └── main.yaml              ← GitHub Actions CI pipeline

job.sh                         ← HPC job submission script
requirements.txt               ← Python dependencies (if any)
Singularity.def                ← Container definition for HPC deployment

ft_project/
│
├── CMakeLists.txt
│
├── include/
│   ├── audio/
│   │   ├── AudioFile.h            ← External dependency
│   │   ├── AudioBuffer.hpp        ← Custom wrapper for AudioFile
│   │   └── WavReader.hpp          ← Header-only WAV reader
│   │
│   ├── fft/
│   │   ├── BaseFFT.hpp            ← CRTP base class (C++20 Concepts)
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
│   │   ├── STFTAnalyzer.hpp       ← Shared-memory STFT (OpenMP)
│   │   ├── MPI_STFTAnalyzer.hpp   ← Distributed STFT (MPI)
│   │   └── SpectrogramData.hpp
│   │
│   └── output/
│       └── ImageExporter.hpp
│
├── src/
│   ├── audio/
│   │   └── WavReader.cpp
│   ├── fft/
│   │   ├── IterativeFFT.cpp
│   │   ├── RecursiveFFT.cpp
│   │   └── ParallelFFT.cpp
│   ├── mpi/
│   │   └── MPIContext.cpp
│   └── output/
│       └── ImageExporter.cpp
│
├── tests/
│   ├── CMakeLists.txt
│   ├── data/                      ← Audio files used during testing
│   └── (test source files)
│
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── benchmark_Suite.hpp
│   ├── benchmark_Main.cpp
│   └── benchmark_MPI_Main.cpp     ← Distributed benchmark entry point
│
└── examples/
    ├── main.cpp                   ← End-to-end pipeline demonstration
    ├── mpi_main.cpp               ← MPI pipeline demonstration
    └── data/                      ← Example audio files
```

---

## Design & Architecture Notes

<!-- To be completed -->

---

## Build Instructions

```bash
git clone <repository-url>
cd ft_project

mkdir build
cd build

cmake ..
make -j
```

The build produces the `amsc_stft_core` static library. OpenMP is enabled automatically if the compiler supports it. A `compile_commands.json` file is generated in the build directory for IDE and tooling support.

---

## Running Tests

The project includes an automated test suite built with GoogleTest. Tests cover all major components of the library:

- audio file loading and decoding
- all three FFT implementations (recursive, iterative, parallel)
- window functions (Hann, Hamming, Blackman)
- STFT analysis
- spectrogram output and image export
- MPI distributed components

To run the full test suite from the build directory:

```bash
ctest --output-on-failure
```

---

## Main Tests Description

<!-- To be completed -->

---

## Running the MPI Version

To run the distributed STFT pipeline across multiple processes:

```bash
mpirun -np 4 ./mpi_main
```

For hybrid execution combining MPI (inter-node) and OpenMP (intra-node) parallelism:

```bash
OMP_NUM_THREADS=4 mpirun -np 8 ./mpi_main
```

---

## Continuous Integration and Deployment

The repository includes a GitHub Actions workflow defined in `.github/workflows/main.yaml` that implements a full CI/CD pipeline triggered on every push and pull request.

### Continuous Integration

The `ci` job runs on an Ubuntu 22.04 runner and automatically:

- installs the required HPC dependencies (CMake, OpenMP, OpenMPI) from `requirements.txt`;
- configures the project with CMake and compiles all source files and tests;
- runs the full test suite via `ctest`;
- builds a [Singularity](https://apptainer.org/) container image (`amsc_stft.sif`) from `Singularity.def`, which compiles and packages the project in an immutable environment based on Ubuntu 22.04;
- uploads the container image as a GitHub Actions artifact (retained for 7 days).

This ensures that new changes do not introduce regressions and that the project builds correctly in a clean, reproducible environment.

### Continuous Deployment on Galileo100

The `cd` job runs only after `ci` completes successfully. It deploys the container to the [Galileo100](https://www.hpc.cineca.it/systems/hardware/galileo100/) HPC cluster at CINECA and submits a SLURM job:

1. **Downloads** the `amsc_stft.sif` artifact produced by the CI stage.
2. **Connects** to the Galileo100 login node via SSH, using a private key and certificate stored as GitHub Actions secrets (`HPC_SSH_PRIVATE_KEY`, `HPC_CERT`, `HPC_USERNAME`, `HPC_SCRATCH_PATH`).
3. **Transfers** the container image and the SLURM job script (`job.sh`) to the cluster scratch directory.
4. **Submits** the job via `sbatch job.sh`.

The SLURM script (`job.sh`) requests 4 CPUs and 2 GB of memory, sets `OMP_NUM_THREADS` from the SLURM allocation, and runs `ctest` inside the container:

```bash
singularity exec --pwd /app/AMSC_STFT/ft_project/build amsc_stft.sif ctest --output-on-failure
```

Because the binary is compiled inside the immutable container during the CI stage, no build step is required on the cluster: the job runs the pre-built test suite directly.

---

## Benchmarks

The `benchmarks/` directory contains a dedicated benchmarking suite for evaluating the computational performance of the different FFT and STFT implementations. The suite measures execution time across combinations of FFT algorithms, window functions, signal lengths, and thread/process counts.

The benchmark infrastructure is intentionally decoupled from the core library, following the Single Responsibility Principle. For distributed benchmarking on HPC clusters, `benchmark_MPI_Main.cpp` serves as the entry point.

Both benchmark executables accept two optional arguments: `[reps] [warmup]` (number of timed repetitions and warm-up iterations).

### Serial / OpenMP benchmarks

Compares the FFT engines (`RecursiveFFT`, `IterativeFFT`, `ParallelFFT`) and the serial vs OpenMP STFT:

```bash
cd build

# Default run (7 repetitions, 2 warm-up iterations)
./benchmarks/benchmark_Main

# Custom repetitions / warm-up
./benchmarks/benchmark_Main 10 3

# Control the OpenMP thread count for the parallel sections
OMP_NUM_THREADS=8 ./benchmarks/benchmark_Main
```

### MPI / hybrid benchmarks

Compares pure MPI against hybrid MPI + OpenMP. Vary `-np` across runs to read the strong-scaling behaviour:

```bash
cd build

# Pure MPI: 1 OpenMP thread per rank
mpirun -np 4 ./benchmarks/benchmark_MPI_Main

# Hybrid MPI + OpenMP: multiple threads per rank
OMP_NUM_THREADS=4 mpirun -np 2 ./benchmarks/benchmark_MPI_Main

# Single-rank baseline
mpirun -np 1 ./benchmarks/benchmark_MPI_Main
```

---

## Additional Notes

- OpenMP parallelism is enabled automatically if the compiler and system support it. No manual configuration is required.
- MPI must be installed and available on the system to compile and run the distributed components (`MPI_STFTAnalyzer`, `mpi_main`, `benchmark_MPI_Main`).