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

The commands below assume they are run **inside the development container** (the same Ubuntu 22.04 environment described by `Singularity.def`), not on the host machine. The container provides the toolchain the project requires: `CMakeLists.txt` declares `find_package(OpenMP REQUIRED)` and `find_package(MPI REQUIRED)`, so configuration aborts on any system where an OpenMP-capable compiler or an MPI installation is missing. Building inside the container also keeps the local build consistent with the CI image and with the cluster deployment.

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

### Running a single test

The `-R` option filters tests by name, using a regular expression. Anchoring the pattern avoids partial matches:

```bash
ctest -R '^STFTAnalyzerTest$' --output-on-failure
```

`ctest -N` lists the registered tests without executing them, and `-V` prints the full GoogleTest output instead of showing it only on failure. Each test is also a standalone executable under `build/tests/`, which can be run directly to access the GoogleTest command-line flags:

```bash
./tests/test_STFTAnalyzer --gtest_list_tests
./tests/test_STFTAnalyzer --gtest_filter='STFTAnalyzerTemplateTest.*'
```

Note that the CTest name (`STFTAnalyzerTest`) and the GoogleTest suite names defined inside the source file (`NumFramesTest`, `STFTAnalyzerConstructionTest`, `SpectrogramDataTest`, …) are independent: the former is used with `ctest -R`, the latter with `--gtest_filter`.

### Running tests in parallel

Two distinct forms of parallelism apply here, and they are controlled separately.

**Running several tests at once.** The `-j` option tells CTest how many *test cases* to execute concurrently. It shortens the wall time of the whole suite, but gives no additional core to any individual test:

```bash
ctest -j4 --output-on-failure
```

**Giving more cores to one test.** The tests that exercise the shared-memory implementations (`ParallelFFTTest`, `STFTAnalyzerTest`) are parallelised internally with OpenMP, so their thread count comes from the `OMP_NUM_THREADS` environment variable:

```bash
OMP_NUM_THREADS=4 ctest -R '^STFTAnalyzerTest$' --output-on-failure
OMP_NUM_THREADS=4 ./tests/test_STFTAnalyzer
```

The two options can be combined, but the product of the two must stay within the available cores: `ctest -j4` with `OMP_NUM_THREADS=4` requests up to sixteen threads and will oversubscribe a four-core machine, making the run slower rather than faster.

**The distributed test.** `MPISTFTTest` is registered with a fixed number of two processes, since the process count is written into the `add_test` command in `tests/CMakeLists.txt`. Running it through CTest therefore always uses two ranks; a different configuration requires invoking `mpirun` on the executable directly:

```bash
mpirun -n 4 --oversubscribe ./tests/test_MPI_STFTAnalyzer
OMP_NUM_THREADS=2 mpirun -n 2 --oversubscribe ./tests/test_MPI_STFTAnalyzer
```

Finally, keep in mind that these are correctness tests running on small signals: raising the thread or process count is meant to verify that the results remain correct as the configuration changes, not to obtain a speed-up. Thread creation overhead dominates at this size. Actual performance measurements belong to the benchmark suite described below.

---

## Main Tests Description

### Numerical validation of the FFT

The correctness of the fast transforms is validated at several levels:

- **Analytical reference cases** (`test_RecursiveFFT`, `test_IterativeFFT`): known input/output pairs such as the Dirac impulse $[1,0,0,0] \rightarrow [1,1,1,1]$ and a constant (DC) signal, whose transform is known in closed form.
- **Forward/inverse round-trip**: a multi-frequency synthetic signal is transformed and then inverse-transformed; the reconstruction must match the original within `1e-9`, verifying that `forward` and `inverse` are mutually consistent.
- **Direct DFT comparison** (`test_DFTReference`): on small sizes ($N = 4, 8, 16$) the output of `RecursiveFFT` and `IterativeFFT` is compared element-by-element against a naive $O(N^2)$ Discrete Fourier Transform,

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i 2\pi k n / N},$$

  computed directly from the definition. The direct DFT is trivially correct by construction and independent of the divide-and-conquer / bit-reversal logic, so it provides an authoritative reference for the fast implementations (both forward and the normalized inverse). The same test also cross-checks the recursive and iterative engines against each other.

### Time-frequency validation

- **Synthetic signals with known frequencies** (`test_STFTAnalyzer`): a bin-aligned sinusoid produces its spectral peak at the expected frequency bin, and a DC signal concentrates its energy in bin 0, confirming that framing, windowing and the FFT are wired together correctly.
- **Multi-tone analysis** (`test_STFTAnalyzer`): a signal built as the sum of three bin-aligned sinusoids with distinct amplitudes ($k = 8, 32, 96$ — i.e. $\approx 344$, $1378$ and $4134$ Hz at 44.1 kHz, with amplitudes $1.0$, $0.6$, $0.3$) is used to verify simultaneously that
  - all three peaks appear at the expected bins as strict local maxima, with the remaining bins essentially silent;
  - the recovered magnitude matches the expected amplitude. With the normalization applied by the analyzer — division by $\text{frameSize} \cdot \text{coherentGain} = \sum_n w[n]$ — a real sinusoid of amplitude $A$ yields a one-sided peak of $A/2$;
  - the *relative* amplitudes between components are preserved, so windowing and normalization do not distort the spectral balance;
  - the peaks are stable across every frame, as expected for a stationary signal;
  - the peak bins map back to the correct physical frequencies via `binFrequency()`.

### Structural and parallel tests

- **Window functions** (`test_HannWindow`, `test_HammingWindow`, `test_BlackmanWindow`): coefficient values and symmetry properties.
- **C++20 concept validation** (`test_BaseFFT`): compile-time check that each FFT engine satisfies the `IsFFT` concept.
- **Distributed STFT** (`test_MPI_STFTAnalyzer`): the MPI result is checked for consistency with the serial `STFTAnalyzer` output.

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

### Distributing the signal: broadcast versus scatter

The first version of `MPI_STFTAnalyzer` distributed the work but not the data. The root rank broadcast two scalars, the signal length and the sample rate, so that every rank could derive the frame layout on its own; it then broadcast the entire sample array with a single `MPI_Bcast`; each rank computed the frames of its assigned block, and `MPI_Gatherv` reassembled the magnitude matrix on the root. The design is simple for a good reason. STFT frames are independent, so the only thing a rank needs in order to compute frame *f* is the samples that frame reads, and handing every rank the whole signal makes the question of which samples those are disappear entirely: there are no block boundaries to get right, there is one collective to reason about, and every rank can address any frame by its global index.

What that design does not do is scale in memory. Every rank ends up holding all *N* samples regardless of how many ranks take part, so the per-rank input footprint is independent of *P*. One hour of mono audio at 44.1 kHz is roughly 159 million samples, which is about 1.27 GB once stored as `double`, and that figure is exactly the same on four ranks as on four hundred. Adding nodes therefore buys time to solution but never capacity: the longest signal the program can analyse remains the longest that fits in the memory of a single node. The strategy is appropriate for signals of moderate size, and it is the large ones it cannot follow.

The default strategy now replaces the broadcast with an `MPI_Scatterv` that sends each rank only the samples its own frames actually read. Rank *r* owns frames `[start_r, start_r + count_r)`, which span

```
offset_r = start_r * hopSize
length_r = (count_r - 1) * hopSize + frameSize
```

so a rank receives approximately `N/P` samples plus a halo of `frameSize - hopSize`: the tail of its block that the following block also needs in order to complete its own first frame. Consecutive send blocks consequently overlap, which `MPI_Scatterv` permits because it only ever reads the send buffer; the same overlap among the displacements of `MPI_Gatherv` would instead be a write race. The halo is a constant fixed by the frame geometry alone, so it does not grow with either the signal or the rank count: at the default 1024/512 geometry it is 512 samples, four kilobytes, set against blocks of tens of megabytes.

`STFTAnalyzer` required no modification to operate on a slice. `analyzeRange` is called with `startFrame = 0`, and the offset it computes internally, `frameIdx * hopSize`, is then already relative to the beginning of the slice, which is `offset_r` in global coordinates. Ranks that receive no frames at all — a signal shorter than one frame, or simply fewer frames than ranks — are given a send count of zero and still participate in the collective, so no configuration deadlocks. The root rank is the one exception to the slice-relative scheme: it passes `MPI_IN_PLACE` and keeps addressing its frames inside the full signal it already holds, because receiving its own block into a second buffer would add another `N/P` to the peak of the rank that is already the largest. Both strategies remain selectable through the `Distribution` parameter of the constructor, which defaults to `Distribution::Scatter`, so the comparison below can be reproduced rather than merely reported.

The table was produced with `benchmark_MPI_Main` on ten minutes of synthetic audio (26 460 000 samples) at frame 1024 and hop 512. The sample counts are analytic and exact; the resident sizes are the `VmHWM` high-water marks read from `/proc/self/status`, reduced across ranks. Because that mark covers the whole lifetime of a process, each row comes from its own run:

```bash
mpirun -np 4 ./benchmarks/benchmark_MPI_Main 1 0 1024 512 600 bcast
mpirun -np 4 ./benchmarks/benchmark_MPI_Main 1 0 1024 512 600 scatter
```

| Ranks | Strategy | Input samples per rank | Input MiB | Peak RSS MIN (MiB) | Peak RSS MAX (MiB) | Peak RSS AVG (MiB) |
|---|---|---|---|---|---|---|
| 2 | broadcast | 26 460 000 | 201.9 | 327.2 | 731.6 | 529.4 |
| 2 | scatter | 13 230 080 | 100.9 | 226.3 | 731.6 | 479.0 |
| 4 | broadcast | 26 460 000 | 201.9 | 276.3 | 680.6 | 377.4 |
| 4 | scatter | 6 615 296 | 50.5 | 124.6 | 680.6 | 263.8 |

The sample column is the average over the ranks, which is exact for the broadcast and for the two-rank scatter; at four ranks the blocks differ by a single hop, between 6 615 040 and 6 615 552 samples, because 51 678 frames do not divide evenly.

The input footprint now falls as `N/P`, and the measured peaks follow it: at four ranks the lightest rank drops from 276 MiB to 125 MiB, a saving of 151 MiB that matches the 151.4 MiB the analytic column predicts. The maximum does not move, and it is worth being explicit about why. The root rank reads the signal, so it holds *N* samples under either strategy, and it is also the rank that assembles the output; the scatter changes what the other *P − 1* ranks must hold, not what the reader holds. These numbers were measured in the Ubuntu 22.04 development container on a laptop, where the absolute values include the container's own baseline; the differences between the rows, which is what the table is about, are unaffected. Re-running the two commands above on Galileo100 reproduces the comparison at cluster scale.

The gather side is deliberately unchanged, and that is where the remaining ceiling lies. The root rank still allocates the complete `totalFrames × numBins` magnitude matrix. At the default geometry that matrix stores `numBins / hopSize`, or 513/512, doubles for every input sample, which makes it very slightly *larger* than the signal it was computed from: the same hour of audio that occupies 1.27 GB as input yields about 1.27 GB of magnitudes on the root, on top of the signal the root read and the copy `analyze()` takes by value. The bottleneck has therefore moved from the input to the output rather than disappeared, and for a long enough recording it is the output that decides whether the run fits in memory. Removing it would mean not assembling the matrix at all — each rank writing its own range of frames directly through MPI-IO, or streaming block by block into the image exporter — which is beyond the scope of this change. A related limit sits in the same place: `MPI_Scatterv` and `MPI_Gatherv` express their counts and displacements as `int`, so both sides overflow above 2³¹ elements, around 13.5 hours of 44.1 kHz audio at this geometry. That limit is documented in the code rather than worked around.


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

### Setting Up SSH Authentication for Galileo100

```bash
# 1. Generate the certificate (choose a temporary passphrase, then log in on the
#    browser page that opens: CINECA username + password + Google Authenticator OTP).
#    Produces ~/.ssh/cineca_key and ~/.ssh/cineca_key-cert.pub
step ssh certificate 'your_email@mail.polimi.it' ~/.ssh/cineca_key --provisioner cineca-hpc

# 2. Remove the passphrase (old = the temporary one, new = empty: press Enter twice)
ssh-keygen -p -f ~/.ssh/cineca_key

# 3. Test the connection (must not ask for a password)
ssh your_username@login.g100.cineca.it \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o BatchMode=yes \
  -i ~/.ssh/cineca_key

# 4. Print the values to paste into the GitHub secrets
cat ~/.ssh/cineca_key
cat ~/.ssh/cineca_key-cert.pub
```

Then, under **Settings → Secrets and variables → Actions**, create:

| Secret | Value |
|---|---|
| `HPC_SSH_PRIVATE_KEY` | content of `~/.ssh/cineca_key` |
| `HPC_CERT` | content of `~/.ssh/cineca_key-cert.pub` |
| `HPC_USERNAME` | CINECA username (e.g. `mcolombo`) |
| `HPC_SCRATCH_PATH` | scratch directory (e.g. `/gpfs/scratch/userspace/your_username`) |

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