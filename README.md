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

The table below was measured on Galileo100, on two ranks over thirty seconds of synthetic audio (1 323 000 samples) at frame 1024 and hop 512. The sample counts are analytic and exact; the resident sizes are the `VmHWM` high-water marks read from `/proc/self/status` and reduced across ranks. Because that mark covers the whole lifetime of a process, each row comes from its own run:

```bash
mpirun -np 2 ./benchmarks/benchmark_MPI_Main 7 2 1024 512 30 bcast
mpirun -np 2 ./benchmarks/benchmark_MPI_Main 7 2 1024 512 30 scatter
```

| Strategy | Input samples per rank | Input MiB | Peak RSS MIN (MiB) | Peak RSS MAX (MiB) | Peak RSS AVG (MiB) |
|---|---|---|---|---|---|
| broadcast | 1 323 000 | 10.09 | 35.30 | 55.57 | 45.44 |
| scatter | 661 504 | 5.05 | 30.35 | 55.68 | 43.01 |

The input footprint now falls as `N/P`, and the measured peak follows it: the lighter rank drops by 4.95 MiB against the 5.04 MiB the analytic column predicts, an agreement within two per cent. The maximum does not move — 55.57 MiB against 55.68, a tenth of a per cent — which is the intended effect of `MPI_IN_PLACE` on the root, and it is worth being explicit about why the maximum is the root's. The root rank reads the signal, so it holds *N* samples under either strategy, and it is also the rank that assembles the output; the scatter changes what the other *P − 1* ranks must hold, not what the reader holds. Subtracting the input from the peak leaves the same residue on both rows, about 25.3 MiB of process baseline — the binary, the MPI runtime, the OpenMP thread pools — which is why halving a thirty-second signal shows up as a fourteen per cent reduction of the process rather than a fifty per cent one. That baseline is a constant; the input is not, and it is the input that the 1.27 GB per rank of an hour of audio is made of.

The same comparison in the development container, at four ranks over ten minutes of audio, shows the block dividing by four as expected: 201.9 MiB of input per rank under the broadcast against 50.5 MiB under the scatter, and a peak of 276.3 MiB against 124.6 MiB on the lightest rank, with the maximum unchanged at 680.6 MiB in both cases. The halo behaves as designed in both settings. In the Galileo100 run the two blocks overlap by exactly 512 samples, `frameSize - hopSize`, while the last 504 samples of the signal are read by no frame at all, so the two blocks together carry 1 323 008 samples against the signal's 1 323 000.

What neither run shows is the case the change is actually about. Two ranks on one node with a thirty-second signal is a scaled-down check: it confirms the model to within two per cent, and it is that agreement which licenses extrapolating the model to a signal that does not fit in the memory of a single node. Demonstrating the latter directly needs the production partition and a considerably longer input; `RANKS="1 2 4 8" DURATION=600 ./benchmarks/run_scaling.sh` on `g100_usr_prod` is the natural next step.

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

Besides the timings, the run reports the per-rank memory footprint: an analytic table of the input samples each rank has to hold under both distribution strategies, and the measured peak resident size taken from `VmHWM`. A sixth argument selects the strategy (`scatter`, the default, or `bcast`), which is how the before/after comparison in [Distributing the signal: broadcast versus scatter](#distributing-the-signal-broadcast-versus-scatter) was produced. The peak is a whole-process high-water mark, so the two strategies have to be run as two separate processes for the figure to be attributable to either of them.

### Results on Galileo100

The numbers discussed here come from a single SLURM job on Galileo100 (job 21727805, 12 August 2026) running the Singularity image built by the CI pipeline, so they were produced by exactly the binaries the pipeline ships. The allocation was two ranks of two threads, four cores of an Intel Xeon Platinum 8260 at 2.4 GHz, on the `g100_all_serial` partition. All fifteen tests pass there, the four rank counts of the distributed STFT included, and the repetition-to-repetition standard deviation stays below half a per cent of the mean in every row, which is what makes differences of a few per cent readable at all.

#### What a frame costs

The serial STFT costs 119.5 µs per frame at every workload measured: 10.161 ms for 85 frames, 51.228 for 429, 102.659 for 860 and 308.663 for 2582, which is linear to four significant figures. A single forward `IterativeFFT` of the same size takes 111 µs, so the transform accounts for about 93 per cent of the cost of a frame, and everything around it — the copy with zero padding, the window, the packing into complex values, the magnitude — for the remaining seven. Two things follow from that. Optimising anything other than the FFT is not worth the effort at this geometry, and a frame is a large enough unit of work to be worth distributing on its own, which is the assumption both the OpenMP loop and the MPI decomposition rest on.

| Transform size | RecursiveFFT | IterativeFFT | ParallelFFT (4 threads) |
|---|---|---|---|
| 1 024 | 0.135 | 0.111 | 0.048 |
| 262 144 | 54.237 | 46.471 | 12.360 |

*(fastest of seven repetitions, in milliseconds)*

The same table settles a design question stated in `STFTAnalyzer.hpp`: whether to parallelise the frame loop and keep each transform serial, or to parallelise the transforms internally. At 1024 points `ParallelFFT` on four threads gains 2.80× over the iterative engine, whereas parallelising the frame loop with the same four threads gains 3.98×. A 1024-point transform is simply too small to keep four threads busy, and frame-level parallelism wins by enough — some forty per cent — that nesting the two is not worth revisiting. The picture would change for much larger frames, where `ParallelFFT` reaches 4.32× on its own at 65 536 points.

#### Shared memory, distributed memory, and the two together

| Workload | serial, 1 thread | OpenMP, 4 threads | MPI, 2 ranks | hybrid, 2 ranks × 2 threads |
|---|---|---|---|---|
| 85 frames | 10.161 | 2.656 | 5.273 | 2.765 |
| 429 frames | 51.228 | 12.980 | 28.068 | 15.252 |
| 860 frames | 102.659 | 25.790 | 55.706 | 29.933 |
| 2582 frames | 308.663 | 77.578 | 165.125 | 87.625 |

*(fastest of seven repetitions, in milliseconds, at frame 1024 and hop 512)*

Reading the last row: four OpenMP threads turn 308.7 ms into 77.6, a speedup of 3.98× at 99.5 per cent efficiency; two MPI ranks turn it into 165.1, which is 1.87× at 93 per cent; the two together, on the same four cores, give 87.6 ms, 3.52× at 88 per cent. The ranking holds at every workload and it is the expected one. Threads share the signal and the output buffer and pay only for the fork and join of the parallel region, whereas ranks have to be sent their input and have their results gathered back.

The cost of that distribution can be given a number, with a caveat about how firm the number is. If the two ranks scaled perfectly the 2582-frame case would take 154.3 ms, and it takes 165.1, so about seven per cent of the time is spent outside the computation; the same estimate over the four workloads gives four, ten, nine and seven per cent. Those figures are best read as a band rather than as a trend, for two reasons: they compare two different executables, the serial baseline coming from `benchmark_Main` and the distributed one from `benchmark_MPI_Main`, and the same MPI-pure configuration measured in two separate job steps differed by 3.4 per cent (165.1 ms against 159.7 at 2582 frames). A cleaner strong-scaling read is what `run_scaling.sh` exists for, since it compares `-np 1` against larger rank counts within a single binary.

The practical conclusion for a single node is that MPI does not buy speed there — pure OpenMP is between 4 and 18 per cent faster than the hybrid at equal core count — and that it is not meant to. What the distributed layer buys is the ability to use more than one node and, since the scatter, the ability to hold a signal larger than the memory of one node. On a single node the hybrid configuration is nevertheless the sensible way to run the distributed binary: at 1.85× to 1.90× over pure MPI, the second thread per rank recovers nearly everything the process-level split gives away.

One expectation stated in the benchmark's own notes is not borne out by the data. The hybrid gain does not improve as the frame count grows: it is 1.90× at 85 frames and 1.88× at 2582, flat within the noise. Eighty-five frames already amount to some ten milliseconds of work per rank, far more than an OpenMP region costs to open and close, so there is nothing left to amortise. The serial-versus-OpenMP table does show the expected ramp, from 3.83× to 3.98×, and it saturates just as early.

The two distribution strategies were also timed against each other at 2582 frames, back to back within the same job step, which is what makes that particular comparison a fair one. The scatter is slightly faster: 159.7 ms against 161.4 with one thread per rank, and 82.3 against 84.0 in hybrid mode, so one to two per cent. That is the expected sign and the expected magnitude. A broadcast has to deliver *N* samples to each of the *P* ranks while the scatter delivers about *N* in total, so at two ranks the saving amounts to half of a ten-megabyte transfer set against a run of 160 ms, and the gap should widen with the rank count. The reason for the change was never speed, but it is worth recording that it costs none.

The memory side of the same job is discussed in [Distributing the signal: broadcast versus scatter](#distributing-the-signal-broadcast-versus-scatter), where the measured drop matches the analytic prediction to within two per cent. Two limits of this campaign are worth stating plainly. The `g100_all_serial` partition runs on a login node shared with other users, so these are not exclusive-node timings, even if their stability across repetitions suggests the sharing did not perturb them; and two ranks on a single node cannot demonstrate memory scalability at the scale that motivated it, only confirm the model that predicts it.

---

## Additional Notes

- OpenMP parallelism is enabled automatically if the compiler and system support it. No manual configuration is required.
- MPI must be installed and available on the system to compile and run the distributed components (`MPI_STFTAnalyzer`, `mpi_main`, `benchmark_MPI_Main`).