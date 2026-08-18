# AMSC_STFT

## Project Overview

AMSC_STFT is a C++ library for spectral analysis of audio signals, built around the Short-Time Fourier Transform (STFT). The project provides a complete pipeline that goes from reading a WAV audio file to generating a time-frequency spectrogram, with multiple FFT implementations and parallelization strategies designed for both shared-memory and distributed-memory architectures.

---

## What the library computes

The Discrete Fourier Transform tells you *which* frequencies a signal contains, but not *when* they occur — a limitation that makes it unusable for audio, whose spectral content changes continuously. The Short-Time Fourier Transform removes it by applying the DFT locally: the signal is split into short overlapping frames, each frame is multiplied by a window function that suppresses spectral leakage, and a DFT is computed on the result.

$$X[m, k] = \sum_{n=0}^{L-1} x[n + mH] \cdot w[n] \cdot e^{-i 2\pi k n / L}$$

Here $L$ is the frame length, $H$ the hop size (the shift between consecutive frames, hence their overlap), $w[n]$ the window and $m$ the frame index. The output is a time-frequency matrix — one row per instant, one column per frequency band — whose magnitudes are exported as a spectrogram image.

Two parameters control the analysis, and they do very different things.

**The frame length $L$ sets both resolutions at once, in opposite directions.** A frame lasts $L/f_s$ seconds, and the STFT produces a single spectrum for the whole frame — so anything that happens inside it cannot be located more precisely than that. At the same time, the DFT of $L$ points produces bins that are $f_s/L$ Hz apart, and two tones closer than that fall into the same bin. Since $L$ sits on top in one formula and underneath in the other, their product is always 1: you cannot improve both. Choosing $L$ only decides how the fixed budget is split between time and frequency.

| $L$ | Time resolution | Frequency resolution |
|---|---|---|
| 256 | 5.8 ms | 172 Hz |
| 1024 (default) | 23.2 ms | 43.1 Hz |
| 4096 | 92.9 ms | 10.8 Hz |

**The hop size $H$ changes neither.** A smaller hop does not make the frames shorter — it just takes more of them, overlapping. What $H$ controls is coverage and cost. With $H = L$ the frames only touch, and the window fades out whatever falls near their edges; with $H = L/2$ (the default) every instant lands inside some frame. The price is linear: since the frame count is $M = 1 + \lfloor (N - L)/H \rfloor$, halving the hop doubles the number of frames, the number of FFTs to compute, and the size of the output matrix.

### Three things to know before running it

- **The frame size must be a power of two.** All three FFT implementations are radix-2 Cooley-Tukey, which requires it. A frame size that is not a power of two is rejected: `main.cpp` exits with an error message, and `STFTAnalyzer` throws `std::invalid_argument` at construction. The hop size has no such constraint.

- **Three windows are available**, selected with the fourth command-line argument: `hann` (the default, a good general-purpose choice), `hamming` (narrower main lobe, suited to speech), `blackman` (lowest sidelobes, for precision spectral analysis).

- **The output has `frameSize / 2 + 1` columns, not `frameSize`.** The input is real, so the upper half of the spectrum is redundant and is not stored. The retained bins run from DC to Nyquist, and bin $k$ sits at frequency $k \cdot f_s / L$ — use `SpectrogramData::binFrequency(k)` rather than computing it by hand. Magnitudes are normalised by $L \cdot \text{coherentGain}(w)$, so values are comparable across frame sizes and window choices.

> Section 1 of the project report covers the reasoning behind all of this: the time-frequency uncertainty relation, where spectral leakage comes from, how the three windows trade off against each other, why half the spectrum suffices, and how the normalisation is derived.

---

## Project Features

- **Multiple FFT implementations**: recursive (Cooley-Tukey), iterative (in-place bit-reversal), and parallel (OpenMP-accelerated).
- **Windowing functions**: Hann, Hamming, and Blackman, each following a common interface.
- **STFT analysis**: parallelized with OpenMP for shared-memory systems.
- **Distributed STFT**: MPI-based implementation that distributes frames across processes, sending each rank only the samples its own frames read, with optional hybrid OpenMP parallelism within each process.
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

system-deps.txt                ← apt packages for the C++/MPI toolchain
Singularity.def                ← Container definition for HPC deployment

scripts/                       ← Everything that RUNS the project
├── run_suite.sh               ← The benchmark suite: one recipe, every machine
├── job.sh                     ← SLURM submission (Galileo100), wraps run_suite.sh
└── job_mox.pbs                ← PBS submission (MOX cluster), same suite

analysis/                      ← Everything that READS the results
├── parse_results.py           ← Text tables → tidy CSV (standard library only)
├── plot_results.py            ← CSV → figures for the report (matplotlib)
└── requirements.txt           ← pip packages for the above (matplotlib)

core/
│
├── CMakeLists.txt
│
├── include/
│   ├── audio/
│   │   ├── AudioFile.h            ← External dependency
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
│   ├── mpi/
│   │   └── MPIContext.hpp         ← RAII wrapper for MPI_Init / MPI_Finalize
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
│   ├── data/                      ← Audio fixtures. Only test_audio.wav is
│   │                                 versioned; the rest is generated output
│   │                                 and stays out of git (see .gitignore)
│   └── (test source files)
│
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── benchmark_Suite.hpp
│   ├── benchmark_Main.cpp
│   └── benchmark_MPI_Main.cpp     ← Distributed benchmark entry point
│
└── examples/
    ├── CMakeLists.txt
    ├── main.cpp                   ← End-to-end pipeline demonstration
    └── mpi_main.cpp               ← MPI pipeline demonstration
```

## Getting Started

### Requirements

| Requirement | Notes |
|---|---|
| C++20 compiler | GCC 11+ or Clang 14+ |
| CMake ≥ 3.20 | |
| OpenMP | `find_package(OpenMP REQUIRED)` — configuration fails without it |
| MPI | `find_package(MPI REQUIRED)` — OpenMPI is what we use |
| Network access on first configure | GoogleTest v1.14.0 is fetched from GitHub by `FetchContent` |

Install the Debian/Ubuntu packages listed in `system-deps.txt`:

```bash
cat system-deps.txt | grep -v '^#' | xargs sudo apt-get install -y
```

CMake checks the compiler version itself and stops with a readable message if it is too old for C++20, rather than failing deep inside a standard header. The first `cmake` also needs the network access noted in the table above: GoogleTest is not an apt package, it is fetched from GitHub the first time the project is configured.

### Build

From the root of the repository:

```bash
cd core
mkdir build && cd build
cmake ..
make -j
```

### What gets built

The example programs, the tests and the benchmark binaries are all run from `core/build/`; the scripts under `scripts/` and `analysis/` are run from the repository root.

| Program | Path | What it does |
|---|---|---|
| `main` | `examples/main` | WAV → spectrogram PNG, OpenMP |
| `mpi_main` | `examples/mpi_main` | the same pipeline, distributed over MPI ranks |
| `benchmark_Main` | `benchmarks/benchmark_Main` | serial and OpenMP timings |
| `benchmark_MPI_Main` | `benchmarks/benchmark_MPI_Main` | MPI and hybrid timings |
| `test_*` | `tests/test_*` | the GoogleTest executables, also registered with CTest |

The build also produces the `amsc_stft_core` and `amsc_stft_mpi` static libraries and a `compile_commands.json` for IDE support.

### Optional: the analysis scripts

Only needed to turn benchmark output into figures:

```bash
python3 -m pip install -r analysis/requirements.txt   # matplotlib
```

`analysis/parse_results.py` uses the standard library only and needs no installation.

---

## Running the Example Programs

The library ships two demonstration programs. `main` is the complete pipeline on a single machine: it reads a WAV file, computes the STFT and writes the spectrogram as a PNG. `mpi_main` is the same analysis distributed over MPI processes, and prints its result to the terminal instead of writing an image.

The repository contains one audio file, `core/tests/data/test_audio.wav`, which the commands below use — any mono WAV works in its place.

### `main` — WAV to spectrogram

```
./examples/main <audio.wav> [frameSize=1024] [hopSize=512] [window=hann|hamming|blackman]
```

The WAV path is required, the rest is optional. `frameSize` must be a power of two. The image is written next to the input file, as `<input>_spectrogram.png`.

```bash
cd core/build

# Defaults: 1024-sample frames, 50 % overlap, Hann window
./examples/main ../tests/data/test_audio.wav
# → core/tests/data/test_audio_spectrogram.png

# Finer frequency resolution and a different window
./examples/main ../tests/data/test_audio.wav 2048 512 hamming

# The frame loop is parallelised with OpenMP
OMP_NUM_THREADS=4 ./examples/main ../tests/data/test_audio.wav
```

### `mpi_main` — distributed analysis

```
mpirun -np <P> ./examples/mpi_main [audio.wav]
```

The WAV path is the only argument; frame size 1024, hop 512 and the Hann window are fixed in the program. Given no argument — or a file it cannot read — it analyses a 2-second synthetic signal instead, so it always has something to run on. It prints the spectrogram dimensions, the first bins of the first frame and the number of ranks used; it does not write a PNG.

```bash
cd core/build

# Pure MPI
mpirun -np 4 ./examples/mpi_main ../tests/data/test_audio.wav

# Hybrid: MPI across processes, OpenMP inside each one
OMP_NUM_THREADS=4 mpirun -np 2 ./examples/mpi_main ../tests/data/test_audio.wav

# No input file: built-in synthetic signal
mpirun -np 4 ./examples/mpi_main
```

Add `--oversubscribe` if you ask for more ranks than the machine has cores.

---

## Running the Tests

The project ships a GoogleTest suite registered with CTest. From `core/build`:

```bash
ctest --output-on-failure    # run everything
ctest -N                     # list the tests without running them
```

### Running a single test

`-R` selects the tests whose name contains what you pass it. `-V` prints the full GoogleTest output instead of showing it only on failure.

```bash
ctest -R STFTAnalyzerTest --output-on-failure
ctest -R MPISTFTTest --output-on-failure     # all four rank counts
```

Each test is also a standalone executable under `tests/`, which gives access to the GoogleTest flags:

```bash
./tests/test_STFTAnalyzer --gtest_list_tests
./tests/test_STFTAnalyzer --gtest_filter='STFTAnalyzerTemplateTest.*'
```

The CTest names (`STFTAnalyzerTest`) and the GoogleTest suite names inside the sources (`NumFramesTest`, `SpectrogramDataTest`, …) are independent: the first go with `ctest -R`, the second with `--gtest_filter`.

### Running tests in parallel

```bash
# Several test cases at once
ctest -j4 --output-on-failure

# More cores to one test (ParallelFFTTest and STFTAnalyzerTest use OpenMP)
OMP_NUM_THREADS=4 ctest -R STFTAnalyzerTest --output-on-failure
OMP_NUM_THREADS=4 ./tests/test_STFTAnalyzer

# A rank count other than the registered 1, 2, 3 and 4
mpirun -n 6 --oversubscribe ./tests/test_MPI_STFTAnalyzer
```

Keep `-j` times `OMP_NUM_THREADS` within the available cores: `ctest -j4` with `OMP_NUM_THREADS=4` asks for sixteen threads and will slow a four-core machine down rather than speed it up.

---

## Benchmarks

The `benchmarks/` directory measures the execution time of the FFT engines and of the STFT under each parallelisation strategy. There are two executables — one for the shared-memory side, one for the distributed side — and a script that runs both in a fixed configuration and writes the results to disk.

### Serial / OpenMP benchmarks

Compares the FFT engines (`RecursiveFFT`, `IterativeFFT`, `ParallelFFT`) and the serial against the OpenMP STFT.

```
./benchmarks/benchmark_Main [reps=7] [warmup=2] [frame=1024] [hop=frame/2] [seconds]
```

By default the STFT is measured on 1, 5, 10 and 30 seconds of audio; giving `seconds` replaces that sweep with a single duration.

```bash
cd core/build

# Defaults
./benchmarks/benchmark_Main

# 10 repetitions, 3 warm-up iterations
./benchmarks/benchmark_Main 10 3

# 8192-sample frames, half overlap, 20 seconds of audio
./benchmarks/benchmark_Main 7 2 8192 4096 20

# The thread count for the parallel rows
OMP_NUM_THREADS=8 ./benchmarks/benchmark_Main
```

### MPI / hybrid benchmarks

```
mpirun -np <P> ./benchmarks/benchmark_MPI_Main [reps=7] [warmup=2] [frame=1024] [hop=frame/2] [seconds] [strategy=scatter|bcast]
```

Each run reports three configurations of the same workload — a serial baseline, pure MPI, and hybrid MPI + OpenMP — with both parallel rows expressed as speedups over the baseline. The baseline is the root rank computing the whole spectrogram alone on one thread, with no scatter and no gather, so baseline and parallel rows share the executable, the allocation, the signal and the repetition count. Divide a speedup by the number of cores in use to get the efficiency. Alongside the timings the run prints the per-rank memory footprint: the input samples each rank holds under either distribution strategy, and the measured peak resident size from `VmHWM`.

```bash
cd core/build

# Pure MPI: 1 OpenMP thread per rank
mpirun -np 4 ./benchmarks/benchmark_MPI_Main

# Hybrid MPI + OpenMP: multiple threads per rank
OMP_NUM_THREADS=4 mpirun -np 2 ./benchmarks/benchmark_MPI_Main

# Single-rank baseline
mpirun -np 1 ./benchmarks/benchmark_MPI_Main

# 30 seconds of audio on two ranks, broadcast instead of scatter
mpirun -np 2 ./benchmarks/benchmark_MPI_Main 7 2 1024 512 30 bcast
```

Vary `-np` across runs to read the strong-scaling behaviour. The memory figure is a whole-process high-water mark, so the two distribution strategies have to be compared across two separate runs.

### The full benchmark suite

`scripts/run_suite.sh` is the single entry point that produces every result file: the test run, both benchmark executables, the memory comparison, and the strong-, thread- and weak-scaling sweeps. The CI, the SLURM job and a person at a terminal all call it, so their numbers are comparable by construction. Run it from the repository root:

```bash
# Against the build produced above
bash scripts/run_suite.sh

# A specific configuration
RANKS=4 THREADS=1 PREFIX=laptop bash scripts/run_suite.sh
SCALING_RANKS="1 2 4 8" SCALING_DURATIONS="5 30" bash scripts/run_suite.sh

# On a cluster, against the Singularity image the CI builds from Singularity.def
apptainer exec --bind "$PWD:$PWD" amsc_stft.sif bash scripts/run_suite.sh
```

Everything is configured through environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `BUILD_DIR` | in-image path, else `core/build` | where the binaries are |
| `RESULTS_DIR` | `./results/raw` | where result files are written |
| `PREFIX` | `cluster` / `github` / `local` | file-name prefix, i.e. which machine |
| `RANKS`, `THREADS` | 2, 2 | MPI ranks and OpenMP threads per rank |
| `BENCH_ARGS` | `"7 2 1024 512"` | reps, warm-up, frame, hop |
| `MEM_DURATION` | 30 | seconds of audio for the memory comparison |
| `SCALING_RANKS` | powers of two up to the core count | rank list of the strong-scaling sweep |
| `SCALING_DURATIONS` | `"5 15 60"` | one scaling curve per duration |
| `THREAD_LIST` | powers of two up to the core count | thread list of the OpenMP sweep |
| `WEAK_BASE` | 5 | seconds of audio per rank in the weak-scaling sweep |
| `SCALING`, `THREAD_SCALING`, `WEAK_SCALING` | on | set any to `0` to skip that sweep |

The results land in `results/raw/` as `<prefix>_env.txt`, `<prefix>_ctest.txt`, `<prefix>_benchmark_openmp.txt`, `<prefix>_benchmark_mpi.txt`, `<prefix>_memory_bcast_vs_scatter.txt`, `<prefix>_scaling.txt`, `<prefix>_scaling_threads.txt` and `<prefix>_weak_scaling.txt`. Two scripts turn them into tables and figures:

```bash
python3 analysis/parse_results.py results/raw/ -o results/csv/
python3 analysis/plot_results.py --csv results/csv/ --out results/figures/
```

`parse_results.py` takes one or more result directories — `results/raw/*/` when several machines have been collected side by side — and produces `timings.csv` and `memory.csv`. It needs nothing but the standard library, so it can run on a cluster where installing packages is not possible. `plot_results.py` needs matplotlib and writes one figure per topic, in PDF and PNG, with the plotted numbers beside each.

### Where we ran it

The suite was run on two HPC systems, in both cases through the scripts in `scripts/`:

| System | Submission | Configuration |
|---|---|---|
| **Galileo100** (CINECA) | `sbatch scripts/job.sh`, from the directory holding `amsc_stft.sif` | inside the Singularity image built by the CI, on the `g100_all_serial` partition |
| **MOX** (MOX laboratory, Politecnico di Milano) | `qsub scripts/job_mox.pbs`, from `/work/$USER/AMSC_STFT` | native build with `gcc@15.2.0` and `openmpi@5.0.8`, up to 28 cores on the `cpu` queue |

Running Galileo100 through the container means the timed binaries are exactly the ones the pipeline ships, and the MOX job reuses the same `run_suite.sh` with a wider sweep, so the two machines produce directly comparable files. The measurements from both, and the discussion of what they show about the shared-memory, distributed and hybrid strategies, are in the project report included in this repository.

---

## Continuous Integration and Deployment

The repository includes a GitHub Actions workflow defined in `.github/workflows/main.yaml` that implements a full CI/CD pipeline triggered on every push and pull request.

### Continuous Integration

The `ci` job runs on an Ubuntu 22.04 runner and automatically:

- installs the required HPC dependencies (CMake, OpenMP, OpenMPI) from `system-deps.txt`;
- configures the project with CMake and compiles all source files and tests;
- runs the full test suite via `ctest`;
- runs `scripts/run_suite.sh` and uploads the measurements as the `benchmark-results-github` artifact;
- builds a [Singularity](https://apptainer.org/) container image (`amsc_stft.sif`) from `Singularity.def`, which compiles and packages the project in an immutable environment based on Ubuntu 22.04;
- uploads the container image as the `amsc-stft-sif` artifact (retained for 7 days).

This ensures that new changes do not introduce regressions and that the project builds correctly in a clean, reproducible environment.

### Continuous Deployment on Galileo100

The `cd` job runs only after `ci` completes successfully. It deploys the container to the [Galileo100](https://www.hpc.cineca.it/systems/hardware/galileo100/) HPC cluster at CINECA and submits a SLURM job:

1. **Downloads** the `amsc_stft.sif` artifact produced by the CI stage.
2. **Connects** to the Galileo100 login node via SSH, using a private key and certificate stored as GitHub Actions secrets (`HPC_SSH_PRIVATE_KEY`, `HPC_CERT`, `HPC_USERNAME`, `HPC_SCRATCH_PATH`).
3. **Transfers** the container image and the whole `scripts/` directory to the cluster scratch directory — the job script and the benchmark suite it calls travel together, so they can never arrive out of step.
4. **Submits** the job via `sbatch --wait scripts/job.sh`.
5. **Copies the results back** and uploads them as the `benchmark-results-cluster` artifact.

The SLURM script (`scripts/job.sh`) requests 2 tasks of 2 CPUs and 4 GB of memory on the `g100_all_serial` partition and runs the benchmark suite inside the container; `run_suite.sh` reads `SLURM_NTASKS` and `SLURM_CPUS_PER_TASK` to size its sweeps to that allocation:

```bash
singularity exec ${BIND} --pwd "${SLURM_SUBMIT_DIR}" amsc_stft.sif bash scripts/run_suite.sh
```

Because the project is compiled inside the immutable container during the CI stage, no build step is required on the cluster: the job runs the pre-built binaries directly.

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
