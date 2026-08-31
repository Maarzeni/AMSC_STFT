# AMSC_STFT

## Project Overview

AMSC_STFT is a C++ library for spectral analysis of audio signals, built around the Short-Time Fourier Transform (STFT). The project provides a complete pipeline that goes from reading a WAV audio file to generating a time-frequency spectrogram, with multiple FFT implementations and parallelization strategies designed for both shared-memory and distributed-memory architectures.

---

## The mathematical problem

The **Short-Time Fourier Transform (STFT)** describes how the frequency content of an audio signal changes over time. The signal is split into short overlapping frames; each frame is multiplied by a window function $w[n]$ that tapers smoothly to zero at its edges, and a **Discrete Fourier Transform (DFT)** is computed on the result. The DFT decomposes a length-$L$ frame into $L$ frequency components; for efficiency it is evaluated with the **Fast Fourier Transform (FFT)**, which lowers the cost from $O(L^2)$ to $O(L\log L)$.

$$X[m, k] = \sum_{n=0}^{L-1} x[n + mH] \cdot w[n] \cdot e^{-i 2\pi k n / L}$$

Here $L$ is the frame length, $H$ the **hop size** (the shift between consecutive frames, hence their overlap), $w[n]$ the window and $m$ the frame index. The output is a time–frequency matrix — row $m$ an instant in time, column $k$ a frequency band — and $|X[m,k]|$ is the strength of that frequency at that instant, exported as the brightness of a pixel in the spectrogram.

The frame length $L$ fixes both resolutions at once and in opposite directions: a frame spans $\Delta t = L/f_s$ seconds and its bins are $\Delta f = f_s/L$ Hz apart, so $\Delta t \cdot \Delta f = 1$ for any $L$. The window limits **spectral leakage**, the smearing of a frequency's energy into neighbouring bins caused by cutting a frame out. Each window is characterised by its **coherent gain** (the mean of its coefficients, used to normalise the magnitude) and its **equivalent noise bandwidth (ENBW)**, its effective width in frequency bins.

### Three things to know before running it

- **The frame size must be a power of two.** All three FFT implementations are radix-2 Cooley-Tukey, which requires it. A frame size that is not a power of two is rejected: `main.cpp` exits with an error message, and `STFTAnalyzer` throws `std::invalid_argument` at construction. The hop size has no such constraint.

- **Three windows are available**, selected with the fourth command-line argument: `hann` (the default, a good general-purpose choice), `hamming` (narrower main lobe, suited to speech), `blackman` (lowest sidelobes, for precision spectral analysis).

- **The output has `frameSize / 2 + 1` columns, not `frameSize`.** The input is real, so the upper half of the spectrum is redundant and is not stored. The retained bins run from constant componenta at 0 Hz to Nyquist (the maximum reprensentable frequency: $f_s/2$), and bin $k$ sits at frequency $k \cdot f_s / L$. Magnitudes are normalised by $L \cdot \text{coherentGain}(w)$, so values are comparable across frame sizes and window choices.

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

## Repository Layout

The repository is organised into four main areas: the C++ library, the scripts
used to run it, the tools used to analyse the results, and the files generated
during experiments. This separation keeps the source code independent from the
execution environment and from the benchmark analysis. The main structure of the
repository is shown below.

**Top level**

```text
AMSC_STFT/
|
|-- core/            <- C++ library
|   |-- include/     <- audio, fft, window, mpi, stft, output
|   |-- src/         <- implementations
|   |-- tests/       <- automated tests
|   |-- benchmarks/  <- performance suite
|   +-- examples/    <- demo programs
|
|-- scripts/         <- run_suite.sh, SLURM job
|-- analysis/        <- Python parsing and plotting
|-- .github/         <- CI pipeline
|-- Singularity.def  <- HPC container
+-- results/         <- generated output (not in git)
```

**Library headers** — `core/include/`

```text
include/
|-- audio/
|   |-- AudioFile.h
|   +-- WavReader.hpp
|-- fft/
|   |-- BaseFFT.hpp
|   |-- IterativeFFT.hpp
|   |-- RecursiveFFT.hpp
|   +-- ParallelFFT.hpp
|-- window/
|   |-- BaseWindow.hpp
|   |-- HannWindow.hpp
|   |-- HammingWindow.hpp
|   +-- BlackmanWindow.hpp
|-- mpi/
|   +-- MPIContext.hpp
|-- stft/
|   |-- STFTAnalyzer.hpp
|   |-- MPI_STFTAnalyzer.hpp
|   +-- SpectrogramData.hpp
+-- output/
    +-- ImageExporter.hpp
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

CMake checks the compiler version itself and stops with a readable message if it is too old for C++20. The first `cmake` also needs the network access noted in the table above: GoogleTest is not an apt package, it is fetched from GitHub the first time the project is configured.

Optional: `doxygen` and, for its dependency graphs, Graphviz's `dot` — needed only for [the API documentation](#optional-the-api-documentation) below (`sudo apt install doxygen graphviz`).

### Build

From the root of the repository:

```bash
cd core
mkdir build && cd build
cmake ..
make
```

### What gets built

The example programs, the tests and the benchmark binaries are all run from `core/build/`; the scripts under `scripts/` and `analysis/` are run from the repository root.

| Program | Path | What it does |
|---|---|---|
| `main` | `examples/main` | WAV → spectrogram PNG, OpenMP |
| `mpi_main` | `examples/mpi_main` | the same pipeline and the same output, distributed over MPI ranks |
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

### Optional: the API documentation

Generates `docs/doxygen/html/` from the Doxygen comments in `core/include/`
and `core/src/`, with this README as its front page. Needs `doxygen`; `dot`
(from Graphviz) is optional and adds include/dependency graphs if present.

```bash
doxygen Doxyfile
python3 -m http.server 8000 --directory docs/doxygen/html
```

Then open <http://localhost:8000/>. The generated HTML is not tracked by git;
only `Doxyfile` is.

---

## Running the Example Programs

The library ships two demonstration programs, and they are deliberately the same program twice. `main` runs the pipeline on one machine, with the frame loop parallelised by OpenMP; `mpi_main` runs it across MPI processes, each of which still uses OpenMP inside. Both read a WAV file, compute the STFT, write the spectrogram as a PNG and print the same report. The two source files are written to be read side by side: same arguments, same helpers in the same order, same output — the real difference is `STFTAnalyzer` against `MPI_STFTAnalyzer`.

### The command line, identical for both

```
./examples/main                    [audio.wav] [frameSize=1024] [hopSize=512] [window=hann|hamming|blackman]
mpirun -np <P> ./examples/mpi_main [audio.wav] [frameSize=1024] [hopSize=512] [window=hann|hamming|blackman]
```

Every argument is optional:

- **`audio.wav`** — a path, or a bare file name, which is looked up in `core/examples/data/`. That directory holds the two versioned audio files, `test_audio.wav` (1 second, 44.1 kHz mono, a 440 Hz tone) and `scale.wav` (2 seconds, same format); any mono WAV works in their place. With no argument at all, `test_audio.wav` is analysed. If the file cannot be read, both programs fall back to a 2-second synthetic 440 + 880 + 1760 Hz signal rather than exiting, so they always have something to run on.
- **`frameSize`** — must be a power of two, or the program stops with an error.
- **`hopSize`** — any value ≥ 1.
- **`window`** — `hann`, `hamming` or `blackman`.

### Where the output goes

Both programs write their PNG into `results/results_examples/`, alongside the benchmark output under `results/`:

```
results/results_examples/
├── test_audio_hann_f1024_h512_serial.png    ← from main
└── test_audio_hann_f1024_h512_mpi4.png      ← the same analysis, from mpi_main on 4 ranks
```

The name carries the parameters — `<input>_<window>_f<frameSize>_h<hopSize>_<serial|mpi P>` — so runs with different settings accumulate side by side, and the serial and distributed results of the same analysis sit next to each other for comparison.

Everything else is printed to the terminal: sample rate, duration, frame and hop, window, the frame and bin counts, the time and frequency resolutions they imply, the dominant frequency of the middle frame and the wall time of the analysis.

The destination is `$STFT_EXAMPLES_DIR` if it is set, otherwise the repository path baked in at build time; if that is not writable — the source tree inside the Singularity image is read-only — the programs fall back to `./results/results_examples` relative to the current directory.

### `main` — shared memory

```bash
cd core/build

# No arguments: the bundled test_audio.wav, 1024-sample frames, 50 % overlap, Hann
./examples/main

# A different file, geometry and window; the frame loop is OpenMP-parallel
OMP_NUM_THREADS=4 ./examples/main scale.wav 2048 512 hamming
```

### `mpi_main` — distributed

The same command line with `mpirun` in front. The frames are block-distributed over the ranks and the root rank gathers the full spectrogram, so it is the one that writes the PNG and prints the report.

```bash
cd core/build

# Pure MPI: one thread per rank
mpirun -np 4 ./examples/mpi_main test_audio.wav

# Hybrid: MPI across processes, OpenMP inside each one
OMP_NUM_THREADS=4 mpirun -np 2 ./examples/mpi_main test_audio.wav 2048 512 hamming
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

`-R` selects the tests whose name contains what you pass it; each test is also
a standalone executable under `tests/`, which gives access to the GoogleTest
flags directly (`--gtest_list_tests`, `--gtest_filter=...`).

```bash
ctest -R STFTAnalyzerTest --output-on-failure
./tests/test_STFTAnalyzer --gtest_filter='STFTAnalyzerTemplateTest.*'
```

### Running tests in parallel

```bash
ctest -j4 --output-on-failure                          # several tests at once
OMP_NUM_THREADS=4 ctest -R STFTAnalyzerTest --output-on-failure  # more cores to one
```

Keep `-j` times `OMP_NUM_THREADS` within the available cores: `ctest -j4` with `OMP_NUM_THREADS=4` asks for sixteen threads and will slow a four-core machine down rather than speed it up.

---

## Benchmarks

The `benchmarks/` directory measures the execution time of the FFT engines and of the STFT under each parallelisation strategy. There are two executables — one for the shared-memory side, one for the distributed side.

### Serial / OpenMP benchmarks

Three comparisons, all against the same serial baseline:

- **FFT engines** — `RecursiveFFT`, `IterativeFFT`, `ParallelFFT` on a single transform of increasing size.
- **STFT, serial vs OpenMP** — `STFTAnalyzer<IterativeFFT, HannWindow>` with one thread against all available threads; the OpenMP parallelism is the frame loop.
- **Parallelism granularity** — the same thread budget spent two ways: across frames (Parallelism::Frames with IterativeFFT) versus inside each frame's transform (Parallelism::Transform with ParallelFFT).

```
./benchmarks/benchmark_Main [reps=7] [warmup=2] [frame=1024] [hop=frame/2] [seconds]
```

By default the STFT comparisons are measured on 1, 5, 10 and 30 seconds of audio; giving `seconds` replaces that sweep with a single duration.

```bash
cd core/build

OMP_NUM_THREADS=8 ./benchmarks/benchmark_Main                  # defaults
./benchmarks/benchmark_Main 7 2 8192 4096 20                    # coarser frames, 20 s
```

### MPI / hybrid benchmarks

```
mpirun -np <P> ./benchmarks/benchmark_MPI_Main [reps=7] [warmup=2] [frame=1024] [hop=frame/2] [seconds] [strategy=scatter|bcast]
```

Each run reports three configurations of the same workload — a serial baseline, pure MPI, and hybrid MPI + OpenMP — with both parallel rows carrying their speedup over the baseline. The baseline is the root rank computing the whole spectrogram alone on one thread, with no scatter and no gather, so baseline and parallel rows share the executable, the allocation, the signal and the repetition count. Divide a speedup by the number of cores in use to get the efficiency. Alongside the timings the run reports the per-rank memory footprint: the input samples each rank holds under either distribution strategy, and the measured peak resident size from `VmHWM`.

```bash
cd core/build

mpirun -np 4 ./benchmarks/benchmark_MPI_Main                          # pure MPI
OMP_NUM_THREADS=4 mpirun -np 2 ./benchmarks/benchmark_MPI_Main        # hybrid
mpirun -np 2 ./benchmarks/benchmark_MPI_Main 7 2 1024 512 30 bcast    # broadcast, not scatter
```

Vary `-np` across runs to read the strong-scaling behaviour. The memory figure is a whole-process high-water mark, so the two distribution strategies have to be compared across two separate runs.

### The full benchmark suite

`scripts/run_suite.sh` is the single entry point that produces every result file. The CI, the SLURM job and a person at a terminal all call it, so their numbers are comparable by construction. Run it from the repository root:

```bash
bash scripts/run_suite.sh                                    # against the build above
RANKS=4 THREADS=1 PREFIX=laptop bash scripts/run_suite.sh     # a specific configuration

# On a cluster, against the Singularity image the CI builds from Singularity.def
apptainer exec --bind "$PWD:$PWD" amsc_stft.sif bash scripts/run_suite.sh
```

Everything is configured through environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `BUILD_DIR` | in-image path, else `core/build` | where the binaries are |
| `RESULTS_DIR` | `./results/results_benchmark` | where benchmark CSVs and the environment log are written |
| `TEST_RESULTS_DIR` | `./results/results_test` | where the ctest log and its working copy are written |
| `PREFIX` | `cluster` / `github` / `local` | file-name prefix, i.e. which machine |
| `RANKS`, `THREADS` | 2, 2 | MPI ranks and OpenMP threads per rank |
| `BENCH_ARGS` | `"7 2 1024 512"` | reps, warm-up, frame, hop |
| `MEM_DURATION` | 30 | seconds of audio for the memory comparison |
| `SCALING_RANKS` | powers of two up to the core count | rank list of the strong-scaling sweep |
| `SCALING_DURATIONS` | `"5 15 60"` | one scaling curve per duration |
| `THREAD_LIST` | powers of two up to the core count | thread list of the OpenMP sweep |
| `WEAK_BASE` | 5 | seconds of audio per rank in the weak-scaling sweep |
| `SCALING`, `THREAD_SCALING`, `WEAK_SCALING` | on | set any to `0` to skip that sweep |

The results land in `results/results_benchmark/` as `<prefix>_env.txt`, `<prefix>_benchmark_openmp.csv`, `<prefix>_benchmark_mpi.csv`, `<prefix>_memory_bcast_vs_scatter.csv`, `<prefix>_scaling.csv`, `<prefix>_scaling_threads.csv` and `<prefix>_weak_scaling.csv` — the `.txt` file is a plain log, the `.csv` ones are the benchmarks' own output, several invocations appended into one file per topic. `<prefix>_ctest.txt` and ctest's own working copy land separately, in `results/results_test/`. Two scripts turn the benchmark CSVs into tidy CSV and figures, both under `results/results_analysis/` by default — no path needed for the common case:

```bash
python3 analysis/parse_results.py
python3 analysis/plot_results.py
```

`parse_results.py` reads `results/results_benchmark/` by default; pass one or more directories explicitly when several machines' output has been collected side by side elsewhere. It tags every row with its machine and which sweep produced it, and writes `timings.csv` and `memory.csv`. It needs nothing but the standard library, so it can run on a cluster where installing packages is not possible. `plot_results.py` needs matplotlib and writes one PNG per figure.

---

## Continuous Integration and Deployment

A GitHub Actions pipeline (`.github/workflows/main.yaml`) runs on every push and pull request: it builds the project and runs the full test suite and benchmarks on the GitHub runner, then packages everything into a container and repeats the same run on the Galileo100 HPC cluster at CINECA. Results from both environments are kept as workflow artifacts.